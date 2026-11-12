"""Durable job control. No OpenAI import, credential read, or paid work here."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import uuid

from .server_config import (MAX_DOSSIER_BYTES, ResearchError, ServerConfig,
                            ensure_state, read_private_json, timestamp,
                            validate_snapshot)
from .signal_ledger import Ledger, LedgerBusy, LedgerError


RECEIPT_FIELDS = (
    "phase", "session_id", "turn_id", "submitted_at", "updated_at", "last_event",
    "result_status", "submission_attempts", "failure_code", "usage", "session_usage",
    "latest_usage_unknown", "latest_session_usage_unknown", "latest_spend_unknown",
    "usage_scope", "session_usage_scope", "usage_is_provisional", "session_usage_is_provisional",
)
PUBLIC_JOB_FIELDS = (
    "job_id", "consumer", "request_id", "status", "model", "selected_count", "pending_count",
    "pending_remaining", "total_count", "researched_count", "coverage", "input_coverage",
    "created_at", "updated_at", "failure_phase", "failure_message",
)


def checked_uuid(value) -> str:
    try:
        parsed = str(uuid.UUID(value))
        if parsed != value:
            raise ValueError()
        return parsed
    except (TypeError, ValueError, AttributeError) as exc:
        raise ResearchError("invalid_id", "Use the exact canonical UUID returned by the service.", 422) from exc


class Controller:
    def __init__(self, config: ServerConfig):
        self.config = config
        ensure_state(config)
        self.ledger = Ledger(config.db_path)

    def worker_status(self) -> dict:
        try:
            state = read_private_json(self.config.state_dir / "worker-status.json", 4096)
            age = (datetime.now(timezone.utc) - timestamp(state["observed_at"])).total_seconds()
            fresh = -5 <= age <= 30
            phase = state.get("phase")
            if phase not in ("idle", "running", "recovering", "recovery_required", "configuration_error", "stopped"):
                fresh = False
                phase = "unavailable"
            return {"phase": phase, "fresh": fresh,
                    "ready": self.config.api_enabled and fresh and phase in ("idle", "running")}
        except (ResearchError, KeyError, TypeError):
            return {"phase": "unavailable", "fresh": False, "ready": False}

    def _run_path(self, job: dict) -> Path | None:
        if not job.get("run_dir"):
            return None
        expected = self.config.state_dir.absolute() / "runs" / checked_uuid(job["job_id"]) / "pilot"
        if Path(job["run_dir"]).absolute() != expected:
            raise ResearchError("state_integrity", "Saved research run is outside its job directory.")
        return expected

    def _receipt(self, job: dict) -> dict | None:
        path = self._run_path(job)
        if path is None or not job.get("run_id") or not (path / "receipt.json").exists():
            return job.get("receipt")
        result = read_private_json(path / "receipt.json")
        if not isinstance(result, dict) or result.get("run_id") != job.get("run_id"):
            raise ResearchError("state_integrity", "Saved research receipt does not match the job.")
        return result

    def public_job(self, job: dict) -> dict:
        result = {"schema_version": 1, **{key: job[key] for key in PUBLIC_JOB_FIELDS if key in job}}
        result.setdefault("updated_at", result.get("created_at"))
        state = self._receipt(job) if job.get("job_id") else None
        result["new_research_submitted"] = bool(state and state.get("submission_attempts", 0) > 0)
        if state is not None:
            result["receipt"] = {key: state[key] for key in RECEIPT_FIELDS if key in state}
        return result

    def submit(self, body: dict) -> dict:
        if not isinstance(body, dict) or set(body) != {"request_id", "consumer", "snapshot"}:
            raise ResearchError("invalid_request", "Supply only request_id, consumer and snapshot.", 422)
        request_id = checked_uuid(body["request_id"])
        consumer, snapshot = body["consumer"], body["snapshot"]
        if consumer not in self.config.consumers:
            raise ResearchError("consumer_not_configured", "This research consumer is not configured.", 422)
        validate_snapshot(snapshot, self.config, check_freshness=False)
        try:
            old = self.ledger.replay(consumer, request_id, snapshot, self.config.model, self.config.max_signals)
            if old is not None:
                return self.public_job(old)
            validate_snapshot(snapshot, self.config)
            # An empty diff can be reported without a live worker or an API key.
            plan = self.ledger.plan(consumer, snapshot, self.config.max_signals)
            if plan["selected_count"] and not self.worker_status()["ready"]:
                raise ResearchError("worker_not_ready", "Research worker is not ready; no new job was accepted.")
            if not self.config.api_enabled:
                raise ResearchError("research_disabled", "New research is not enabled on this server.")
            job = self.ledger.enqueue(consumer, request_id, snapshot, self.config.model, self.config.max_signals)
            return self.public_job(job)
        except LedgerBusy as exc:
            raise ResearchError("consumer_busy", f"Recover the existing research job {exc.job_id} first.", 409) from None
        except LedgerError as exc:
            code = "request_conflict" if "different payload" in str(exc) else "invalid_snapshot"
            raise ResearchError(code, str(exc), 409 if code == "request_conflict" else 422) from None

    def status(self, job_id: str) -> dict:
        checked_uuid(job_id)
        try:
            return self.public_job(self.ledger.get_job(job_id))
        except LedgerError:
            raise ResearchError("job_not_found", "Research job does not exist.", 404) from None

    def result(self, job_id: str) -> dict:
        checked_uuid(job_id)
        try:
            job = self.ledger.get_job(job_id)
        except LedgerError:
            raise ResearchError("job_not_found", "Research job does not exist.", 404) from None
        if job["status"] != "completed":
            raise ResearchError("result_not_ready", "This research job has no verified final dossier.", 409)
        root = self._run_path(job)
        path = root / "dossier.md" if root is not None else None
        try:
            if path is None or path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_DOSSIER_BYTES:
                raise ValueError()
            with path.open("rb") as stream:
                raw = stream.read(MAX_DOSSIER_BYTES + 1)
            if len(raw) > MAX_DOSSIER_BYTES or hashlib.sha256(raw).hexdigest() != job["dossier_sha256"]:
                raise ValueError()
            text = raw.decode("utf-8")
        except (OSError, ValueError, UnicodeError) as exc:
            raise ResearchError("result_integrity", "Saved research dossier failed verification.") from exc
        return {"schema_version": 1, "job_id": job_id, "status": "completed",
                "dossier_markdown": text, "sha256": job["dossier_sha256"], "bytes": len(raw)}
