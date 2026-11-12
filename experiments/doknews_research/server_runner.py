"""Isolated Linux research worker. One durable job creates at most one API session."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time

# The service launches this trusted script with Python -I. Add only its own
# directory, not an arbitrary working directory or caller-controlled path.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pilot
    import launcher
    from server_config import (ResearchError, ServerConfig, ensure_state, json_bytes,
                               load_config, read_private_json, utc_now, validate_snapshot)
    from signal_ledger import Ledger, LedgerError
else:
    from . import pilot, launcher
    from .server_config import (ResearchError, ServerConfig, ensure_state, json_bytes,
                                load_config, read_private_json, utc_now, validate_snapshot)
    from .signal_ledger import Ledger, LedgerError


def load_credential(cfg: ServerConfig):
    """Read only inside the worker; never print, store, or forward it as job data."""
    value = os.environ.get("OPENAI_API_KEY", "").strip()
    if cfg.openai_key_file is not None:
        path = cfg.openai_key_file
        try:
            stat = path.stat()
            if path.is_symlink() or not path.is_file() or stat.st_size > 16_384:
                raise ValueError()
            if os.name != "nt" and stat.st_mode & 0o027:
                raise ValueError()
            value = path.read_text(encoding="utf-8-sig").strip()
        except (OSError, UnicodeError, ValueError) as exc:
            raise ResearchError("credential_unavailable", "The worker credential file is unavailable or insufficiently protected.") from exc
    if not re.fullmatch(r"sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}", value):
        raise ResearchError("credential_unavailable", "The worker has no valid configured OpenAI credential.")
    os.environ["OPENAI_API_KEY"] = value
    try:
        version = importlib.metadata.version("openai")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ResearchError("runtime_unavailable", "Install the isolated research runtime first.") from exc
    if version != "3.49.0":
        raise ResearchError("runtime_unavailable", "The research worker requires the checked OpenAI SDK 3.49.0.")


def validate_runtime(cfg: ServerConfig) -> dict:
    """Check local prerequisites without opening the queue or calling OpenAI."""
    load_credential(cfg)
    files = (*pilot.POLICY_FILES, pilot.AI_GUIDE)
    try:
        for relative in files:
            pilot.read_text(cfg.skill_dir / relative)
    except (pilot.PilotError, OSError) as exc:
        raise ResearchError("policy_unavailable", "The complete local research policy bundle is unavailable or invalid.") from exc
    return {"status": "ready", "api_enabled": cfg.api_enabled, "model": cfg.model,
            "sdk_version": "3.49.0", "policy_files_count": len(files), "api_request_made": False}


@contextmanager
def worker_lock(cfg: ServerConfig):
    """Only one OS worker may observe/submit jobs in this state directory."""
    import fcntl
    path = cfg.state_dir / "worker.lock"
    if path.is_symlink():
        raise ResearchError("invalid_lock", "Worker lock cannot be a link.")
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ResearchError("worker_busy", "Another research worker is already active.") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class Heartbeat:
    def __init__(self, cfg: ServerConfig):
        self.path = cfg.state_dir / "worker-status.json"
        self.state = {"phase": "recovering", "job_id": None}
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def write(self):
        with self.lock:
            value = {"schema_version": 1, **self.state, "observed_at": utc_now()}
        pilot.atomic_bytes(self.path, json_bytes(value))

    def update(self, phase, job_id=None):
        with self.lock:
            self.state = {"phase": phase, "job_id": job_id}
        self.write()

    def _loop(self):
        while not self.stop_event.wait(10):
            try:
                self.write()
            except Exception:
                # A stale heartbeat closes new HTTP submissions; no secret logs.
                return

    def __enter__(self):
        self.write()
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join(timeout=2)


def checked_run_path(cfg: ServerConfig, job: dict) -> Path:
    expected = cfg.state_dir.absolute() / "runs" / job["job_id"] / "pilot"
    if job.get("run_dir") is not None and Path(job["run_dir"]).absolute() != expected:
        raise ResearchError("state_integrity", "Research job binding does not match its directory.")
    if expected.is_symlink() or expected.parent.is_symlink():
        raise ResearchError("state_integrity", "Research job directory cannot be a link.")
    return expected


def completed(state: dict) -> bool:
    return state.get("phase") == "completed" and state.get("result_status") == "available"


def execute_job(cfg: ServerConfig, ledger: Ledger, job_id: str, *, client_factory=None) -> dict:
    """Claim and freeze before the paid call. Re-entry never calls create again."""
    if not cfg.api_enabled:
        raise ResearchError("research_disabled", "New research is not enabled.")
    job = ledger.claim(job_id)
    run_dir = checked_run_path(cfg, job)
    try:
        validate_snapshot(job["snapshot"], cfg)
        if job["model"] != cfg.model:
            raise ResearchError("profile_changed", "Queued job no longer matches the configured model.")
        ledger.bind_run(job_id, run_dir)
        source = cfg.state_dir / "inputs" / (job_id + ".json")
        pilot.atomic_bytes(source, json_bytes(job["snapshot"]))
        manifest = pilot.prepare(
            cfg.skill_dir, source, run_dir, model=job["model"], include_ai_guide=True,
            no_spend_limit=True, research_mode="new_signals", research_as_of=utc_now(),
            reasoning_effort=cfg.reasoning_effort,
        )
        ledger.bind_run(job_id, run_dir, run_id=manifest["run_id"])
        launcher.preflight(run_dir, "run", max_snapshot_hours=cfg.max_snapshot_hours)
        factory = client_factory or pilot.client_for_run
        with factory(run_dir) as client:
            state = pilot.run(client, run_dir, max_seconds=cfg.max_seconds)
        if completed(state):
            return ledger.finish(job_id, state, run_dir / "dossier.md")
        return ledger.fail(job_id, state.get("phase", "unknown"), "No verified result")
    except BaseException as exc:
        phase = "preflight_failed"
        try:
            state = pilot.receipt(run_dir)
            if state.get("submission_attempts", 0):
                phase = state.get("phase", "unknown")
        except Exception:
            pass
        ledger.fail(job_id, phase, "Worker stopped; inspect saved receipt")
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        # Only fixed public errors leave this boundary. SDK details are already
        # sanitized and persisted by pilot.run, never serialized here.
        return ledger.get_job(job_id)


def reconcile_job(cfg: ServerConfig, ledger: Ledger, job: dict, *, client_factory=None) -> dict:
    """GET-only recovery of persisted IDs; it never creates a new session."""
    run_dir = checked_run_path(cfg, job)
    try:
        state = pilot.receipt(run_dir)
        if not job.get("run_id") or state.get("run_id") != job["run_id"]:
            raise ResearchError("state_integrity", "Saved receipt does not match this job.")
        if completed(state):
            return ledger.finish(job["job_id"], state, run_dir / "dossier.md")
        if not state.get("session_id") or state.get("session_deleted"):
            return job
        factory = client_factory or pilot.client_for_run
        with factory(run_dir) as client:
            state = pilot.recover(client, run_dir)
            deadline = state.get("deadline_epoch")
            last_cancel = state.get("server_cancel_retry_at", 0)
            if type(last_cancel) not in (int, float):
                last_cancel = 0
            if (state.get("phase") not in pilot.TERMINAL and type(deadline) in (int, float)
                    and time.time() >= deadline and time.time() - last_cancel >= 60):
                # Cancellation is separately recorded and is never reported as
                # confirmed until the saved remote state says cancelled.
                pilot.update_receipt(run_dir, server_cancel_retry_at=time.time())
                pilot.request_cancel(client, run_dir, "server_recovery_deadline")
        if completed(state):
            return ledger.finish(job["job_id"], state, run_dir / "dossier.md")
        return ledger.fail(job["job_id"], state.get("phase", "unknown"), "Remote result pending")
    except Exception:
        return ledger.get_job(job["job_id"])


def worker(cfg: ServerConfig, *, once=False, stop_event=None) -> int:
    stop_requested = stop_event if stop_event is not None else threading.Event()
    ensure_state(cfg)
    if not cfg.api_enabled:
        return 0
    with worker_lock(cfg), Heartbeat(cfg) as heartbeat:
        try:
            validate_runtime(cfg)
            ledger = Ledger(cfg.db_path)
            while True:
                if stop_requested.is_set():
                    raise KeyboardInterrupt()
                unresolved = ledger.list_unresolved()
                if unresolved:
                    # A previously interrupted job is reconciled before any
                    # further paid work, including jobs for another consumer.
                    heartbeat.update("recovering", unresolved[0]["job_id"])
                    for job in unresolved:
                        reconcile_job(cfg, ledger, job)
                    if ledger.list_unresolved():
                        heartbeat.update("recovery_required", unresolved[0]["job_id"])
                        if once:
                            return 2
                        time.sleep(max(10, cfg.poll_seconds))
                        continue
                job = ledger.next_queued()
                if job is None:
                    heartbeat.update("idle")
                    if once:
                        heartbeat.update("stopped")
                        return 0
                    time.sleep(cfg.poll_seconds)
                    continue
                heartbeat.update("running", job["job_id"])
                result = execute_job(cfg, ledger, job["job_id"])
                # pilot.run records/cancels an interrupted observation and then
                # raises PilotError. Preserve the OS stop request across that
                # conversion so shutdown cannot continue with another queued job.
                if stop_requested.is_set():
                    raise KeyboardInterrupt()
                heartbeat.update("idle" if result["status"] == "completed" else "recovery_required", job["job_id"])
                if once:
                    heartbeat.update("stopped" if result["status"] == "completed" else "recovery_required", job["job_id"])
                    return 0 if result["status"] == "completed" else 2
        except KeyboardInterrupt:
            heartbeat.update("stopped")
            return 130
        except Exception:
            heartbeat.update("configuration_error")
            return 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "plan", "check"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--consumer", default="doknews-api-hourly")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        if args.command == "check":
            print(json.dumps(validate_runtime(cfg)))
            return 0
        if args.command == "plan":
            if args.snapshot is None or args.consumer not in cfg.consumers:
                raise ResearchError("invalid_request", "Plan requires a snapshot and configured consumer.", 422)
            snapshot = read_private_json(args.snapshot, 1_000_000)
            validate_snapshot(snapshot, cfg)
            ensure_state(cfg)
            plan = Ledger(cfg.db_path).plan(args.consumer, snapshot, cfg.max_signals)
            print(json.dumps({key: plan[key] for key in ("consumer", "selected_count", "pending_count",
                "pending_remaining", "researched_count", "coverage", "input_coverage", "blocked_job_id")}, ensure_ascii=False))
            return 0
        if os.name != "posix":
            raise ResearchError("unsupported_host", "The server worker requires a Linux host.")
        stop_requested = threading.Event()
        def stop(*_):
            stop_requested.set()
            raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        return worker(cfg, once=args.once, stop_event=stop_requested)
    except (ResearchError, LedgerError, pilot.PilotError) as exc:
        code = exc.code if isinstance(exc, ResearchError) else "research_error"
        print(json.dumps({"error": {"code": code, "message": str(exc)}}), file=sys.stderr)
        return 2
    except Exception:
        print('{"error":{"code":"research_error","message":"Research worker stopped; inspect its private state."}}', file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
