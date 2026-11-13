"""Durable, local research reservations; never writes editorial radar state.

One unresolved job blocks a consumer, including after an uncertain or failed
submission. Only a verified dossier advances researched signal identities.
Every write uses a SQLite transaction; claiming is a separate atomic operation
so merely inspecting a queued job cannot submit it twice.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import uuid


UNRESOLVED = ("queued", "running", "unknown", "failed")
FINISHABLE = ("running", "unknown", "failed", "completed")
MAX_DOSSIER_BYTES = 2_000_000
# Deliberately exclude fetch/read timestamps, retrieval, review state, and
# current_version. Hash actual supplied content even when an upstream digest or
# version was accidentally reused. Publication time is source content.
CONTENT_FIELDS = (
    "source_id", "source_url", "canonical_url", "original_id", "title",
    "excerpt", "summary", "text", "body", "content", "content_sha256",
    "published_at", "author", "authors", "label", "kind", "topics",
    "language", "attachments", "media",
)
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
MODEL = re.compile(r"[A-Za-z0-9._-]{1,100}")
SHA256 = re.compile(r"[a-f0-9]{64}")
FAILURE_MESSAGE = "Research did not produce a verified result; inspect the saved run receipt."


class LedgerError(Exception):
    """Fixed, application-authored messages safe to show without source data."""


class LedgerBusy(LedgerError):
    def __init__(self, job_id: str):
        super().__init__("This consumer has an unresolved research job; recover that job first.")
        self.job_id = job_id


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise LedgerError("Ledger input must contain only valid JSON data.") from exc


def _name(value, description: str):
    if not isinstance(value, str) or not NAME.fullmatch(value):
        raise LedgerError(description)


def _limit(limit):
    if type(limit) is not int or not 1 <= limit <= 30:
        raise LedgerError("Research batch limit must be an integer from 1 to 30.")


def signal_fingerprint(signal: dict) -> str:
    """Content identity independent of observational/editorial metadata."""
    payload = {key: signal[key] for key in CONTENT_FIELDS if key in signal}
    return hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()


def _snapshot(value: dict) -> tuple[dict, list[dict]]:
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise LedgerError("Snapshot schema_version must be 1.")
    task = value.get("task")
    if not isinstance(task, dict) or not all(isinstance(task.get(k), str) and task[k].strip()
                                             for k in ("name", "topic")):
        raise LedgerError("Snapshot task requires a nonempty name and topic.")
    hours = task.get("time_window_hours")
    if type(hours) is not int or not 1 <= hours <= 168:
        raise LedgerError("Snapshot time window must be an integer from 1 to 168 hours.")
    try:
        captured = datetime.fromisoformat(value["captured_at"].replace("Z", "+00:00"))
        if captured.tzinfo is None:
            raise ValueError()
    except (KeyError, ValueError, AttributeError, TypeError) as exc:
        raise LedgerError("Snapshot captured_at must be an ISO timestamp with a timezone.") from exc
    coverage = value.get("coverage")
    if not isinstance(coverage, dict):
        raise LedgerError("Snapshot must describe source coverage.")
    if any(coverage.get(key) not in ("not_provided", "partial", "complete")
           for key in ("archive", "radar", "memory")):
        raise LedgerError("Snapshot coverage must be not_provided, partial or complete.")
    if not isinstance(coverage.get("missing"), list) or not all(isinstance(x, str) for x in coverage["missing"]):
        raise LedgerError("Snapshot coverage.missing must be an array of strings.")
    for key in ("radar_signals", "archive_articles", "memory_records"):
        if not isinstance(value.get(key), list) or not all(isinstance(x, dict) for x in value[key]):
            raise LedgerError("Snapshot signal, article and memory collections must be arrays of objects.")
    # A JSON round trip freezes the caller's mutable objects and rejects values
    # SQLite/another process could not reconstruct faithfully.
    frozen = json.loads(_json(value))
    identities = []
    seen = set()
    for signal in frozen["radar_signals"]:
        signal_id, version = signal.get("signal_id"), signal.get("version")
        if not isinstance(signal_id, str) or not signal_id.strip() or len(signal_id) > 512:
            raise LedgerError("Every radar signal requires a nonempty signal_id.")
        if type(version) is not int or not 1 <= version <= 2**63 - 1:
            raise LedgerError("Every radar signal requires a positive integer version.")
        if (signal_id, version) in seen:
            raise LedgerError("Snapshot contains a duplicate signal_id and version.")
        seen.add((signal_id, version))
        for key in ("source_id", "source_url", "title"):
            if not isinstance(signal.get(key), str) or not signal[key].strip():
                raise LedgerError("Every radar signal requires source_id, source_url and title.")
        for key in ("text", "excerpt", "summary", "body"):
            if key in signal and signal[key] is not None and not isinstance(signal[key], str):
                raise LedgerError("Radar signal text fields must contain text.")
        supplied_hash = signal.get("content_sha256")
        if supplied_hash is not None and (not isinstance(supplied_hash, str) or not SHA256.fullmatch(supplied_hash)):
            raise LedgerError("Radar content_sha256 must be a lowercase SHA256 digest.")
        identities.append({"signal_id": signal_id, "version": version,
                           "fingerprint": signal_fingerprint(signal)})
    return frozen, identities


def _payload_identity(consumer: str, snapshot: dict, model: str, limit: int):
    """Use identical validation and canonical payload hashing for replay/enqueue."""
    _name(consumer, "Invalid research consumer identifier.")
    _limit(limit)
    if not isinstance(model, str) or not MODEL.fullmatch(model):
        raise LedgerError("Invalid research model identifier.")
    frozen, identities = _snapshot(snapshot)
    payload_sha256 = hashlib.sha256(_json({"consumer": consumer, "snapshot": frozen,
        "model": model, "limit": limit}).encode("utf-8")).hexdigest()
    return frozen, identities, payload_sha256


class Ledger:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).absolute()
        try:
            self.db_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            if self.db_path.is_symlink():
                raise LedgerError("Refusing a symlink research ledger.")
            try:
                fd = os.open(self.db_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                if not self.db_path.is_file():
                    raise LedgerError("The research ledger must be a regular local file.")
            else:
                os.close(fd)
            os.chmod(self.db_path, 0o600)
        except OSError as exc:
            raise LedgerError("Cannot create or protect the research ledger.") from exc
        with self._connect(write=True) as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise LedgerError("Unsupported research ledger schema.")
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY, consumer TEXT NOT NULL,
                request_id TEXT NOT NULL, payload_sha256 TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('queued','running','unknown','failed','completed')),
                model TEXT NOT NULL, batch_limit INTEGER NOT NULL,
                snapshot_json TEXT NOT NULL, identities_json TEXT NOT NULL,
                counts_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                run_dir TEXT, run_id TEXT, receipt_json TEXT, dossier_sha256 TEXT,
                failure_phase TEXT, failure_message TEXT,
                UNIQUE(consumer, request_id))""")
            db.execute("""CREATE UNIQUE INDEX IF NOT EXISTS one_unresolved_consumer
                ON jobs(consumer) WHERE status IN ('queued','running','unknown','failed')""")
            db.execute("""CREATE TABLE IF NOT EXISTS requests (
                consumer TEXT NOT NULL, request_id TEXT NOT NULL, payload_sha256 TEXT NOT NULL,
                job_id TEXT REFERENCES jobs(job_id), response_json TEXT,
                PRIMARY KEY(consumer, request_id))""")
            db.execute("""CREATE TABLE IF NOT EXISTS researched_signals (
                consumer TEXT NOT NULL, signal_id TEXT NOT NULL, version INTEGER NOT NULL,
                fingerprint TEXT NOT NULL, job_id TEXT NOT NULL REFERENCES jobs(job_id),
                researched_at TEXT NOT NULL,
                PRIMARY KEY(consumer, signal_id, version, fingerprint))""")
            db.execute("PRAGMA user_version = 1")

    @contextmanager
    def _connect(self, write=False):
        db = None
        try:
            db = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys = ON")
            db.execute("PRAGMA synchronous = FULL")
            if write:
                db.execute("BEGIN IMMEDIATE")
            else:
                db.execute("PRAGMA query_only = ON")
                db.execute("BEGIN")
            yield db
            db.commit()
        except sqlite3.Error as exc:
            if db is not None:
                db.rollback()
            raise LedgerError("Cannot access the research ledger safely.") from exc
        except BaseException:
            if db is not None:
                db.rollback()
            raise
        finally:
            if db is not None:
                db.close()

    @staticmethod
    def _job(row) -> dict:
        if row is None:
            raise LedgerError("Research job does not exist.")
        result = dict(row)
        result["snapshot"] = json.loads(result.pop("snapshot_json"))
        result["selected_signals"] = result["snapshot"]["radar_signals"]
        result["selected_identities"] = json.loads(result.pop("identities_json"))
        result.update(json.loads(result.pop("counts_json")))
        raw_receipt = result.pop("receipt_json")
        result["receipt"] = json.loads(raw_receipt) if raw_receipt else None
        result["limit"] = result.pop("batch_limit")
        return result

    def _get(self, db, job_id: str) -> dict:
        return self._job(db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone())

    def get_job(self, job_id: str) -> dict:
        with self._connect() as db:
            return self._get(db, job_id)

    def next_queued(self, consumer: str | None = None) -> dict | None:
        """Inspect the oldest queued job. The worker must still call claim()."""
        with self._connect() as db:
            if consumer is None:
                row = db.execute("SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at, job_id LIMIT 1").fetchone()
            else:
                _name(consumer, "Invalid research consumer identifier.")
                row = db.execute("""SELECT * FROM jobs WHERE status = 'queued' AND consumer = ?
                                  ORDER BY created_at, job_id LIMIT 1""", (consumer,)).fetchone()
            return self._job(row) if row is not None else None

    def list_unresolved(self, limit: int = 100) -> list[dict]:
        """Inspect claimed jobs needing receipt reconciliation; never requeue."""
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise LedgerError("Unresolved job list limit must be an integer from 1 to 1000.")
        with self._connect() as db:
            rows = db.execute("""SELECT * FROM jobs WHERE status IN ('running','unknown','failed')
                ORDER BY created_at, job_id LIMIT ?""", (limit,)).fetchall()
            return [self._job(row) for row in rows]

    def next_usage_refresh(self, before: str) -> dict | None:
        """One completed job whose delayed, best-effort usage read is still due."""
        with self._connect() as db:
            row = db.execute("""SELECT * FROM jobs WHERE status = 'completed'
                AND updated_at <= ?
                AND json_extract(receipt_json, '$.usage_refresh_attempted_at') IS NULL
                ORDER BY updated_at, job_id LIMIT 1""", (before,)).fetchone()
            return self._job(row) if row is not None else None

    def _plan(self, db, consumer, snapshot, identities, limit):
        selected, selected_ids = [], []
        pending = 0
        for signal, identity in zip(snapshot["radar_signals"], identities):
            found = db.execute("""SELECT 1 FROM researched_signals
                WHERE consumer = ? AND signal_id = ? AND version = ? AND fingerprint = ?""",
                (consumer, identity["signal_id"], identity["version"], identity["fingerprint"])).fetchone()
            if found is None:
                pending += 1
                if len(selected) < limit:
                    selected.append(signal)
                    selected_ids.append(identity)
        frozen = json.loads(_json(snapshot))
        frozen["radar_signals"] = selected
        total = len(identities)
        counts = {"selected_count": len(selected), "pending_count": pending,
                  "pending_remaining": pending - len(selected), "total_count": total,
                  "researched_count": total - pending}
        frozen["research_selection"] = dict(counts)
        frozen["research_selection"].update(consumer=consumer, limit=limit)
        if len(selected) < total:
            frozen["coverage"]["radar"] = "partial"
            frozen["coverage"]["missing"].append(
                "Research ledger selected a bounded batch; omitted signals are not included in this request.")
        counts.update(coverage=frozen["coverage"], input_coverage=snapshot["coverage"])
        blocked = db.execute("""SELECT job_id FROM jobs WHERE consumer = ?
            AND status IN ('queued','running','unknown','failed')""", (consumer,)).fetchone()
        return {"consumer": consumer, **counts, "selected_signals": selected,
                "selected_identities": selected_ids, "snapshot": frozen,
                "blocked_job_id": blocked["job_id"] if blocked else None}

    def plan(self, consumer: str, snapshot: dict, limit: int = 3) -> dict:
        """Read only. No reservation or research marker is created."""
        _name(consumer, "Invalid research consumer identifier.")
        _limit(limit)
        frozen, identities = _snapshot(snapshot)
        with self._connect() as db:
            return self._plan(db, consumer, frozen, identities, limit)

    def _replay(self, db, consumer: str, request_id: str, payload_sha256: str) -> dict | None:
        old = db.execute("SELECT * FROM requests WHERE consumer = ? AND request_id = ?",
                         (consumer, request_id)).fetchone()
        if old is None:
            return None
        if old["payload_sha256"] != payload_sha256:
            raise LedgerError("This request identifier was already used with a different payload.")
        return self._get(db, old["job_id"]) if old["job_id"] else json.loads(old["response_json"])

    def replay(self, consumer: str, request_id: str, snapshot: dict,
               model: str, limit: int = 3) -> dict | None:
        """Read a prior request even when new submission is no longer permitted."""
        _name(request_id, "Invalid research request identifier.")
        _, _, payload_sha256 = _payload_identity(consumer, snapshot, model, limit)
        with self._connect() as db:
            return self._replay(db, consumer, request_id, payload_sha256)

    def enqueue(self, consumer: str, request_id: str, snapshot: dict,
                model: str, limit: int = 3) -> dict:
        _name(request_id, "Invalid research request identifier.")
        frozen, identities, payload_sha256 = _payload_identity(consumer, snapshot, model, limit)
        with self._connect(write=True) as db:
            old = self._replay(db, consumer, request_id, payload_sha256)
            if old is not None:
                return old
            plan = self._plan(db, consumer, frozen, identities, limit)
            if plan["blocked_job_id"]:
                raise LedgerBusy(plan["blocked_job_id"])
            counts = {key: plan[key] for key in ("selected_count", "pending_count", "pending_remaining",
                                                "total_count", "researched_count", "coverage", "input_coverage")}
            created = _now()
            if not plan["selected_count"]:
                result = {"job_id": None, "status": "no_signals", "consumer": consumer,
                          "request_id": request_id, "model": model, "limit": limit,
                          "payload_sha256": payload_sha256, "created_at": created, **counts}
                db.execute("INSERT INTO requests VALUES (?, ?, ?, NULL, ?)",
                           (consumer, request_id, payload_sha256, _json(result)))
                return result
            job_id = str(uuid.uuid4())
            db.execute("""INSERT INTO jobs (job_id, consumer, request_id, payload_sha256, status,
                model, batch_limit, snapshot_json, identities_json, counts_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?)""",
                (job_id, consumer, request_id, payload_sha256, model, limit,
                 _json(plan["snapshot"]), _json(plan["selected_identities"]), _json(counts), created, created))
            db.execute("INSERT INTO requests VALUES (?, ?, ?, ?, NULL)",
                       (consumer, request_id, payload_sha256, job_id))
            return self._get(db, job_id)

    def claim(self, job_id: str) -> dict:
        with self._connect(write=True) as db:
            changed = db.execute("UPDATE jobs SET status = 'running', updated_at = ? WHERE job_id = ? AND status = 'queued'",
                                 (_now(), job_id)).rowcount
            if changed != 1:
                raise LedgerError("Research job is not queued; it must not be submitted again.")
            return self._get(db, job_id)

    def bind_run(self, job_id: str, run_dir: str | Path, run_id: str | None = None) -> dict:
        path = str(Path(run_dir).absolute())
        if run_id is not None:
            _name(run_id, "Invalid research run identifier.")
        with self._connect(write=True) as db:
            job = self._get(db, job_id)
            if job["run_dir"] not in (None, path) or (run_id is not None and job["run_id"] not in (None, run_id)):
                raise LedgerError("Refusing to replace the research job's bound run.")
            if job["status"] == "completed" and (job["run_dir"] != path or run_id not in (None, job["run_id"])):
                raise LedgerError("Completed research job binding cannot change.")
            db.execute("UPDATE jobs SET run_dir = ?, run_id = ?, updated_at = ? WHERE job_id = ?",
                       (path, run_id or job["run_id"], _now(), job_id))
            return self._get(db, job_id)

    def finish(self, job_id: str, receipt: dict, dossier_path: str | Path) -> dict:
        if (not isinstance(receipt, dict) or receipt.get("phase") != "completed"
                or receipt.get("result_status") != "available"):
            raise LedgerError("Only a completed run with an available dossier can finish research.")
        if receipt.get("dossier_file") != "dossier.md":
            raise LedgerError("Completed run receipt must identify dossier.md.")
        expected = receipt.get("dossier_sha256")
        if not isinstance(expected, str) or not SHA256.fullmatch(expected):
            raise LedgerError("Completed run receipt must contain a valid dossier SHA256.")
        saved_receipt = _json(receipt)
        path = Path(dossier_path).absolute()
        with self._connect(write=True) as db:
            job = self._get(db, job_id)
            if job["status"] not in FINISHABLE:
                raise LedgerError("Research job has not been claimed.")
            if not job["run_id"] or receipt.get("run_id") != job["run_id"]:
                raise LedgerError("Run receipt does not match the research job's bound run.")
            if not job["run_dir"] or path != Path(job["run_dir"]) / "dossier.md":
                raise LedgerError("Dossier does not belong to the research job's bound run.")
            try:
                if path.is_symlink() or not path.is_file():
                    raise LedgerError("Dossier must be a regular local file.")
                if path.stat().st_size > MAX_DOSSIER_BYTES:
                    raise LedgerError("Completed dossier exceeds the permitted size.")
                with path.open("rb") as stream:
                    raw = stream.read(MAX_DOSSIER_BYTES + 1)
                if len(raw) > MAX_DOSSIER_BYTES:
                    raise LedgerError("Completed dossier exceeds the permitted size.")
            except OSError as exc:
                raise LedgerError("Cannot read the completed dossier.") from exc
            try:
                text = raw.decode("utf-8")
            except UnicodeError as exc:
                raise LedgerError("Completed dossier must contain valid UTF-8 text.") from exc
            actual = hashlib.sha256(raw).hexdigest()
            if not text.strip() or actual != expected:
                raise LedgerError("Dossier content does not match the completed run receipt.")
            if job["status"] == "completed":
                if job["dossier_sha256"] != actual:
                    raise LedgerError("Refusing to replace a completed research dossier.")
                # Late accounting does not change the completion time or signal ledger.
                db.execute("UPDATE jobs SET receipt_json = ? WHERE job_id = ?",
                           (saved_receipt, job_id))
                return self._get(db, job_id)
            finished = _now()
            for identity in job["selected_identities"]:
                db.execute("""INSERT INTO researched_signals
                    (consumer, signal_id, version, fingerprint, job_id, researched_at)
                    VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING""",
                    (job["consumer"], identity["signal_id"], identity["version"],
                     identity["fingerprint"], job_id, finished))
            db.execute("""UPDATE jobs SET status = 'completed', receipt_json = ?, dossier_sha256 = ?,
                failure_phase = NULL, failure_message = NULL, updated_at = ? WHERE job_id = ?""",
                (saved_receipt, actual, finished, job_id))
            return self._get(db, job_id)

    def fail(self, job_id: str, phase: str, message: str) -> dict:
        """Keep the consumer blocked. Arbitrary exception text is never persisted.

        ``message`` is intentionally not echoed: a worker may accidentally pass
        an SDK exception containing request data, credentials or local paths.
        The bound pilot receipt remains the detailed diagnostic record.
        """
        if not isinstance(phase, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", phase):
            raise LedgerError("Invalid research failure phase.")
        status = "unknown" if phase in ("unknown", "creation_unknown", "submitting", "submission_unknown") else "failed"
        with self._connect(write=True) as db:
            job = self._get(db, job_id)
            if job["status"] == "completed":
                raise LedgerError("A completed research job cannot be changed to failed.")
            db.execute("""UPDATE jobs SET status = ?, failure_phase = ?, failure_message = ?,
                updated_at = ? WHERE job_id = ?""", (status, phase, FAILURE_MESSAGE, _now(), job_id))
            return self._get(db, job_id)
