"""Manual, research-only Agents API pilot. No production imports or credentials in files.

prepare is offline. run submits exactly once per run directory. resume/status
only inspect the saved session; they never send another research prompt.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import uuid

if __package__:
    from . import context as research_context
else:
    import context as research_context


BASE = Path(__file__).resolve().parent
MAX_FILE_BYTES = 512_000
MAX_REQUEST_BYTES = 2_000_000
DEFAULT_MODEL = "gpt-6.1-sol"
DEFAULT_SPEND_LIMIT_CENTS = 300
REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
TERMINAL = {"completed", "failed", "cancelled"}
POLICY_FILES = (
    "SKILL.md",
    "references/DOKNEWS_EDITORIAL_POLICY.md",
    "references/DOKNEWS_EDITORIAL_CALIBRATION.md",
    "references/DOKNEWS_EDITORIAL_MEMORY.md",
)
AI_GUIDE = "references/DOKNEWS_AI_EDITORIAL_GUIDE.md"
RESEARCH_POLICY_FILES = POLICY_FILES[1:3]
SERVICE_TIERS = {"default", "flex"}
SECRET_PATTERN = re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}")
PROJECT_ID_PATTERN = re.compile(r"proj_[A-Za-z0-9_-]{1,100}")


class PilotError(Exception):
    """Only messages authored by this program may reach the terminal."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def encode(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_text(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise PilotError(f"Expected a regular local input file: {path.name}")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise PilotError(f"Input file exceeds {MAX_FILE_BYTES} bytes: {path.name}")
    try:
        value = path.read_text(encoding="utf-8")
    except (UnicodeError, OSError) as exc:
        raise PilotError(f"Cannot read UTF-8 input: {path.name}") from exc
    if SECRET_PATTERN.search(value):
        raise PilotError("An input appears to contain an API secret; remove it before preparing.")
    return value


def read_json(path: Path):
    try:
        return json.loads(read_text(path))
    except json.JSONDecodeError as exc:
        raise PilotError(f"Invalid JSON: {path.name}") from exc


def atomic_bytes(path: Path, content: bytes):
    if path.is_symlink():
        raise PilotError("Refusing a symlink output.")
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


@contextmanager
def locked(run_dir: Path):
    """OS locks are released after process death; no stale PID probing is needed."""
    path = run_dir / ".lock"
    if path.is_symlink():
        raise PilotError("Refusing a symlink lock.")
    with open(path, "a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if stream.seek(0, os.SEEK_END) == 0:
                stream.write(b" ")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise PilotError("The run receipt is busy; retry the read shortly.") from exc
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def receipt(run_dir: Path) -> dict:
    result = read_json(run_dir / "receipt.json")
    if not isinstance(result, dict) or result.get("schema_version") != 1:
        raise PilotError("Unsupported or corrupt run receipt.")
    return result


def update_receipt(run_dir: Path, **changes) -> dict:
    with locked(run_dir):
        value = receipt(run_dir)
        for name in ("session_id", "turn_id"):
            if value.get(name) and changes.get(name) and value[name] != changes[name]:
                raise PilotError(f"Refusing to replace the saved {name}.")
        value.update(changes, updated_at=now())
        atomic_bytes(run_dir / "receipt.json", encode(value))
        return value


def validate_snapshot(value: dict):
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise PilotError("Snapshot schema_version must be 1.")
    task = value.get("task")
    if not isinstance(task, dict) or not all(isinstance(task.get(k), str) and task[k].strip() for k in ("name", "topic")):
        raise PilotError("Snapshot task requires nonempty name and topic.")
    hours = task.get("time_window_hours")
    if type(hours) is not int or not 1 <= hours <= 168:
        raise PilotError("task.time_window_hours must be an integer from 1 to 168.")
    try:
        captured = datetime.fromisoformat(value["captured_at"].replace("Z", "+00:00"))
        if captured.tzinfo is None:
            raise ValueError()
    except (KeyError, ValueError, AttributeError, TypeError) as exc:
        raise PilotError("captured_at must be an ISO timestamp with a timezone.") from exc
    coverage = value.get("coverage")
    if not isinstance(coverage, dict) or not all(k in coverage for k in ("archive", "radar", "memory", "missing")):
        raise PilotError("Snapshot must describe archive, radar, memory and missing coverage.")
    if any(coverage[k] not in {"not_provided", "partial", "complete"} for k in ("archive", "radar", "memory")):
        raise PilotError("Coverage must be not_provided, partial or complete.")
    if not isinstance(coverage["missing"], list) or not all(isinstance(x, str) for x in coverage["missing"]):
        raise PilotError("coverage.missing must be an array of strings.")
    for name in ("radar_signals", "archive_articles", "memory_records"):
        if not isinstance(value.get(name), list) or not all(isinstance(x, dict) for x in value[name]):
            raise PilotError(f"Snapshot {name} must be an array, including when empty.")


def prepare(skill_dir: Path, snapshot_path: Path, run_dir: Path, model: str = DEFAULT_MODEL,
            include_ai_guide: bool = False, spend_limit_cents: int | None = DEFAULT_SPEND_LIMIT_CENTS,
            project_id: str | None = None, no_spend_limit: bool = False, *,
            research_mode: str = "general", research_as_of: str | None = None,
            reasoning_effort: str | None = None, service_tier: str = "default",
            research_only: bool = False):
    if service_tier not in SERVICE_TIERS or type(research_only) is not bool:
        raise PilotError("Unsupported research profile or service tier.")
    if reasoning_effort is not None and (type(reasoning_effort) is not str or reasoning_effort not in REASONING_EFFORTS):
        raise PilotError("Unsupported explicit reasoning effort.")
    if type(no_spend_limit) is not bool:
        raise PilotError("--no-spend-limit must be an explicit boolean selection.")
    if no_spend_limit and project_id is not None:
        raise PilotError("--no-spend-limit and --project-id cannot be used together.")
    if no_spend_limit:
        spend_limit_cents = None
    elif type(spend_limit_cents) is not int or not 1 <= spend_limit_cents <= 10_000:
        raise PilotError("Pilot spend limit must be from 1 to 10000 USD cents.")
    budget_control = {"kind": "unlimited"} if no_spend_limit else None
    if project_id is not None:
        if type(project_id) is not str or not PROJECT_ID_PATTERN.fullmatch(project_id):
            raise PilotError("Expected an explicit project ID beginning with proj_.")
        if spend_limit_cents > 300:
            raise PilotError("This pilot's project monthly hard limit must be from 1 to 300 USD cents.")
        budget_control = {"kind": "project_hard_limit", "project_id": project_id,
                          "monthly_limit_cents": spend_limit_cents}
    snapshot = read_json(snapshot_path)
    validate_snapshot(snapshot)
    contract = read_text(BASE / ("research_scope.md" if research_only else "dossier_contract.md"))
    bundle = []
    files = [*(RESEARCH_POLICY_FILES if research_only else POLICY_FILES),
             *([AI_GUIDE] if include_ai_guide else [])]
    for relative in files:
        text = read_text(skill_dir / relative)
        bundle.append({"path": relative, "sha256": digest(text.encode()), "text": text})
    run_id = str(uuid.uuid4())
    instructions = contract + "\n\n# Действующие инструкции исследовательского этапа\n"
    for source in bundle:
        instructions += f"\n## Файл: {source['path']}\n{source['text']}\n"
    if research_only:
        instructions += ("\n\nВыполни только исследовательский контракт в начале пакета. "
            "Операционные поручения редактору внутри справочных правил здесь не выполняются.\n")
    try:
        built_context = research_context.build_context(
            snapshot, run_id, research_mode=research_mode,
            research_as_of=now() if research_as_of is None else research_as_of)
    except research_context.ContextError as exc:
        raise PilotError(str(exc)) from exc
    request = {
        "agent": {
            "model": model,
            "service_tier": service_tier,
            "instructions": instructions,
            "tools": [{"type": "web_search", "mode": "live", "context_size": "medium"}],
        },
        "environment": {"type": "openai_hosted"},
        "spend_control": {"limit": spend_limit_cents},
        "metadata": {"dok_research_run_id": run_id, "purpose": "doknews_manual_research_pilot"},
        "input": built_context.input,
    }
    if reasoning_effort is not None:
        request["agent"]["reasoning"] = {"effort": reasoning_effort}
    if budget_control is not None:
        # No session limit is sent for these explicitly selected modes.
        del request["spend_control"]
    request_bytes = encode(request)
    if len(request_bytes) > MAX_REQUEST_BYTES:
        raise PilotError("Prepared request is too large; reduce the source snapshot.")
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", model):
        raise PilotError("Invalid model identifier.")
    # Only after all input validation, reserve a new run directory.
    try:
        run_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise PilotError("Run directory already exists. Use status/resume for an existing run.") from exc
    manifest = {
        "schema_version": 1, "run_id": run_id, "created_at": now(),
        "model": model, "request_sha256": digest(request_bytes),
        "contract_sha256": digest(contract.encode()),
        "policy_files": [{k: s[k] for k in ("path", "sha256")} for s in bundle],
        "snapshot_captured_at": snapshot["captured_at"],
        "coverage": snapshot["coverage"], "request_bytes": len(request_bytes),
        "ai_guide_included": include_ai_guide,
        "spend_limit_cents": spend_limit_cents,
        "reasoning_effort": reasoning_effort,
        "service_tier": service_tier,
        "instruction_profile": "research_v1" if research_only else "legacy_full",
        "instructions_sha256": digest(instructions.encode("utf-8")),
        "context_prefix_sha256": digest((instructions + "\n\n" + built_context.stable_prefix).encode("utf-8")),
        **built_context.metadata,
    }
    if budget_control is not None:
        manifest["budget_control"] = budget_control
    atomic_bytes(run_dir / "request.json", request_bytes)
    atomic_bytes(run_dir / "manifest.json", encode(manifest))
    state = {
        "schema_version": 1, "run_id": run_id, "phase": "prepared",
        "session_id": None, "turn_id": None, "created_at": now(),
        "request_sha256": manifest["request_sha256"], "usage": None,
        "submission_attempts": 0,
        "service_tier": service_tier,
        "spend_limit_cents": spend_limit_cents,
        "research_mode": built_context.metadata["research_mode"],
        "research_as_of": built_context.metadata["research_as_of"],
        "research_window": built_context.metadata["research_window"],
    }
    if budget_control is not None:
        state["budget_control"] = budget_control
    atomic_bytes(run_dir / "receipt.json", encode(state))
    return manifest


def _checked_project_budget(policy, legacy_limit) -> dict:
    fields = {"kind", "project_id", "monthly_limit_cents"}
    if (type(policy) is not dict or set(policy) != fields
            or policy.get("kind") != "project_hard_limit"):
        raise PilotError("Invalid project budget policy format.")
    project_id, limit = policy["project_id"], policy["monthly_limit_cents"]
    if (type(project_id) is not str or not PROJECT_ID_PATTERN.fullmatch(project_id)
            or type(limit) is not int or not 1 <= limit <= 300
            or type(legacy_limit) is not int or legacy_limit != limit):
        raise PilotError("Invalid or inconsistent project monthly hard-limit policy.")
    return dict(policy)


def _checked_unlimited_budget(record: dict) -> dict:
    policy = record.get("budget_control")
    if (type(policy) is not dict or policy != {"kind": "unlimited"}
            or "spend_limit_cents" not in record or record["spend_limit_cents"] is not None):
        raise PilotError("Invalid explicit unlimited budget policy.")
    return {"kind": "unlimited"}


def budget_policy(run_dir: Path, request: dict | None = None) -> dict:
    """Validate the saved budget mode; this does not verify Platform settings."""
    state = receipt(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    if type(manifest) is not dict or manifest.get("schema_version") != 1:
        raise PilotError("Unsupported or corrupt run manifest.")
    if (type(state.get("run_id")) is not str or manifest.get("run_id") != state["run_id"]
            or type(state.get("request_sha256")) is not str
            or not re.fullmatch(r"[a-f0-9]{64}", state["request_sha256"])
            or manifest.get("request_sha256") != state["request_sha256"]):
        raise PilotError("Prepared manifest and receipt identities do not match.")
    if request is None:
        path = run_dir / "request.json"
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_REQUEST_BYTES:
            raise PilotError("Expected the original bounded prepared request.")
        raw = path.read_bytes()
        if digest(raw) != state.get("request_sha256") or digest(raw) != manifest.get("request_sha256"):
            raise PilotError("Prepared request changed; refusing to use its budget policy.")
        try:
            request = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise PilotError("Invalid prepared request JSON.") from exc
    if type(request) is not dict:
        raise PilotError("Prepared request must be an object.")
    if "budget_control" in manifest or "budget_control" in state:
        values = (manifest.get("budget_control"), state.get("budget_control"))
        if any(type(value) is dict and value.get("kind") == "unlimited" for value in values):
            policy = _checked_unlimited_budget(manifest)
            saved_policy = _checked_unlimited_budget(state)
        else:
            policy = _checked_project_budget(values[0], manifest.get("spend_limit_cents"))
            saved_policy = _checked_project_budget(values[1], state.get("spend_limit_cents"))
        if policy != saved_policy:
            raise PilotError("Budget policy does not match the manifest and receipt.")
        if "spend_control" in request:
            raise PilotError("This budget mode must not contain session spend_control.")
        return policy
    spend = request.get("spend_control")
    limit = spend.get("limit") if type(spend) is dict else None
    if (type(spend) is not dict or set(spend) != {"limit"}
            or type(limit) is not int or not 1 <= limit <= 10_000
            or type(manifest.get("spend_limit_cents")) is not int
            or type(state.get("spend_limit_cents")) is not int
            or limit != manifest["spend_limit_cents"] or limit != state["spend_limit_cents"]):
        raise PilotError("Prepared run has no matching session spend limit; prepare a new bounded run.")
    return {"kind": "session", "limit_cents": limit}


def as_dict(value) -> dict:
    if isinstance(value, dict):
        return value
    return value.model_dump(mode="json")


def safe_error(exc: Exception) -> dict:
    """Allowlisted SDK diagnostics, never an exception/request/body serialization."""
    result = {"type": "Exception"}
    try:
        if __package__:
            from .bootstrap import redact
        else:
            from bootstrap import redact

        def clean(value, limit):
            if type(value) is not str or len(value) > 1_000_000:
                return None
            key = os.environ.get("OPENAI_API_KEY", "")
            for secret in sorted({key, key.strip()}, key=len, reverse=True):
                if secret:
                    value = value.replace(secret, "[REDACTED_KEY]")
            # Redact the complete value before truncating, including identifiers.
            value = redact(value)
            value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", value)
            return value[:limit]

        def identifier(value, limit, pattern):
            if type(value) is not str or len(value) > limit:
                return None
            value = clean(value, limit)
            return value if value and re.fullmatch(pattern, value) else None

        result["type"] = identifier(type(exc).__name__, 100, r"[A-Za-z_][A-Za-z0-9_]*") or "Exception"
        status = getattr(exc, "status_code", None)
        if type(status) is int and 100 <= status <= 599:
            result["http_status"] = status
        from openai import APIError, APIStatusError
        if not isinstance(exc, APIError):
            return result
        request_id = identifier(getattr(exc, "request_id", None), 200, r"[A-Za-z0-9][A-Za-z0-9_.:-]*")
        if request_id:
            result["request_id"] = request_id
        if isinstance(exc, APIStatusError):
            content_type = exc.response.headers.get("content-type", "").split(";", 1)[0].strip()
            content_type = identifier(content_type, 100, r"[A-Za-z0-9.+-]+/[A-Za-z0-9.+-]+")
            if content_type:
                result["content_type"] = content_type
        body = exc.body
        error = None
        if type(body) is dict:
            candidate = body.get("error", body)
            if type(candidate) is dict and any(name in candidate for name in ("message", "code", "type", "param")):
                error = candidate
            result["body_format"] = "error_object" if error is not None else "json_object"
        elif type(body) is str:
            result["body_format"] = "html" if body.lstrip().startswith("<") else "text"
        else:
            result["body_format"] = "none" if body is None else "json_array" if type(body) is list else "other"
        fields = {}
        if error is not None:
            for name, limit, pattern in (
                ("code", 128, r"[A-Za-z_][A-Za-z0-9_.:-]*"),
                ("type", 128, r"[A-Za-z_][A-Za-z0-9_.:-]*"),
                ("param", 256, r"[A-Za-z0-9_][A-Za-z0-9_.\[\]/-]*"),
            ):
                value = identifier(error.get(name), limit, pattern)
                if value:
                    fields[name] = value
            message = clean(error.get("message"), 2000)
            if message:
                fields["message"] = message
        elif body is None:
            # SDK status messages often contain a stringified body. Only a plain
            # message with no body is eligible for this conservative fallback.
            message = getattr(exc, "message", None)
            if (type(message) is str and not message.lstrip().startswith(("Error code:", "{", "[", "<"))
                    and "\n" not in message and "\r" not in message):
                message = clean(message, 2000)
                if message:
                    fields["message"] = message
        if fields:
            result["error"] = fields
    except Exception:
        # Diagnostics must not replace the error being diagnosed.
        pass
    return result


def is_http_400(exc: Exception) -> bool:
    return is_http_status(exc, 400)


def is_http_status(exc: Exception, status: int) -> bool:
    try:
        from openai import APIStatusError
        return (isinstance(exc, APIStatusError) and exc.status_code == status
                and exc.response.status_code == status)
    except Exception:
        return False


def failure_code(value):
    code = value.get("code") if isinstance(value, dict) else None
    return code if isinstance(code, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code) else None


def make_client(project_id: str | None = None, *, timeout: float = 60):
    if project_id is not None and (type(project_id) is not str or not PROJECT_ID_PATTERN.fullmatch(project_id)):
        raise PilotError("Invalid project ID for the API client.")
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise PilotError("OPENAI_API_KEY is not configured in this process. Do not paste it into chat.")
    try:
        version = importlib.metadata.version("openai")
        parts = tuple(int(p) for p in version.split(".")[:2])
        if parts < (3, 49) or parts >= (4, 0):
            raise PilotError("Use the isolated pilot environment with the checked openai 3.49 SDK.")
        from openai import OpenAI
    except importlib.metadata.PackageNotFoundError as exc:
        raise PilotError("Install requirements-pilot.txt in the isolated pilot environment.") from exc
    # Never inherit a different provider's OPENAI_BASE_URL. Disable SDK POST retries.
    return OpenAI(api_key=key, base_url="https://api.openai.com/v1", max_retries=0, timeout=timeout,
                  **({"project": project_id} if project_id is not None else {}))


def client_for_run(run_dir: Path):
    """Keep lifecycle operations bound even if the original inputs are damaged.

    This validates routing from the receipt only. run() separately validates the
    entire immutable packet before any new submission.
    """
    state = receipt(run_dir)
    client_options = {"timeout": 600} if state.get("service_tier") == "flex" else {}
    if "budget_control" in state:
        policy = state["budget_control"]
        if type(policy) is dict and policy.get("kind") == "unlimited":
            _checked_unlimited_budget(state)
            return make_client(**client_options)
        policy = _checked_project_budget(policy, state.get("spend_limit_cents"))
        return make_client(project_id=policy["project_id"], **client_options)
    return make_client(**client_options)


def record_event(run_dir: Path, event):
    value = as_dict(event)
    changes = {"last_event": value.get("type")}
    session = value.get("session") or {}
    session_id = value.get("session_id") or session.get("id")
    if session_id:
        changes["session_id"] = session_id
    if session.get("usage") is not None:
        changes.update(session_usage=session["usage"], latest_session_usage_unknown=False,
                       session_usage_scope="session", session_usage_is_provisional=True)
    turn = value.get("turn") or {}
    if turn and turn.get("subagent_id") is None and turn.get("id"):
        changes["turn_id"] = turn["id"]
        usage = turn.get("usage") if turn.get("usage") is not None else value.get("usage")
        if usage is not None:
            changes["usage"] = usage
            changes.update(usage_scope="root_turn", usage_is_provisional=True)
        if turn.get("status") in TERMINAL:
            changes["phase"] = turn["status"]
            changes["failure_code"] = failure_code(turn.get("error"))
    if value.get("type") in {"agent.session.failed", "agent.session.environment.failed"}:
        changes["phase"] = "failed"
    return update_receipt(run_dir, **changes)


def request_cancel(client, run_dir: Path, reason: str = "operator") -> dict:
    state = receipt(run_dir)
    if state["phase"] in TERMINAL:
        return state
    if not state.get("session_id"):
        raise PilotError("Session ID is unknown. Inspect Platform sessions; do not resubmit this run.")
    try:
        client.beta.agents.sessions.events.create(
            state["session_id"], events=[{"type": "agent.session.input.cancel"}]
        )
    except Exception as exc:
        return update_receipt(run_dir, cancel_requested=True, cancel_reason=reason,
                              cancel_confirmed=False, cancel_error=safe_error(exc))
    return update_receipt(run_dir, cancel_requested=True, cancel_reason=reason,
                          cancel_confirmed=False)


def save_output(run_dir: Path, text: str):
    if not isinstance(text, str) or not text.strip():
        raise PilotError("The completed turn has no final text; inspect persisted items.")
    if SECRET_PATTERN.search(text):
        raise PilotError("Output appears to contain an API secret; output was not saved.")
    atomic_bytes(run_dir / "dossier.md", text.encode("utf-8"))
    update_receipt(run_dir, dossier_sha256=digest(text.encode()), dossier_file="dossier.md")


def fetch_pages(fetch, **kwargs):
    """Consume complete pages with the server cursor, never derive a cursor from items."""
    cursor = None
    seen = set()
    for _ in range(20):
        page = as_dict(fetch(**kwargs, **({"after": cursor} if cursor else {})))
        data = page.get("data")
        if not isinstance(data, list):
            raise PilotError("Unsupported API page shape.")
        yield from data
        if not page.get("has_more"):
            return
        cursor = page.get("last_id")
        if not isinstance(cursor, str) or not cursor or cursor in seen:
            raise PilotError("Missing or repeated API cursor; refusing a partial result.")
        seen.add(cursor)
    raise PilotError("Saved history exceeds the pilot retrieval bound; inspect the session directly.")


def recover(client, run_dir: Path):
    """One read-only reconciliation. Does not reconnect a stream or submit any input."""
    state = receipt(run_dir)
    if state["phase"] in {"prepared", "creation_rejected"} or state.get("session_deleted"):
        return state
    session_id = state.get("session_id")
    if not session_id:
        raise PilotError("Creation outcome is unknown. Inspect Platform sessions before any new run.")
    sessions = client.beta.agents.sessions
    session = as_dict(sessions.retrieve(session_id))
    session_usage = session.get("usage")
    spend = session.get("spend_control")
    safe_spend = None
    if isinstance(spend, dict) and type(spend.get("limit")) is int:
        consumed = spend.get("consumed")
        safe_spend = {"limit": spend["limit"],
                      "consumed": consumed if type(consumed) is int and consumed >= 0 else None}
    latest_spend_unknown = safe_spend is None or safe_spend["consumed"] is None
    if safe_spend is not None and latest_spend_unknown:
        previous_spend = state.get("spend_control") or {}
        if previous_spend.get("limit") == safe_spend["limit"]:
            safe_spend["consumed"] = previous_spend.get("consumed")
    update_receipt(run_dir, session_status=session.get("status"),
                   requires_action=bool(session.get("required_actions")),
                   spend_control=safe_spend if safe_spend is not None else state.get("spend_control"),
                   latest_spend_unknown=latest_spend_unknown,
                   spend_is_provisional=True,
                   session_usage=session_usage if session_usage is not None else state.get("session_usage"),
                   latest_session_usage_unknown=session_usage is None,
                   session_usage_scope="session", session_usage_is_provisional=True)
    if not state.get("turn_id"):
        roots = [t for t in fetch_pages(sessions.turns.list, session_id=session_id,
                                        order="asc", limit=100) if t.get("subagent_id") is None]
        if not roots:
            phase = "failed" if session.get("status") == "failed" else "waiting_for_turn"
            return update_receipt(run_dir, phase=phase)
        if len(roots) != 1:
            raise PilotError("Multiple root turns found without a saved target; operator reconciliation required.")
        state = update_receipt(run_dir, turn_id=roots[0]["id"])
    turn_id = state["turn_id"]
    turn = as_dict(sessions.turns.retrieve(turn_id=turn_id, session_id=session_id))
    if turn.get("id") != turn_id or turn.get("session_id") != session_id or turn.get("subagent_id") is not None:
        raise PilotError("Retrieved turn does not match this run's root turn.")
    phase = turn.get("status")
    if phase not in TERMINAL | {"queued", "in_progress", "waiting"}:
        raise PilotError("Unknown turn status; no success inferred.")
    latest_usage = turn.get("usage")
    state = update_receipt(run_dir, phase=phase,
                           usage=latest_usage if latest_usage is not None else receipt(run_dir).get("usage"),
                           latest_usage_unknown=latest_usage is None,
                           usage_scope="root_turn", usage_is_provisional=True,
                           failure_code=failure_code(turn.get("error")),
                           cancel_confirmed=phase == "cancelled")
    if phase != "completed":
        return state
    items = list(fetch_pages(sessions.turns.items.list, session_id=session_id,
                            turn_id=turn_id, order="asc", limit=100))
    final = []
    for item in items:
        if (item.get("type") == "message" and item.get("role") == "assistant"
                and item.get("phase") == "final_answer" and item.get("status") == "completed"
                and item.get("turn_id") == turn_id):
            final.extend(part["text"] for part in item.get("content", [])
                         if part.get("type") == "output_text" and isinstance(part.get("text"), str))
    text = "\n\n".join(final)
    if not text.strip():
        return update_receipt(run_dir, result_status="missing_final")
    save_output(run_dir, text)
    return update_receipt(run_dir, result_status="available")


def close_session(client, run_dir: Path):
    state = receipt(run_dir)
    if state.get("session_deleted"):
        return state
    if state["phase"] not in TERMINAL or not state.get("session_id"):
        raise PilotError("First verify a terminal turn with status/resume or request cancel.")
    if state["phase"] == "completed" and not state.get("dossier_file"):
        raise PilotError("Save the final dossier before deleting this session.")
    client.beta.agents.sessions.delete(state["session_id"])
    return update_receipt(run_dir, session_deleted=True, session_deleted_at=now(),
                           environment_cleanup="asynchronous")


def run(client, run_dir: Path, max_seconds: int = 900, project_limit_confirmed: bool = False):
    with locked(run_dir):
        state = receipt(run_dir)
        if state["phase"] != "prepared":
            raise PilotError("This run was already submitted or has an uncertain outcome. Use status/resume.")
        raw = (run_dir / "request.json").read_bytes()
        if digest(raw) != state["request_sha256"]:
            raise PilotError("Prepared request changed; refusing to submit it.")
        request = json.loads(raw)
        policy = budget_policy(run_dir, request)
        if type(project_limit_confirmed) is not bool:
            raise PilotError("Project hard-limit confirmation must be an explicit boolean flag.")
        if policy["kind"] == "project_hard_limit":
            if not project_limit_confirmed:
                raise PilotError("Confirm the configured monthly project hard limit with --project-limit-confirmed before run.")
            if getattr(client, "project", None) != policy["project_id"]:
                raise PilotError("The API client is not bound to the saved budget project; no request was submitted.")
            state["project_limit_confirmation"] = {
                "confirmed_at": now(), "verified_by_api": False,
                "project_id": policy["project_id"], "monthly_limit_cents": policy["monthly_limit_cents"],
            }
        elif project_limit_confirmed:
            raise PilotError("--project-limit-confirmed is only valid for a prepared project hard-limit run.")
        # Persist BEFORE the potentially billable POST. A crash never enables automatic replay.
        state.update(phase="submitting", submitted_at=now(), api_stage="create_request",
                     deadline_epoch=time.time() + max_seconds, submission_attempts=1)
        atomic_bytes(run_dir / "receipt.json", encode(state))
    timer = None
    created_stream = None
    create_accepted = False
    api_stage = "create_request"
    try:
        created_stream = client.beta.agents.sessions.create(**request, stream=True)
        # create() has received a successful HTTP response. Any failure after
        # this point is not evidence that the server rejected session creation.
        create_accepted = True
        api_stage = "observe_stream"
        state = update_receipt(run_dir, api_stage=api_stage, create_accepted=True)
        with created_stream.with_result_collection() as stream:
            for event in stream:
                state = record_event(run_dir, event)
                # Creation streams may stay open at idle. Only the initial root
                # turn's confirmed terminal state makes this a result boundary.
                if as_dict(event).get("type") == "agent.session.idle" and state["phase"] in TERMINAL:
                    break
                if timer is None and state.get("session_id"):
                    delay = max(0, state["deadline_epoch"] - time.time())
                    def deadline_cancel():
                        try:
                            request_cancel(client, run_dir, "deadline")
                        except Exception:
                            # A final synchronous status check remains necessary. Never dump client details.
                            pass
                    timer = threading.Timer(delay, deadline_cancel)
                    timer.daemon = True
                    timer.start()
            result = stream.get_final_result()
        api_stage = "recover_saved_output"
        state = update_receipt(run_dir, session_id=result.session_id, api_stage=api_stage)
        # Reconcile persisted root turn + final items, not intermediate commentary.
        return recover(client, run_dir)
    except KeyboardInterrupt as exc:
        try:
            if receipt(run_dir).get("session_id"):
                request_cancel(client, run_dir, "keyboard_interrupt")
        except Exception:
            pass
        try:
            update_receipt(run_dir, observer_interrupted=True, api_stage=api_stage)
        except Exception:
            pass
        raise PilotError("Observation interrupted. Cancellation is best effort; use status to verify.") from exc
    except Exception as exc:
        current = state
        try:
            current = receipt(run_dir)
        except Exception:
            pass
        rejected = api_stage == "create_request" and not create_accepted and is_http_400(exc)
        diagnostic = safe_error(exc)
        capacity_rejected = (api_stage == "create_request" and not create_accepted
            and not current.get("session_id") and request["agent"].get("service_tier") == "flex"
            and is_http_status(exc, 429)
            and diagnostic.get("http_status") == 429
            and diagnostic.get("error", {}).get("code") == "resource_unavailable")
        changes = {"observer_error": diagnostic, "api_stage": api_stage,
                   "create_accepted": create_accepted}
        if capacity_rejected:
            changes.update(phase="capacity_unavailable", failure_code="resource_unavailable")
        elif rejected:
            changes["phase"] = "creation_rejected"
        elif not current.get("session_id"):
            changes["phase"] = "creation_unknown"
        elif current.get("phase") not in TERMINAL:
            changes["phase"] = "observation_interrupted"
        diagnostic_saved = False
        try:
            update_receipt(run_dir, **changes)
            diagnostic_saved = True
        except Exception:
            pass
        if current.get("session_id") and current.get("phase") not in TERMINAL:
            # The process is about to close the client and its watchdog. Request
            # cancellation now rather than silently leaving an unbounded cloud turn.
            try:
                request_cancel(client, run_dir, "observer_error")
            except Exception:
                pass
        if isinstance(exc, PilotError):
            raise
        diagnostic = ("See receipt.json observer_error." if diagnostic_saved
                      else "The receipt diagnostic could not be saved.")
        if capacity_rejected:
            message = "Flex capacity is unavailable; this session creation was rejected."
        elif rejected:
            message = "The API rejected session creation (HTTP 400)."
        elif api_stage == "create_request":
            message = "Session creation could not be confirmed. Inspect Platform sessions."
        elif api_stage == "recover_saved_output":
            message = "Retrieval of saved output failed. Use status/resume with the saved identifiers."
        else:
            message = "The observer stopped. Saved identifiers may permit status/resume."
        raise PilotError(f"{message} {diagnostic} Never repeat run.") from exc
    finally:
        if timer is not None:
            timer.cancel()
        if created_stream is not None:
            try:
                close = getattr(created_stream, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Build and hash inputs offline; no SDK/key needed")
    prep.add_argument("--skill-dir", type=Path, required=True)
    prep.add_argument("--snapshot", type=Path, required=True)
    prep.add_argument("--run-dir", type=Path, required=True)
    prep.add_argument("--model", default=DEFAULT_MODEL)
    prep.add_argument("--include-ai-guide", action="store_true")
    prep.add_argument("--spend-limit-cents", type=int, default=DEFAULT_SPEND_LIMIT_CENTS,
                      help="Session limit, or project monthly hard limit with --project-id, in USD cents (default: 300)")
    prep.add_argument("--project-id", help="Select externally configured project monthly hard-limit mode (maximum 300 cents)")
    prep.add_argument("--no-spend-limit", action="store_true",
                      help="Explicitly prepare this pilot without a session or project monetary limit")
    prep.add_argument("--research-mode", choices=tuple(research_context.MODE_INSTRUCTIONS), default="general",
                      help="general topic research or targeted verification of supplied new_signals")
    prep.add_argument("--research-as-of", help="Explicit window end with timezone; defaults to preparation time")
    prep.add_argument("--reasoning-effort", choices=sorted(REASONING_EFFORTS),
                      help="Opt-in reasoning effort; omitted requests keep the model default")
    for name in ("run", "status", "resume", "cancel", "close"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
        if name == "run":
            command.add_argument("--max-seconds", type=int, default=900)
            command.add_argument("--project-limit-confirmed", action="store_true",
                                 help="Confirm the stated monthly hard limit is enforced on the saved project; not an API verification")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.skill_dir, args.snapshot, args.run_dir, args.model,
                             args.include_ai_guide, args.spend_limit_cents, args.project_id, args.no_spend_limit,
                             research_mode=args.research_mode, research_as_of=args.research_as_of,
                             reasoning_effort=args.reasoning_effort)
            summary = {"phase": "prepared", "run_id": result["run_id"],
                       "request_bytes": result["request_bytes"], "request_sha256": result["request_sha256"],
                       "model": result["model"], "research_mode": result["research_mode"],
                       "research_as_of": result["research_as_of"], "research_window": result["research_window"]}
            if "budget_control" in result:
                summary["budget_control"] = result["budget_control"]
            print(json.dumps(summary, ensure_ascii=False))
        else:
            if args.run_dir.is_symlink() or not args.run_dir.is_dir():
                raise PilotError("Expected an existing regular run directory.")
            if args.command == "run" and not 30 <= args.max_seconds <= 3600:
                raise PilotError("--max-seconds must be from 30 to 3600; it is not a dollar cap.")
            with client_for_run(args.run_dir) as client:
                if args.command == "run":
                    result = run(client, args.run_dir, args.max_seconds, args.project_limit_confirmed)
                elif args.command == "cancel":
                    result = request_cancel(client, args.run_dir)
                elif args.command == "close":
                    result = close_session(client, args.run_dir)
                else:
                    result = recover(client, args.run_dir)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if result.get("phase") in {"failed", "cancelled", "creation_unknown", "creation_rejected"} or result.get("result_status") == "missing_final":
                return 2
        return 0
    except PilotError as exc:
        print(f"Pilot: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        # Only bounded, redacted, allowlisted SDK fields may reach the terminal.
        print(json.dumps({"error": safe_error(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
