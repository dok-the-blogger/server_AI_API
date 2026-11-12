"""Configuration shared by the small HTTP control plane and isolated worker."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re


# Leave room for the ledger's selection/provenance additions under the pilot's
# 512,000-byte local input-file bound.
MAX_SNAPSHOT_BYTES = 480_000
MAX_DOSSIER_BYTES = 2_000_000
CONSUMER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
SECRET = re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}")


class ResearchError(Exception):
    def __init__(self, code: str, message: str, status: int = 503):
        super().__init__(message)
        self.code, self.status = code, status


@dataclass(frozen=True)
class ServerConfig:
    state_dir: Path
    skill_dir: Path
    worker_python: Path
    openai_key_file: Path | None = None
    api_enabled: bool = False
    consumers: tuple[str, ...] = ("doknews-api-hourly",)
    model: str = "gpt-6.1-sol"
    max_signals: int = 3
    max_snapshot_hours: float = 1.0
    max_seconds: int = 900
    poll_seconds: float = 5.0
    reasoning_effort: str = "medium"

    @property
    def db_path(self) -> Path:
        return self.state_dir / "research.sqlite3"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_bytes(value) -> bytes:
    try:
        return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError, RecursionError, UnicodeError) as exc:
        raise ResearchError("invalid_json", "Research input must contain valid JSON data.", 422) from exc


def timestamp(value) -> datetime:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None:
            raise ValueError()
        return result.astimezone(timezone.utc)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ResearchError("invalid_timestamp", "Research timestamp must include a timezone.", 422) from exc


def _path(value, name: str, optional=False):
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ResearchError("invalid_config", f"{name} must be an absolute configured path.")
    return Path(value)


def load_config(path: Path) -> ServerConfig:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 16_384:
            raise ResearchError("invalid_config", "Research configuration must be a bounded regular file.")
        raw = path.read_bytes()
        obj = json.loads(raw)
        if not isinstance(obj, dict) or set(obj) - set(ServerConfig.__dataclass_fields__):
            raise ValueError()
        consumers = obj.get("consumers", ["doknews-api-hourly"])
        if not isinstance(consumers, list) or not all(isinstance(item, str) for item in consumers):
            raise ValueError()
        if SECRET.search(raw.decode("utf-8")):
            raise ResearchError("secret_in_config", "Keep the OpenAI key in its separate credential file.")
        cfg = ServerConfig(
            **{key: value for key, value in obj.items()
               if key not in ("state_dir", "skill_dir", "worker_python", "openai_key_file", "consumers")},
            state_dir=_path(obj.get("state_dir"), "state_dir"),
            skill_dir=_path(obj.get("skill_dir"), "skill_dir"),
            worker_python=_path(obj.get("worker_python"), "worker_python"),
            openai_key_file=_path(obj.get("openai_key_file"), "openai_key_file", optional=True),
            consumers=tuple(consumers),
        )
    except ResearchError:
        raise
    except (OSError, TypeError, ValueError, UnicodeError) as exc:
        raise ResearchError("invalid_config", "Research configuration could not be loaded.") from exc
    if (type(cfg.api_enabled) is not bool or cfg.model != "gpt-6.1-sol"
            or type(cfg.max_signals) is not int or not 1 <= cfg.max_signals <= 3
            or not isinstance(cfg.max_snapshot_hours, (int, float)) or isinstance(cfg.max_snapshot_hours, bool)
            or not 0 < cfg.max_snapshot_hours <= 6
            or type(cfg.max_seconds) is not int or not 30 <= cfg.max_seconds <= 3600
            or not isinstance(cfg.poll_seconds, (int, float)) or isinstance(cfg.poll_seconds, bool)
            or not 1 <= cfg.poll_seconds <= 30 or cfg.reasoning_effort not in ("low", "medium", "high")
            or not cfg.consumers or len(cfg.consumers) > 16
            or len(set(cfg.consumers)) != len(cfg.consumers)
            or any(not isinstance(x, str) or not CONSUMER.fullmatch(x) for x in cfg.consumers)):
        raise ResearchError("invalid_config", "Research configuration is outside the supported Sol profile.")
    repo = Path(__file__).resolve().parents[2]
    state = cfg.state_dir.resolve()
    if state == Path(state.anchor) or state == repo or repo in state.parents or state in repo.parents:
        raise ResearchError("invalid_config", "Use a dedicated research state directory outside the checkout.")
    if cfg.state_dir.is_symlink():
        raise ResearchError("invalid_config", "Research state directory cannot be a link.")
    return cfg


def ensure_state(cfg: ServerConfig):
    try:
        cfg.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if cfg.state_dir.is_symlink() or not cfg.state_dir.is_dir():
            raise OSError()
        os.chmod(cfg.state_dir, 0o700)
        for name in ("runs", "inputs"):
            child = cfg.state_dir / name
            if child.is_symlink():
                raise OSError()
            child.mkdir(mode=0o700, exist_ok=True)
            os.chmod(child, 0o700)
    except OSError as exc:
        raise ResearchError("state_unavailable", "Private research state is unavailable.") from exc


def validate_snapshot(snapshot: dict, cfg: ServerConfig, *, check_freshness=True):
    raw = json_bytes(snapshot)
    if len(raw) > MAX_SNAPSHOT_BYTES:
        raise ResearchError("snapshot_too_large", "Research snapshot exceeds the size limit.", 413)
    if SECRET.search(raw.decode("utf-8")):
        raise ResearchError("secret_in_snapshot", "Research snapshot appears to contain an API credential.", 422)
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 1:
        raise ResearchError("invalid_snapshot", "Research snapshot schema_version must be 1.", 422)
    captured = timestamp(snapshot.get("captured_at"))
    if check_freshness:
        age = (datetime.now(timezone.utc) - captured).total_seconds()
        if age < -300 or age > cfg.max_snapshot_hours * 3600:
            raise ResearchError("stale_snapshot", "Prepare current source data before new research.", 422)


def read_private_json(path: Path, max_bytes: int = 512_000):
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes:
            raise ValueError()
        with path.open("rb") as stream:
            raw = stream.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise ValueError()
        return json.loads(raw)
    except (OSError, ValueError, UnicodeError) as exc:
        raise ResearchError("state_unavailable", "Saved research state could not be read.") from exc
