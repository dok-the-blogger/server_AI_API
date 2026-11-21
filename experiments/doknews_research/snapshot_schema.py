"""Structural validation shared by the research service and the Work journal writer."""
from datetime import datetime
import re


SHA256 = re.compile(r"[a-f0-9]{64}")


class SnapshotShapeError(ValueError):
    """The payload is structurally invalid regardless of queue state or time."""


def validate_snapshot_shape(value: dict) -> None:
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise SnapshotShapeError("Snapshot schema_version must be 1.")
    task = value.get("task")
    if not isinstance(task, dict) or not all(isinstance(task.get(k), str) and task[k].strip()
                                             for k in ("name", "topic")):
        raise SnapshotShapeError("Snapshot task requires a nonempty name and topic.")
    hours = task.get("time_window_hours")
    if type(hours) is not int or not 1 <= hours <= 168:
        raise SnapshotShapeError("Snapshot time window must be an integer from 1 to 168 hours.")
    try:
        captured = datetime.fromisoformat(value["captured_at"].replace("Z", "+00:00"))
        if captured.tzinfo is None:
            raise ValueError()
    except (KeyError, ValueError, AttributeError, TypeError) as exc:
        raise SnapshotShapeError("Snapshot captured_at must be an ISO timestamp with a timezone.") from exc
    coverage = value.get("coverage")
    if not isinstance(coverage, dict):
        raise SnapshotShapeError("Snapshot must describe source coverage.")
    if any(coverage.get(key) not in ("not_provided", "partial", "complete")
           for key in ("archive", "radar", "memory")):
        raise SnapshotShapeError("Snapshot coverage must be not_provided, partial or complete.")
    if not isinstance(coverage.get("missing"), list) or not all(isinstance(x, str) for x in coverage["missing"]):
        raise SnapshotShapeError("Snapshot coverage.missing must be an array of strings.")
    for key in ("radar_signals", "archive_articles", "memory_records"):
        if not isinstance(value.get(key), list) or not all(isinstance(x, dict) for x in value[key]):
            raise SnapshotShapeError(f"snapshot.{key} must be an array of objects.")
    seen = set()
    for signal in value["radar_signals"]:
        signal_id, version = signal.get("signal_id"), signal.get("version")
        if not isinstance(signal_id, str) or not signal_id.strip() or len(signal_id) > 512:
            raise SnapshotShapeError("Every radar signal requires a nonempty signal_id.")
        if type(version) is not int or not 1 <= version <= 2**63 - 1:
            raise SnapshotShapeError("Every radar signal requires a positive integer version.")
        if (signal_id, version) in seen:
            raise SnapshotShapeError("Snapshot contains a duplicate signal_id and version.")
        seen.add((signal_id, version))
        for key in ("source_id", "source_url", "title"):
            if not isinstance(signal.get(key), str) or not signal[key].strip():
                raise SnapshotShapeError("Every radar signal requires source_id, source_url and title.")
        for key in ("text", "excerpt", "summary", "body"):
            if key in signal and signal[key] is not None and not isinstance(signal[key], str):
                raise SnapshotShapeError("Radar signal text fields must contain text.")
        supplied_hash = signal.get("content_sha256")
        if supplied_hash is not None and (not isinstance(supplied_hash, str) or not SHA256.fullmatch(supplied_hash)):
            raise SnapshotShapeError("Radar content_sha256 must be a lowercase SHA256 digest.")
