"""Read-only API inspection of a saved attempt; never submits research or edits its receipt."""
from __future__ import annotations

import json
from pathlib import Path
import re
import uuid

import pilot
from bootstrap import redact


def prepared_attempt(run_dir: Path) -> tuple[dict, dict]:
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise pilot.PilotError("Expected a regular saved run directory.")
    state = pilot.receipt(run_dir)
    try:
        run_id = state["run_id"]
        if str(uuid.UUID(run_id)) != run_id or state.get("submission_attempts") != 1:
            raise ValueError()
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise pilot.PilotError("Inspection requires one recorded submission attempt.") from exc
    request = pilot.read_json(run_dir / "request.json")
    manifest = pilot.read_json(run_dir / "manifest.json")
    raw = (run_dir / "request.json").read_bytes()
    if (pilot.digest(raw) != state.get("request_sha256")
            or pilot.digest(raw) != manifest.get("request_sha256")
            or manifest.get("run_id") != run_id):
        raise pilot.PilotError("The saved request, manifest and receipt do not match.")
    expected = {"dok_research_run_id": run_id, "purpose": "doknews_manual_research_pilot"}
    model = (request.get("agent") or {}).get("model")
    if (request.get("metadata") != expected or not isinstance(model, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", model)):
        raise pilot.PilotError("The saved request does not identify this research pilot.")
    return state, request


def preflight(run_dir: Path) -> dict:
    state, request = prepared_attempt(run_dir)
    return {"action": "inspect", "phase": state.get("phase"), "run_id": state["run_id"],
            "model": request["agent"]["model"], "new_research": False,
            "api_methods": ["GET"], "receipt_will_change": False}


def safe_label(value, maximum: int = 128):
    return redact(value)[:maximum] if isinstance(value, str) else None


def session_summary(session: dict, model: str) -> dict:
    spend = session.get("spend_control") or {}
    agent = session.get("agent") or {}
    return {"id": safe_label(session.get("id")), "status": safe_label(session.get("status")),
            "created_at": session.get("created_at") if type(session.get("created_at")) is int else None,
            "model_matches_request": agent.get("model") == model,
            "spend_control": {name: spend.get(name) if type(spend.get(name)) is int else None
                              for name in ("limit", "consumed")},
            "spend_is_provisional": True}


def inspect_attempt(client, run_dir: Path) -> dict:
    state, request = prepared_attempt(run_dir)
    original_receipt = (run_dir / "receipt.json").read_bytes()
    model = request["agent"]["model"]
    report = {"schema_version": 1, "observed_at": pilot.now(), "run_id": state["run_id"],
              "recorded_phase": state.get("phase"), "new_research": False,
              "receipt_modified": False, "session_lookup": {}, "model_lookup": {}}
    try:
        # A metadata-filtered GET only. An empty filtered list is not proof of absence.
        page = pilot.as_dict(client.beta.agents.sessions.list(
            metadata=request["metadata"], limit=20, order="desc"))
        data = page.get("data")
        if not isinstance(data, list) or type(page.get("has_more")) is not bool:
            raise pilot.PilotError("The session list has an unsupported shape.")
        matched = [item for item in data if isinstance(item, dict)
                   and isinstance(item.get("metadata"), dict)
                   and all(item["metadata"].get(key) == value
                           for key, value in request["metadata"].items())]
        report["session_lookup"] = {
            "status": "matches_found" if matched else "no_match_in_filtered_results",
            "sessions": [session_summary(item, model) for item in matched],
            "has_more": page["has_more"], "unmatched_records": len(data) - len(matched),
            "absence_is_not_proven": not bool(matched),
            "filtered_results_can_lag": True,
        }
    except Exception as exc:
        report["session_lookup"] = {"status": "error", "error": pilot.safe_error(exc)}
    try:
        model_info = pilot.as_dict(client.models.retrieve(model))
        report["model_lookup"] = {"status": "available", "requested_id": model,
                                  "returned_id": safe_label(model_info.get("id")),
                                  "does_not_validate_session_configuration": True}
    except Exception as exc:
        report["model_lookup"] = {"status": "error", "requested_id": model,
                                  "error": pilot.safe_error(exc)}
    report["receipt_modified"] = (run_dir / "receipt.json").read_bytes() != original_receipt
    report["automatic_retry"] = False
    pilot.atomic_bytes(run_dir / "inspection.json", pilot.encode(report))
    return report


def execute(run_dir: Path) -> int:
    with pilot.client_for_run(run_dir) as client:
        report = inspect_attempt(client, run_dir)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    print(f"Read-only inspection saved: {run_dir / 'inspection.json'}")
    return 2 if any(report[name].get("status") == "error"
                    for name in ("session_lookup", "model_lookup")) else 0
