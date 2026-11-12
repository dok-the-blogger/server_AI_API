"""Offline preflight and entry point for the Windows launcher; never reads a key file."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

if __package__:
    from . import context as research_context
    from . import pilot
else:
    import context as research_context
    import pilot


def timestamp(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None:
            raise ValueError()
        return result.astimezone(timezone.utc)
    except (AttributeError, TypeError, ValueError) as exc:
        raise pilot.PilotError("Invalid prepared timestamp; prepare the packet again.") from exc


def preflight(run_dir: Path, action: str, max_snapshot_hours: float = 6,
              current_time: datetime | None = None,
              project_limit_confirmed: bool = False) -> dict:
    """Validate locally before collecting credentials, and again before a paid run."""
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise pilot.PilotError("Prepared run directory is missing or is a link.")
    state = pilot.receipt(run_dir)
    if project_limit_confirmed and action != "run":
        raise pilot.PilotError("Project-limit confirmation is only accepted for a new run.")
    if action == "inspect":
        if __package__:
            from . import inspect_pilot
        else:
            import inspect_pilot
        return inspect_pilot.preflight(run_dir)
    if action != "run":
        if not state.get("session_id") or state.get("session_deleted"):
            raise pilot.PilotError("No saved session can be inspected. Check Platform if creation is unknown.")
        return {"action": action, "phase": state.get("phase"), "new_research": False}
    if (state.get("phase") != "prepared" or state.get("submission_attempts") != 0
            or state.get("session_id") or state.get("turn_id")):
        raise pilot.PilotError("This packet was already submitted or its outcome is unknown. Use resume/status.")
    if not 0 < max_snapshot_hours <= 24:
        raise pilot.PilotError("Snapshot age limit must be greater than zero and at most 24 hours.")
    path = run_dir / "request.json"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > pilot.MAX_REQUEST_BYTES:
        raise pilot.PilotError("Prepared request is missing, is a link, or exceeds the size bound.")
    raw = path.read_bytes()
    if pilot.SECRET_PATTERN.search(raw.decode("utf-8")):
        raise pilot.PilotError("Prepared request appears to contain a secret; refusing to launch.")
    manifest = pilot.read_json(run_dir / "manifest.json")
    if (manifest.get("schema_version") != 1
            or manifest.get("run_id") != state.get("run_id")
            or manifest.get("request_sha256") != pilot.digest(raw)
            or state.get("request_sha256") != pilot.digest(raw)):
        raise pilot.PilotError("Prepared request, manifest and receipt do not match.")
    request = json.loads(raw)
    budget = pilot.budget_policy(run_dir, request)
    project_budget = budget["kind"] == "project_hard_limit"
    unlimited_budget = budget["kind"] == "unlimited"
    if project_budget and project_limit_confirmed is not True:
        raise pilot.PilotError(
            "Confirm the configured hard monthly limit for this exact Platform project before running. "
            "The pilot cannot verify that setting with its ordinary API key.")
    if not project_budget and project_limit_confirmed:
        raise pilot.PilotError("Project-limit confirmation is only valid for a project-limit packet.")
    # This launcher accepts only the small research-only configuration we prepared.
    expected_keys = {"agent", "environment", "input", "metadata"}
    if budget["kind"] == "session":
        expected_keys.add("spend_control")
    if set(request) != expected_keys:
        raise pilot.PilotError("Unexpected session configuration; prepare a supported research packet.")
    agent = request.get("agent")
    expected_tools = [{"type": "web_search", "mode": "live", "context_size": "medium"}]
    expected_agent_keys = {"model", "instructions", "tools", "service_tier"}
    reasoning_effort = manifest.get("reasoning_effort")
    if reasoning_effort is not None:
        if type(reasoning_effort) is not str or reasoning_effort not in pilot.REASONING_EFFORTS:
            raise pilot.PilotError("Prepared reasoning effort is unsupported.")
        expected_agent_keys.add("reasoning")
    if (not isinstance(agent, dict) or set(agent) != expected_agent_keys
            or agent.get("tools") != expected_tools
            or agent.get("service_tier") != "default"
            or request.get("environment") != {"type": "openai_hosted"}
            or agent.get("model") != manifest.get("model")
            or (reasoning_effort is not None and agent.get("reasoning") != {"effort": reasoning_effort})):
        raise pilot.PilotError("Packet configuration differs from the reviewed web-research pilot.")
    limit = None
    if not unlimited_budget:
        limit = budget["monthly_limit_cents"] if project_budget else budget["limit_cents"]
        if (type(limit) is not int or not 1 <= limit <= 300
                or limit != manifest.get("spend_limit_cents") or limit != state.get("spend_limit_cents")):
            raise pilot.PilotError("Spend threshold must match the prepared packet and be at most 300 USD cents.")
    if request.get("metadata") != {"dok_research_run_id": state["run_id"],
                                   "purpose": "doknews_manual_research_pilot"}:
        raise pilot.PilotError("Session metadata does not match this research run.")
    body = request.get("input")
    try:
        snapshot = research_context.extract_snapshot(body, manifest)
    except research_context.ContextError as exc:
        raise pilot.PilotError(str(exc)) from exc
    pilot.validate_snapshot(snapshot)
    layout = manifest.get("context_layout_version", 1)
    snapshot_bytes = (research_context.canonical_json(snapshot).encode("utf-8")
                      if layout == 2 else pilot.encode(snapshot))
    if (pilot.digest(snapshot_bytes) != manifest.get("snapshot_sha256")
            or snapshot.get("captured_at") != manifest.get("snapshot_captured_at")
            or snapshot.get("coverage") != manifest.get("coverage")):
        raise pilot.PilotError("Snapshot provenance does not match the prepared manifest.")
    if layout == 2:
        try:
            built = research_context.build_context(snapshot, state["run_id"],
                                                   research_mode=manifest["research_mode"],
                                                   research_as_of=manifest["research_as_of"])
        except research_context.ContextError as exc:
            raise pilot.PilotError(str(exc)) from exc
        instructions = agent.get("instructions")
        if (type(instructions) is not str
                or pilot.digest(instructions.encode("utf-8")) != manifest.get("instructions_sha256")
                or pilot.digest((instructions + "\n\n" + built.stable_prefix).encode("utf-8")) != manifest.get("context_prefix_sha256")
                or any(state.get(key) != manifest[key] for key in ("research_mode", "research_as_of", "research_window"))):
            raise pilot.PilotError("Research context provenance or window does not match the prepared packet.")
    current_time = current_time or datetime.now(timezone.utc)
    age = (current_time - timestamp(snapshot["captured_at"])).total_seconds()
    if age < -300:
        raise pilot.PilotError("Snapshot is in the future. Check the clock and prepare a fresh packet.")
    if age > max_snapshot_hours * 3600:
        raise pilot.PilotError("Snapshot is stale. Prepare fresh source data before starting a paid run.")
    if layout == 2:
        research_age = (current_time - timestamp(manifest["research_as_of"])).total_seconds()
        if research_age < -300 or research_age > max_snapshot_hours * 3600:
            raise pilot.PilotError("Research window is stale or in the future; prepare a fresh packet.")
    prepared_age = (current_time - timestamp(manifest.get("created_at"))).total_seconds()
    if prepared_age < -300:
        raise pilot.PilotError("Prepared manifest is in the future. Check the local clock.")
    result = {"action": "run", "phase": "prepared", "run_id": state["run_id"],
              "model": agent["model"], "snapshot_age_minutes": round(max(0, age) / 60, 1),
              "new_research": True}
    if layout == 2:
        result.update(research_mode=manifest["research_mode"], research_as_of=manifest["research_as_of"],
                      research_window=manifest["research_window"], context_layout_version=layout,
                      context_prefix_sha256=manifest["context_prefix_sha256"])
    if unlimited_budget:
        result.update(budget_mode="unlimited", spend_limit_cents=None)
    elif project_budget:
        result.update(project_id=budget["project_id"], project_monthly_limit_cents=limit,
                      project_limit_confirmed_by_operator=True, project_limit_verified_by_api=False)
    else:
        result["spend_limit_cents"] = limit
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "execute"))
    parser.add_argument("--action", choices=("run", "status", "resume", "cancel", "inspect"), default="run")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--max-snapshot-hours", type=float, default=6)
    parser.add_argument("--max-seconds", type=int, default=900)
    parser.add_argument("--project-limit-confirmed", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not 30 <= args.max_seconds <= 3600:
            raise pilot.PilotError("Observation limit must be from 30 to 3600 seconds.")
        result = preflight(args.run_dir, args.action, args.max_snapshot_hours,
                           project_limit_confirmed=args.project_limit_confirmed)
        if args.command == "check":
            print(json.dumps(result, ensure_ascii=True))
            return 0
        if args.action == "inspect":
            if __package__:
                from . import inspect_pilot
            else:
                import inspect_pilot
            return inspect_pilot.execute(args.run_dir)
        command = [args.action, "--run-dir", str(args.run_dir)]
        if args.action == "run":
            command += ["--max-seconds", str(args.max_seconds)]
            if args.project_limit_confirmed:
                command.append("--project-limit-confirmed")
        return pilot.main(command)
    except pilot.PilotError as exc:
        print(f"Pilot: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(json.dumps({"error": {"type": type(exc).__name__}}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
