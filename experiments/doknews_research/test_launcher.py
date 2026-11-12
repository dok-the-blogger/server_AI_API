"""Offline guards around the Windows entry point; no credentials or API calls."""
from datetime import datetime, timedelta, timezone
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

import launcher
import pilot


class LauncherPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.skill = self.base / "skill"
        for name in pilot.POLICY_FILES:
            path = self.skill / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"Test policy: {name}", encoding="utf-8")
        self.snapshot = json.loads((pilot.BASE / "input.example.json").read_text(encoding="utf-8"))
        self.current = datetime.now(timezone.utc)
        self.run_dir = self.base / "prepared-run"

    def prepare(self, age_hours=0.1, limit=300, project_id=None, no_spend_limit=False, **research_options):
        self.snapshot["captured_at"] = (self.current - timedelta(hours=age_hours)).isoformat()
        path = self.base / "snapshot.json"
        path.write_bytes(pilot.encode(self.snapshot))
        return pilot.prepare(self.skill, path, self.run_dir, pilot.DEFAULT_MODEL, False, limit,
                             project_id=project_id, no_spend_limit=no_spend_limit, **research_options)

    def test_fresh_packet_displays_reviewed_threshold(self):
        self.prepare()
        result = launcher.preflight(self.run_dir, "run", current_time=self.current)
        self.assertEqual(result["spend_limit_cents"], 300)
        self.assertTrue(result["new_research"])
        self.assertEqual(result["snapshot_age_minutes"], 6.0)

    def test_new_signals_window_and_explicit_reasoning_pass_preflight(self):
        self.snapshot["radar_signals"] = [{"signal_id": "s1", "version": 2, "text": "A changed signal"}]
        self.prepare(no_spend_limit=True, research_mode="new_signals", reasoning_effort="medium",
                     research_as_of=self.current.isoformat())
        result = launcher.preflight(self.run_dir, "run", current_time=self.current)
        self.assertEqual(result["model"], "gpt-6.1-sol")
        self.assertEqual(result["research_mode"], "new_signals")
        self.assertEqual(result["research_window"], {
            "start_at": (self.current - timedelta(hours=48)).isoformat(),
            "end_at": self.current.isoformat(), "hours": 48})
        self.assertEqual(result["research_as_of"], self.current.isoformat())

    def test_legacy_packet_and_astra_model_remain_unchanged_and_compatible(self):
        manifest = self.prepare()
        request = pilot.read_json(self.run_dir / "request.json")
        request["agent"]["model"] = manifest["model"] = "gpt-6-astra"
        request["input"] = (f"Выполни только исследование. DOK_RESEARCH_RUN_ID={manifest['run_id']}.\n\n"
                            + json.dumps(self.snapshot, ensure_ascii=False))
        fields = ("context_layout_version", "research_mode", "research_as_of", "research_window",
                  "snapshot_hash_format", "context_prefix_sha256", "instructions_sha256",
                  "archive_context_sha256", "memory_context_sha256", "reasoning_effort")
        for field in fields:
            manifest.pop(field, None)
        manifest["snapshot_sha256"] = pilot.digest(pilot.encode(self.snapshot))
        raw = pilot.encode(request)
        manifest["request_sha256"] = pilot.digest(raw)
        state = pilot.receipt(self.run_dir)
        for field in ("research_mode", "research_as_of", "research_window"):
            state.pop(field)
        state["request_sha256"] = pilot.digest(raw)
        pilot.atomic_bytes(self.run_dir / "request.json", raw)
        pilot.atomic_bytes(self.run_dir / "manifest.json", pilot.encode(manifest))
        pilot.atomic_bytes(self.run_dir / "receipt.json", pilot.encode(state))
        before = {name: (self.run_dir / name).read_bytes() for name in ("request.json", "manifest.json", "receipt.json")}
        result = launcher.preflight(self.run_dir, "run", current_time=self.current)
        self.assertEqual(result["model"], "gpt-6-astra")
        self.assertTrue(result["new_research"])
        self.assertNotIn("context_layout_version", result)
        self.assertEqual(before, {name: (self.run_dir / name).read_bytes() for name in before})

    def test_inconsistent_research_window_and_mixed_layout_fail_before_execution(self):
        manifest = self.prepare(research_mode="new_signals", research_as_of=self.current.isoformat())
        for changes in ({"research_window": {**manifest["research_window"], "hours": 24}},
                        {"context_layout_version": 1}, {"context_layout_version": "2"}):
            altered = {**manifest, **changes}
            pilot.atomic_bytes(self.run_dir / "manifest.json", pilot.encode(altered))
            with self.subTest(changes=changes), self.assertRaises(pilot.PilotError):
                launcher.preflight(self.run_dir, "run", current_time=self.current)
        pilot.atomic_bytes(self.run_dir / "manifest.json", pilot.encode(manifest))
        altered = deepcopy(manifest["research_window"])
        altered["hours"] = 24
        pilot.update_receipt(self.run_dir, research_window=altered)
        with self.assertRaises(pilot.PilotError):
            launcher.preflight(self.run_dir, "run", current_time=self.current)
        self.assertEqual(pilot.receipt(self.run_dir)["submission_attempts"], 0)

    def test_future_research_time_is_rejected_even_with_a_fresh_snapshot(self):
        self.prepare(research_as_of=(self.current + timedelta(hours=1)).isoformat())
        with self.assertRaisesRegex(pilot.PilotError, "window.*future"):
            launcher.preflight(self.run_dir, "run", current_time=self.current)

    def test_stale_snapshot_stops_before_pilot_execution(self):
        self.prepare(age_hours=7)
        with patch.object(pilot, "main") as execute:
            self.assertEqual(launcher.main(["execute", "--run-dir", str(self.run_dir)]), 2)
        execute.assert_not_called()

    def test_submitted_packet_is_never_started_again(self):
        self.prepare()
        pilot.update_receipt(self.run_dir, phase="creation_unknown", submission_attempts=1)
        with self.assertRaises(pilot.PilotError):
            launcher.preflight(self.run_dir, "run", current_time=self.current)

    def test_over_budget_prepared_packet_is_rejected(self):
        self.prepare(limit=301)
        with self.assertRaises(pilot.PilotError):
            launcher.preflight(self.run_dir, "run", current_time=self.current)

    def test_modified_request_fails_before_execution(self):
        self.prepare()
        with open(self.run_dir / "request.json", "ab") as stream:
            stream.write(b" ")
        with self.assertRaises(pilot.PilotError):
            launcher.preflight(self.run_dir, "run", current_time=self.current)

    def test_old_snapshot_can_still_be_read_without_new_work(self):
        self.prepare(age_hours=100)
        pilot.update_receipt(self.run_dir, phase="in_progress", submission_attempts=1,
                             session_id="sess_test", turn_id="turn_test")
        result = launcher.preflight(self.run_dir, "resume", current_time=self.current)
        self.assertFalse(result["new_research"])

    def test_project_packet_without_confirmation_never_reaches_execution(self):
        self.prepare(project_id="proj_pilot_test")
        with patch.object(pilot, "main") as execute:
            self.assertEqual(launcher.main(["execute", "--run-dir", str(self.run_dir)]), 2)
        execute.assert_not_called()
        self.assertEqual(pilot.receipt(self.run_dir)["submission_attempts"], 0)

    def test_project_budget_is_labelled_monthly_and_confirmation_is_forwarded(self):
        self.prepare(project_id="proj_pilot_test")
        result = launcher.preflight(self.run_dir, "run", current_time=self.current,
                                    project_limit_confirmed=True)
        self.assertEqual(result["project_monthly_limit_cents"], 300)
        self.assertEqual(result["project_id"], "proj_pilot_test")
        self.assertFalse(result["project_limit_verified_by_api"])
        self.assertNotIn("spend_limit_cents", result)
        with patch.object(pilot, "main", return_value=0) as execute:
            code = launcher.main(["execute", "--run-dir", str(self.run_dir),
                                  "--project-limit-confirmed"])
        self.assertEqual(code, 0)
        self.assertIn("--project-limit-confirmed", execute.call_args.args[0])

    def test_project_confirmation_cannot_relabel_a_session_packet(self):
        self.prepare()
        with self.assertRaises(pilot.PilotError):
            launcher.preflight(self.run_dir, "run", project_limit_confirmed=True)
        with self.assertRaises(pilot.PilotError):
            launcher.preflight(self.run_dir, "inspect", project_limit_confirmed=True)

    def test_explicit_unlimited_packet_runs_without_project_confirmation(self):
        self.prepare(no_spend_limit=True)
        result = launcher.preflight(self.run_dir, "run", current_time=self.current)
        self.assertEqual(result["budget_mode"], "unlimited")
        self.assertIsNone(result["spend_limit_cents"])
        self.assertNotIn("project_id", result)
        with patch.object(pilot, "main", return_value=0) as execute:
            self.assertEqual(launcher.main(["execute", "--run-dir", str(self.run_dir)]), 0)
        self.assertNotIn("--project-limit-confirmed", execute.call_args.args[0])
        with self.assertRaises(pilot.PilotError):
            launcher.preflight(self.run_dir, "run", project_limit_confirmed=True)


if __name__ == "__main__":
    unittest.main()
