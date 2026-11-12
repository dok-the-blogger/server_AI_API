"""Offline guards around the Windows entry point; no credentials or API calls."""
from datetime import datetime, timedelta, timezone
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

    def prepare(self, age_hours=0.1, limit=300):
        self.snapshot["captured_at"] = (self.current - timedelta(hours=age_hours)).isoformat()
        path = self.base / "snapshot.json"
        path.write_bytes(pilot.encode(self.snapshot))
        return pilot.prepare(self.skill, path, self.run_dir, pilot.DEFAULT_MODEL, False, limit)

    def test_fresh_packet_displays_reviewed_threshold(self):
        self.prepare()
        result = launcher.preflight(self.run_dir, "run", current_time=self.current)
        self.assertEqual(result["spend_limit_cents"], 300)
        self.assertTrue(result["new_research"])
        self.assertEqual(result["snapshot_age_minutes"], 6.0)

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


if __name__ == "__main__":
    unittest.main()
