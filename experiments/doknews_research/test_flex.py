"""Flex acceptance, bounded failure handling and legacy ledger migration; no network."""
import importlib.util
from copy import deepcopy
import json
import sqlite3
import unittest
from unittest.mock import patch
import uuid

import test_server as fixtures
from test_pilot import turn, SID, TID
from experiments.doknews_research import pilot, launcher
from experiments.doknews_research.server_runner import execute_job, reconcile_job
from experiments.doknews_research.signal_ledger import Ledger, LedgerError


class FlexFlowTests(unittest.TestCase):
    setUp = fixtures.ServerFlowTests.setUp
    heartbeat = fixtures.ServerFlowTests.heartbeat
    submit = fixtures.ServerFlowTests.submit
    factory = fixtures.ServerFlowTests.factory

    def test_flex_profile_passes_real_preflight_and_produces_dossier(self):
        for name in (*pilot.POLICY_FILES, pilot.AI_GUIDE):
            (self.skill / name).unlink()
        job = self.submit().json()
        result = execute_job(self.cfg, self.controller.ledger, job["job_id"], client_factory=self.factory)
        self.assertEqual(result["status"], "completed")
        request = self.sessions.request
        self.assertEqual(request["agent"]["service_tier"], "flex")
        text = request["agent"]["instructions"]
        self.assertEqual(text, pilot.read_text(pilot.RESEARCH_CONTRACT))
        self.assertNotIn("dossier_contract", request["input"])
        self.assertNotIn('"task_instructions":', request["input"])
        self.assertIn("ai_inference_ceiling", text)
        run_dir = self.cfg.state_dir / "runs" / job["job_id"] / "pilot"
        manifest = pilot.read_json(run_dir / "manifest.json")
        self.assertEqual(manifest["instruction_profile"], "signal_facts_v2")
        self.assertEqual(manifest["policy_files"], [])
        self.assertFalse(manifest["ai_guide_included"])
        state = pilot.receipt(run_dir)
        self.assertEqual(state["service_tier"], "flex")
        self.assertEqual(self.sessions.creates, 1)
        self.assertEqual(request["environment"], {"type": "none"})
        self.assertEqual(request["agent"]["tools"], [{"type": "web_search", "mode": "live", "context_size": "medium"}])

    def test_container_trial_stops_on_three_dossiers_or_six_attempts_after_restart(self):
        for index in range(4):
            signal = {**self.snapshot["radar_signals"][0], "signal_id": f"signal{index}"}
            packet = {**self.body, "request_id": str(uuid.uuid4()),
                      "snapshot": {**self.snapshot, "radar_signals": [signal]}}
            job_id = self.submit(packet).json()["job_id"]
            ledger = Ledger(self.controller.ledger.db_path)
            completed = execute_job(self.cfg, ledger, job_id, client_factory=self.factory)
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(self.sessions.request["environment"]["type"], "none" if index < 3 else "openai_hosted")
            self.assertEqual(self.submit(packet).json()["job_id"], job_id)
        self.assertEqual(self.sessions.creates, 4)

    def test_unsuccessful_trial_stops_after_six_ordinary_claims(self):
        for index in range(7):
            job_id = self.submit({**self.body, "request_id": str(uuid.uuid4())}).json()["job_id"]
            ledger = Ledger(self.controller.ledger.db_path)
            claimed = ledger.claim(job_id, trial_on_flex=True)
            self.assertEqual(claimed["trial_without_container"], index < 6)
            ledger.fail(job_id, "preflight_failed", "offline pre-submission failure")

    @unittest.skipUnless(importlib.util.find_spec("openai"), "isolated SDK not installed")
    def test_confirmed_capacity_rejection_releases_next_run_but_never_replays(self):
        import httpx2
        from openai import RateLimitError
        error = RateLimitError("resource unavailable", body={"error": {"code": "resource_unavailable"}},
            response=httpx2.Response(429, request=httpx2.Request("POST", "https://api.openai.com/v1/agents/sessions")))
        job = self.submit().json()
        with patch.object(self.sessions, "create", side_effect=error) as create:
            result = execute_job(self.cfg, self.controller.ledger, job["job_id"], client_factory=self.factory)
        create.assert_called_once()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_phase"], "capacity_unavailable")
        # DOK Agents keeps a strict fixed-message contract; detail lives in codes.
        public = self.http.get(f"/research/jobs/{job['job_id']}", headers=self.headers).json()
        self.assertEqual(public["failure_message"],
            "Research did not produce a verified result; inspect the saved run receipt.")
        self.assertEqual(public["receipt"]["phase"], "capacity_unavailable")
        self.assertEqual(public["receipt"]["failure_code"], "resource_unavailable")
        self.assertIsNone(public["receipt"]["session_id"])
        self.assertEqual(self.controller.ledger.list_unresolved(), [])
        self.assertEqual(self.submit().json()["job_id"], job["job_id"])
        self.assertEqual(self.http.get(f"/research/jobs/{job['job_id']}/result", headers=self.headers).status_code, 409)
        # Restart/reconcile does not resubmit the rejected job or consume its signals.
        with patch.object(self.sessions, "create", side_effect=AssertionError("no replay")):
            reconcile_job(self.cfg, self.controller.ledger, result, client_factory=self.factory)
        later = self.submit({**self.body, "request_id": str(uuid.uuid4())})
        self.assertEqual(later.status_code, 202, later.text)
        self.assertEqual(later.json()["selected_count"], 1)
        finished = execute_job(self.cfg, self.controller.ledger, later.json()["job_id"], client_factory=self.factory)
        self.assertEqual(finished["status"], "completed")

    @unittest.skipUnless(importlib.util.find_spec("openai"), "isolated SDK not installed")
    def test_generic_429_timeout_and_post_accept_failure_keep_consumer_blocked(self):
        import httpx2
        from openai import RateLimitError
        for mode in ("rate_limit", "timeout", "after_accept"):
            with self.subTest(mode=mode):
                self.setUp()
                job = self.submit().json()
                if mode == "rate_limit":
                    error = RateLimitError("quota", body={"error": {"code": "rate_limit_exceeded"}},
                        response=httpx2.Response(429, request=httpx2.Request("POST", "https://api.openai.com/v1/agents/sessions")))
                else:
                    error = TimeoutError("unknown outcome")
                if mode == "after_accept":
                    self.sessions.disconnect = True
                    result = execute_job(self.cfg, self.controller.ledger, job["job_id"], client_factory=self.factory)
                else:
                    with patch.object(self.sessions, "create", side_effect=error):
                        result = execute_job(self.cfg, self.controller.ledger, job["job_id"], client_factory=self.factory)
                self.assertNotEqual(result.get("failure_phase"), "capacity_unavailable")
                self.assertEqual(len(self.controller.ledger.list_unresolved()), 1)
                blocked = self.submit({**self.body, "request_id": str(uuid.uuid4())})
                self.assertEqual(blocked.status_code, 409, blocked.text)

    def test_capacity_label_without_proof_cannot_release_a_job(self):
        job = self.submit().json()
        self.controller.ledger.claim(job["job_id"])
        with self.assertRaises(LedgerError):
            self.controller.ledger.fail(job["job_id"], "capacity_unavailable", "unchecked")
        with self.assertRaises(LedgerError):
            self.controller.ledger.defer_capacity(job["job_id"], {"phase": "capacity_unavailable"})
        self.assertEqual(len(self.controller.ledger.list_unresolved()), 1)

    def test_terminal_flex_refusal_releases_queue_and_preserves_exact_replay(self):
        self.sessions.status = "failed"
        job = self.submit().json()
        refused = {**turn("failed"), "error": {"code": "flex_unavailable"}}
        with patch.object(self.sessions.turns, "retrieve", return_value=refused):
            result = execute_job(self.cfg, self.controller.ledger, job["job_id"], client_factory=self.factory)
        self.assertEqual(result["failure_phase"], "capacity_unavailable")
        public = self.http.get(f"/research/jobs/{job['job_id']}", headers=self.headers).json()
        self.assertEqual(public["status"], "failed")
        self.assertEqual(public["receipt"]["phase"], "capacity_unavailable")
        self.assertEqual(public["receipt"]["failure_code"], "flex_unavailable")
        self.assertEqual((public["receipt"]["session_id"], public["receipt"]["turn_id"]), (SID, TID))
        self.assertNotIn("terminal_capacity_proof", public["receipt"])
        self.assertEqual(public["failure_message"],
            "Research did not produce a verified result; inspect the saved run receipt.")
        self.assertTrue(public["new_research_submitted"])
        self.assertEqual(self.controller.ledger.list_unresolved(), [])
        self.assertEqual(self.submit().json()["job_id"], job["job_id"])
        self.assertEqual(self.http.get(f"/research/jobs/{job['job_id']}/result", headers=self.headers).status_code, 409)
        # A later legitimate run can use the still-unresearched signal.
        later = self.submit({**self.body, "request_id": str(uuid.uuid4())})
        self.assertEqual(later.status_code, 202, later.text)
        self.assertEqual(later.json()["selected_count"], 1)
        self.sessions.status = "completed"
        finished = execute_job(self.cfg, self.controller.ledger, later.json()["job_id"], client_factory=self.factory)
        self.assertEqual(finished["status"], "completed")
        self.assertEqual((self.sessions.creates, self.sessions.cancels, self.sessions.deletes), (2, 0, 0))

    def test_preexisting_failed_session_is_reconciled_once_without_new_input(self):
        self.sessions.status = "failed"
        job = self.submit().json()
        failed = execute_job(self.cfg, self.controller.ledger, job["job_id"], client_factory=self.factory)
        self.assertEqual(failed["failure_phase"], "failed")
        refused = {**turn("failed"), "error": {"code": "flex_unavailable"}}
        with patch.object(self.sessions.turns, "retrieve", return_value=refused) as get:
            settled = reconcile_job(self.cfg, self.controller.ledger, failed, client_factory=self.factory)
            again = reconcile_job(self.cfg, self.controller.ledger, settled, client_factory=self.factory)
        get.assert_called_once()
        self.assertEqual(settled, again)
        self.assertEqual(settled["failure_phase"], "capacity_unavailable")
        self.assertEqual(self.controller.ledger.list_unresolved(), [])
        self.assertEqual((self.sessions.creates, self.sessions.cancels, self.sessions.deletes), (1, 0, 0))
        state = pilot.receipt(self.cfg.state_dir / "runs" / job["job_id"] / "pilot")
        for field, value in (("turn_id", "wrong"), ("session_id", "wrong"),
                             ("status", "in_progress"), ("failure_code", "other_error"),
                             ("source", "stream_event"), ("verified_at", None)):
            with self.subTest(field=field):
                tampered = deepcopy(state)
                tampered["terminal_capacity_proof"][field] = value
                with self.assertRaises(LedgerError):
                    self.controller.ledger.defer_capacity(job["job_id"], tampered)
        state.pop("terminal_capacity_proof")
        with self.assertRaises(LedgerError):
            self.controller.ledger.defer_capacity(job["job_id"], state)

    def test_unverified_terminal_failure_keeps_queue_blocked(self):
        for mode in ("running", "other_code", "wrong_turn", "wrong_session", "subagent",
                     "wrong_session_response", "required_action", "get_timeout", "default_tier"):
            with self.subTest(mode=mode):
                self.setUp()
                self.sessions.status = "failed"
                job = self.submit().json()
                failed = execute_job(self.cfg, self.controller.ledger, job["job_id"], client_factory=self.factory)
                run_dir = self.cfg.state_dir / "runs" / job["job_id"] / "pilot"
                # A cached error code, including one from an event, is not proof.
                pilot.update_receipt(run_dir, failure_code="flex_unavailable",
                                     **({"service_tier": "default"} if mode == "default_tier" else {}))
                refused = {**turn("failed"), "error": {"code": "flex_unavailable"}}
                mutations = {"running": {"status": "in_progress"}, "other_code": {"error": {"code": "other_error"}},
                             "wrong_turn": {"id": "wrong"}, "wrong_session": {"session_id": "wrong"},
                             "subagent": {"subagent_id": "other"}}
                refused.update(mutations.get(mode, {}))
                session = self.sessions.retrieve(SID)
                if mode == "wrong_session_response":
                    session["id"] = "wrong"
                if mode == "required_action":
                    session["required_actions"] = [{"type": "unknown"}]
                with patch.object(self.sessions, "retrieve", return_value=session), patch.object(
                        self.sessions.turns, "retrieve", return_value=refused,
                        side_effect=TimeoutError("unknown") if mode == "get_timeout" else None):
                    result = reconcile_job(self.cfg, self.controller.ledger, failed, client_factory=self.factory)
                self.assertNotEqual(result.get("failure_phase"), "capacity_unavailable")
                self.assertEqual(len(self.controller.ledger.list_unresolved()), 1)
                self.assertEqual(self.submit({**self.body, "request_id": str(uuid.uuid4())}).status_code, 409)
                self.assertEqual(self.sessions.creates, 1)

    def test_version_one_index_migrates_without_changing_existing_jobs(self):
        job = self.submit().json()
        with sqlite3.connect(self.cfg.db_path) as db:
            db.execute("DROP INDEX one_unresolved_consumer")
            db.execute("CREATE UNIQUE INDEX one_unresolved_consumer ON jobs(consumer) WHERE status IN ('queued','running','unknown','failed')")
            db.execute("PRAGMA user_version=1")
        reopened = Ledger(self.cfg.db_path)
        self.assertEqual(reopened.get_job(job["job_id"])["status"], "queued")
        self.assertEqual(reopened.plan("doknews-api-hourly", self.snapshot)["blocked_job_id"], job["job_id"])
        with sqlite3.connect(self.cfg.db_path) as db:
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 4)

    def test_unrecognized_or_mismatched_tier_fails_before_submission(self):
        source = self.root / "snapshot.json"
        source.write_text(json.dumps(self.snapshot))
        root = self.root / "tier-test"
        pilot.prepare(self.skill, source, root, no_spend_limit=True, service_tier="flex", research_only=True)
        pilot.update_receipt(root, service_tier="default")
        with self.assertRaises(pilot.PilotError):
            launcher.preflight(root, "run")
        with self.assertRaises(pilot.PilotError):
            pilot.prepare(self.skill, source, self.root / "invalid", service_tier="auto")
        self.assertEqual(self.sessions.creates, 0)

    def test_flex_client_waits_without_sdk_retries_or_default_fallback(self):
        source = self.root / "snapshot.json"
        source.write_text(json.dumps(self.snapshot))
        root = self.root / "client-test"
        pilot.prepare(self.skill, source, root, no_spend_limit=True, service_tier="flex")
        with patch.object(pilot, "make_client", return_value=self.client) as make:
            self.assertIs(pilot.client_for_run(root), self.client)
        make.assert_called_once_with(timeout=600)
