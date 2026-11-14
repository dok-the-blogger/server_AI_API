"""Flex acceptance, bounded failure handling and legacy ledger migration; no network."""
import importlib.util
import json
import sqlite3
import unittest
from unittest.mock import patch
import uuid

import test_server as fixtures
from experiments.doknews_research import pilot, launcher
from experiments.doknews_research.server_runner import execute_job, reconcile_job
from experiments.doknews_research.signal_ledger import Ledger, LedgerError


class FlexFlowTests(unittest.TestCase):
    setUp = fixtures.ServerFlowTests.setUp
    heartbeat = fixtures.ServerFlowTests.heartbeat
    submit = fixtures.ServerFlowTests.submit
    factory = fixtures.ServerFlowTests.factory

    def test_flex_profile_passes_real_preflight_and_produces_dossier(self):
        for name in pilot.POLICY_FILES:
            (self.skill / name).write_text("Unique policy " + name)
        job = self.submit().json()
        result = execute_job(self.cfg, self.controller.ledger, job["job_id"], client_factory=self.factory)
        self.assertEqual(result["status"], "completed")
        request = self.sessions.request
        self.assertEqual(request["agent"]["service_tier"], "flex")
        text = request["agent"]["instructions"]
        for name in pilot.RESEARCH_POLICY_FILES:
            self.assertIn("Unique policy " + name, text)
        self.assertNotIn("Unique policy SKILL.md", text)
        self.assertNotIn("Unique policy references/DOKNEWS_EDITORIAL_MEMORY.md", text)
        self.assertIn("ai_inference_ceiling", text)
        state = pilot.receipt(self.cfg.state_dir / "runs" / job["job_id"] / "pilot")
        self.assertEqual(state["service_tier"], "flex")
        self.assertEqual(self.sessions.creates, 1)

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
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 2)

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
