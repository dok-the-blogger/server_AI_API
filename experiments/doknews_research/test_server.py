"""Offline HTTP → durable queue → real pilot state machine → saved dossier checks."""
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
from pathlib import Path
import sys
import threading
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(BASE))

from fastapi import FastAPI
from fastapi.testclient import TestClient
from experiments.doknews_research.server_config import ResearchError, ServerConfig, json_bytes, load_config, utc_now
from experiments.doknews_research.server_control import Controller
from experiments.doknews_research.server_runner import execute_job, main as server_main, reconcile_job, refresh_usage_job, worker
from experiments.doknews_research.server_process import start_research
from experiments.doknews_research import pilot as server_pilot
from experiments.doknews_research.signal_ledger import LedgerError
from test_pilot import FakeSessions, FakeStream

spec = importlib.util.spec_from_file_location("offline_research_router", REPO / "routers" / "research.py")
route_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(route_module)


class ServerFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.skill = self.root / "skill"
        for name in (*server_pilot.POLICY_FILES, server_pilot.AI_GUIDE):
            path = self.skill / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Offline research policy fixture.\n", encoding="utf-8")
        self.cfg = ServerConfig(self.root / "state", self.skill, Path(sys.executable), api_enabled=True)
        self.controller = Controller(self.cfg)
        self.heartbeat()
        self.sessions = FakeSessions()
        self.client = SimpleNamespace(beta=SimpleNamespace(agents=SimpleNamespace(sessions=self.sessions)))
        self.factory_calls = 0
        self.snapshot = {
            "schema_version": 1,
            "task": {"name": "Offline research", "topic": "videogames", "time_window_hours": 24},
            "captured_at": utc_now(),
            "coverage": {"archive": "partial", "radar": "partial", "memory": "not_provided",
                         "missing": ["Only supplied fixture data is covered."]},
            "archive_articles": [], "memory_records": [],
            "radar_signals": [{"signal_id": "radar_test", "version": 1, "source_id": "fixture",
                "source_url": "https://example.com/primary", "title": "A new source event", "text": "Source facts"}],
        }
        self.body = {"request_id": str(uuid.uuid4()), "consumer": "doknews-api-hourly", "snapshot": self.snapshot}
        self.app = FastAPI()
        self.app.state.research_controller = self.controller
        self.app.state.research_service_token = "offline-service-token"
        self.app.include_router(route_module.router)
        self.http = TestClient(self.app)
        self.addCleanup(self.http.close)
        self.headers = {"Authorization": "Bearer offline-service-token"}

    def heartbeat(self):
        (self.cfg.state_dir / "worker-status.json").write_bytes(json_bytes({"phase": "idle", "observed_at": utc_now()}))

    def factory(self, _):
        self.factory_calls += 1
        return nullcontext(self.client)

    def submit(self, body=None):
        return self.http.post("/research/jobs", headers=self.headers, json=body or self.body)

    def test_http_job_finishes_once_and_unchanged_signals_skip_without_worker(self):
        accepted = self.submit()
        self.assertEqual(accepted.status_code, 202, accepted.text)
        job = accepted.json()
        self.assertFalse(job["new_research_submitted"])
        self.assertEqual(self.sessions.creates, 0)
        self.assertNotIn("snapshot", job)
        self.assertNotIn("run_dir", job)
        final = execute_job(self.cfg, self.controller.ledger, job["job_id"], client_factory=self.factory)
        self.assertEqual(final["status"], "completed")
        self.assertEqual(self.sessions.creates, 1)
        self.assertEqual(self.sessions.request["agent"]["model"], "gpt-6.1-sol")
        self.assertNotIn("spend_control", self.sessions.request)
        status = self.http.get(f"/research/jobs/{job['job_id']}", headers=self.headers).json()
        self.assertEqual(status["status"], "completed")
        self.assertTrue(status["new_research_submitted"])
        self.assertEqual(status["receipt"]["phase"], "completed")
        self.assertEqual(status["receipt"]["usage_status"], "pending")
        self.assertNotIn("latest_spend_unknown", status["receipt"])
        result = self.http.get(f"/research/jobs/{job['job_id']}/result", headers=self.headers).json()
        self.assertEqual(result["dossier_markdown"], "Проверяемое досье")
        self.assertEqual(result["sha256"], final["dossier_sha256"])
        (self.cfg.state_dir / "worker-status.json").unlink()
        self.assertEqual(self.submit().json()["job_id"], job["job_id"])
        fresh_body = {**self.body, "request_id": str(uuid.uuid4())}
        skipped = self.submit(fresh_body)
        self.assertEqual(skipped.status_code, 200, skipped.text)
        self.assertEqual(skipped.json()["status"], "no_signals")
        self.assertEqual(skipped.json()["coverage"]["radar"], "partial")
        self.assertEqual(self.sessions.creates, 1)

    def completed_job(self):
        job = self.submit().json()
        return execute_job(self.cfg, self.controller.ledger, job["job_id"], client_factory=self.factory)

    def test_delayed_usage_is_refreshed_once_without_changing_result_or_completion_time(self):
        job = self.completed_job()
        ledger = self.controller.ledger
        self.assertIsNone(ledger.next_usage_refresh("2000-01-01T00:00:00+00:00"))
        due = ledger.next_usage_refresh(utc_now())
        self.assertEqual(due["job_id"], job["job_id"])
        self.sessions.session_usage = {"input_tokens": 500, "output_tokens": 55, "total_tokens": 555}
        with patch.object(self.sessions, "retrieve", wraps=self.sessions.retrieve) as get:
            refresh_usage_job(self.cfg, ledger, due, client_factory=self.factory)
            refresh_usage_job(self.cfg, ledger, due, client_factory=self.factory)
            self.assertEqual(get.call_count, 1)
            self.assertEqual(get.call_args.kwargs, {"timeout": 10.0})
        saved = ledger.get_job(job["job_id"])
        self.assertEqual(saved["updated_at"], job["updated_at"])
        self.assertEqual(saved["dossier_sha256"], job["dossier_sha256"])
        self.assertEqual(self.controller.result(job["job_id"])["dossier_markdown"], "Проверяемое досье")
        receipt = self.controller.status(job["job_id"])["receipt"]
        self.assertEqual(receipt["usage_status"], "available")
        self.assertEqual(receipt["session_usage"], self.sessions.session_usage)
        self.assertEqual(receipt["session_usage_scope"], "session")
        self.assertTrue(receipt["session_usage_is_provisional"])
        self.assertTrue(receipt["usage_checked_at"])
        self.assertIsNone(ledger.next_usage_refresh(utc_now()))
        self.assertEqual((self.sessions.creates, self.sessions.cancels, self.sessions.deletes), (1, 0, 0))

    def test_missing_or_failed_late_usage_stops_waiting_and_preserves_dossier(self):
        job = self.completed_job()
        for failure in (False, True):
            with self.subTest(failure=failure):
                run_dir = Path(job["run_dir"])
                server_pilot.update_receipt(run_dir, usage_refresh_attempted_at=None)
                with patch.object(self.sessions, "retrieve", wraps=self.sessions.retrieve,
                                  side_effect=ConnectionError("private provider body") if failure else None) as get:
                    refresh_usage_job(self.cfg, self.controller.ledger, job, client_factory=self.factory)
                    refresh_usage_job(self.cfg, self.controller.ledger, job, client_factory=self.factory)
                    self.assertEqual(get.call_count, 1)
                status = self.controller.status(job["job_id"])
                self.assertEqual(status["status"], "completed")
                self.assertEqual(status["receipt"]["usage_status"], "unavailable")
                self.assertIsNone(status["receipt"]["session_usage"])
                self.assertNotIn("private provider body", json.dumps(status))
                self.assertIsNone(self.controller.ledger.next_usage_refresh(utc_now()))
                self.assertEqual(self.controller.result(job["job_id"])["sha256"], job["dossier_sha256"])
        self.assertEqual(self.sessions.creates, 1)

    def test_worker_refreshes_due_usage_and_keeps_last_known_counts(self):
        self.sessions.session_usage = {"input_tokens": 500, "output_tokens": 55}
        job = self.completed_job()
        with self.controller.ledger._connect(write=True) as db:
            db.execute("UPDATE jobs SET updated_at = ? WHERE job_id = ?",
                       ((datetime.now(timezone.utc) - timedelta(minutes=6)).isoformat(), job["job_id"]))
        self.sessions.session_usage = None
        with patch("experiments.doknews_research.server_runner.validate_runtime"), \
                patch.object(server_pilot, "client_for_run", self.factory):
            self.assertEqual(worker(self.cfg, once=True), 0)
        receipt = self.controller.status(job["job_id"])["receipt"]
        self.assertEqual(receipt["usage_status"], "available")
        self.assertEqual(receipt["session_usage"]["input_tokens"], 500)
        self.assertTrue(receipt["latest_session_usage_unknown"])
        self.assertIsNone(self.controller.ledger.next_usage_refresh(utc_now()))
        self.assertEqual(self.sessions.creates, 1)

    def test_unknown_create_remains_blocked_and_never_reposts(self):
        self.sessions.fail_create = True
        job = self.submit().json()
        stopped = execute_job(self.cfg, self.controller.ledger, job["job_id"], client_factory=self.factory)
        self.assertEqual(stopped["status"], "unknown")
        recovered = reconcile_job(self.cfg, self.controller.ledger, stopped, client_factory=self.factory)
        self.assertEqual(recovered["status"], "unknown")
        self.assertEqual(self.factory_calls, 1)
        with self.assertRaises(LedgerError):
            execute_job(self.cfg, self.controller.ledger, job["job_id"], client_factory=self.factory)
        conflict = self.submit({**self.body, "request_id": str(uuid.uuid4())})
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(self.sessions.creates, 1)
        serialized = self.http.get(f"/research/jobs/{job['job_id']}", headers=self.headers).text
        self.assertNotIn("sk-proj-", serialized)
        self.assertNotIn("observer_error", serialized)

    def test_interrupted_observer_recovers_saved_result_without_second_create(self):
        self.sessions.disconnect = True
        job = self.submit().json()
        interrupted = execute_job(self.cfg, self.controller.ledger, job["job_id"], client_factory=self.factory)
        self.assertNotEqual(interrupted["status"], "completed")
        self.assertEqual(self.sessions.creates, 1)
        self.sessions.disconnect = False
        final = reconcile_job(self.cfg, self.controller.ledger, interrupted, client_factory=self.factory)
        self.assertEqual(final["status"], "completed")
        self.assertEqual(self.sessions.creates, 1)

    def test_failed_cancel_is_retried_at_deadline_with_spacing_and_no_new_create(self):
        self.sessions.disconnect = True
        self.sessions.status = "in_progress"
        job = self.submit().json()
        interrupted = execute_job(self.cfg, self.controller.ledger, job["job_id"], client_factory=self.factory)
        root = Path(interrupted["run_dir"])
        server_pilot.update_receipt(root, deadline_epoch=1, cancel_requested=True,
                                   cancel_error={"type": "APIConnectionError"})
        cancels = self.sessions.cancels
        reconcile_job(self.cfg, self.controller.ledger, interrupted, client_factory=self.factory)
        self.assertEqual(self.sessions.cancels, cancels + 1)
        reconcile_job(self.cfg, self.controller.ledger, self.controller.ledger.get_job(job["job_id"]), client_factory=self.factory)
        self.assertEqual(self.sessions.cancels, cancels + 1)
        server_pilot.update_receipt(root, server_cancel_retry_at=1)
        reconcile_job(self.cfg, self.controller.ledger, self.controller.ledger.get_job(job["job_id"]), client_factory=self.factory)
        self.assertEqual(self.sessions.cancels, cancels + 2)
        self.assertEqual(self.sessions.creates, 1)

    def test_worker_is_not_ready_until_runtime_and_credentials_checked(self):
        observed = []
        def missing(_):
            observed.append(self.controller.worker_status()["ready"])
            raise ResearchError("credential_unavailable", "Credential is not configured.")
        with patch("experiments.doknews_research.server_runner.load_credential", side_effect=missing):
            self.assertEqual(worker(self.cfg, once=True), 2)
        self.assertEqual(observed, [False])
        self.assertFalse(self.controller.worker_status()["ready"])
        self.assertEqual(self.submit().status_code, 503)

    def test_worker_stops_after_pilot_converts_shutdown_to_pilot_error(self):
        job = self.submit().json()
        pending = self.controller.ledger.enqueue("another-consumer", str(uuid.uuid4()), self.snapshot, self.cfg.model)
        stop_requested = threading.Event()
        def interrupt_stream(stream):
            yield stream.events[0]
            stop_requested.set()
            raise KeyboardInterrupt()
        with patch("experiments.doknews_research.server_runner.load_credential"):
            with patch.object(server_pilot, "client_for_run", side_effect=self.factory):
                with patch.object(FakeStream, "__iter__", interrupt_stream):
                    self.assertEqual(worker(self.cfg, once=True, stop_event=stop_requested), 130)
        stopped = self.controller.ledger.get_job(job["job_id"])
        self.assertNotEqual(stopped["status"], "completed")
        self.assertTrue(server_pilot.receipt(Path(stopped["run_dir"]))["observer_interrupted"])
        self.assertEqual(self.sessions.creates, 1)
        self.assertEqual(self.sessions.cancels, 1)
        self.assertEqual(self.controller.ledger.get_job(pending["job_id"])["status"], "queued")
        self.assertEqual(self.controller.worker_status()["phase"], "stopped")

    def test_runtime_check_never_opens_queue_or_submits_an_existing_job(self):
        job = self.submit().json()
        before = self.controller.ledger.get_job(job["job_id"])
        output, errors = io.StringIO(), io.StringIO()
        with patch("experiments.doknews_research.server_runner.load_config", return_value=self.cfg):
            with patch("experiments.doknews_research.server_runner.load_credential"):
                with patch("experiments.doknews_research.server_runner.Ledger", side_effect=AssertionError("check must not open the ledger")):
                    with patch.object(server_pilot, "client_for_run", side_effect=AssertionError("check must not open an API client")):
                        with redirect_stdout(output):
                            self.assertEqual(server_main(["check", "--config", "/private/config.json"]), 0)
                        (self.skill / server_pilot.AI_GUIDE).unlink()
                        with redirect_stderr(errors):
                            self.assertEqual(server_main(["check", "--config", "/private/config.json"]), 2)
        self.assertEqual(json.loads(output.getvalue()), {
            "status": "ready", "api_enabled": True, "model": self.cfg.model,
            "sdk_version": "3.49.0", "policy_files_count": 5, "api_request_made": False,
        })
        self.assertEqual(json.loads(errors.getvalue())["error"]["code"], "policy_unavailable")
        self.assertNotIn(str(self.skill), errors.getvalue())
        self.assertEqual(self.controller.ledger.get_job(job["job_id"]), before)
        self.assertEqual(self.sessions.creates, 0)

    def test_bad_ledger_does_not_escape_research_start_boundary(self):
        with patch("experiments.doknews_research.server_process.load_config", return_value=self.cfg):
            with patch("experiments.doknews_research.server_process.Controller", side_effect=LedgerError("Invalid schema")):
                with self.assertRaises(ResearchError) as raised:
                    start_research("/private/config.json", service_token_configured=True)
        self.assertEqual(raised.exception.code, "state_unavailable")

    def test_auth_disabled_and_payload_boundaries_create_no_job(self):
        self.assertEqual(self.http.post("/research/jobs", json=self.body).status_code, 401)
        self.app.state.research_service_token = ""
        self.assertEqual(self.submit().status_code, 503)
        self.app.state.research_service_token = "offline-service-token"
        self.controller.config = replace(self.cfg, api_enabled=False)
        self.assertEqual(self.submit().status_code, 503)
        self.controller.config = self.cfg
        for bad in ({**self.body, "model": "expensive"}, {**self.body, "request_id": "invalid"}):
            self.assertEqual(self.submit(bad).status_code, 422)
        self.assertEqual(self.http.post("/research/jobs", headers=self.headers, content=b"{" + b" " * 1_010_000).status_code, 413)
        self.assertEqual(self.http.post("/research/jobs", headers=self.headers, content='{"snapshot":NaN}').status_code, 422)
        self.assertIsNone(self.controller.ledger.next_queued())
        self.assertEqual(self.factory_calls, 0)

    def test_stale_new_job_rejected_but_exact_replay_remains_available(self):
        accepted = self.submit().json()
        self.controller.config = replace(self.cfg, max_snapshot_hours=1e-12)
        self.assertEqual(self.submit().json()["job_id"], accepted["job_id"])
        rejected = self.submit({**self.body, "request_id": str(uuid.uuid4())})
        self.assertEqual(rejected.status_code, 422)
        self.assertEqual(rejected.json()["detail"]["code"], "stale_snapshot")

    def test_changed_payload_conflict_and_unfinished_result(self):
        accepted = self.submit().json()
        changed = {**self.body, "snapshot": {**self.snapshot, "captured_at": "2020-01-01T00:00:00Z"}}
        conflict = self.submit(changed)
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(conflict.json()["detail"]["code"], "request_conflict")
        result = self.http.get(f"/research/jobs/{accepted['job_id']}/result", headers=self.headers)
        self.assertEqual(result.status_code, 409)

    def test_stale_queued_job_fails_before_client_creation(self):
        stale = {**self.snapshot, "captured_at": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()}
        job = self.controller.ledger.enqueue("doknews-api-hourly", str(uuid.uuid4()), stale, self.cfg.model)
        stopped = execute_job(self.cfg, self.controller.ledger, job["job_id"], client_factory=self.factory)
        self.assertEqual(stopped["status"], "failed")
        self.assertEqual(stopped["failure_phase"], "preflight_failed")
        self.assertEqual(self.factory_calls, 0)

    def test_private_result_tampering_is_detected(self):
        job = self.submit().json()
        final = execute_job(self.cfg, self.controller.ledger, job["job_id"], client_factory=self.factory)
        (Path(final["run_dir"]) / "dossier.md").write_text("Changed", encoding="utf-8")
        self.assertEqual(self.http.get(f"/research/jobs/{job['job_id']}/result", headers=self.headers).status_code, 503)

    def test_config_rejects_credentials_checkout_state_and_bad_consumers(self):
        path = self.root / "config.json"
        good = {"state_dir": str(self.cfg.state_dir), "skill_dir": str(self.skill), "worker_python": sys.executable}
        path.write_text(json.dumps(good))
        self.assertFalse(load_config(path).api_enabled)
        for changes in ({"state_dir": str(REPO)}, {"consumers": "hourly"},
                        {"consumers": [{}]}, {"model": "gpt-6-astra"},
                        {"openai_key_file": "sk-proj-" + "X" * 40}):
            path.write_text(json.dumps({**good, **changes}))
            with self.assertRaises(ResearchError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
