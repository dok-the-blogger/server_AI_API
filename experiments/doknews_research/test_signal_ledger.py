"""Offline durability and paid-work deduplication invariants."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest

from signal_ledger import MAX_DOSSIER_BYTES, Ledger, LedgerBusy, LedgerError


def signal(number=1, version=1):
    return {"signal_id": f"signal-{number}", "version": version,
            "source_id": "source", "source_url": f"https://example.invalid/news/{number}",
            "title": f"News {number}", "text": f"Meaningful text {number}",
            "content_sha256": hashlib.sha256(f"Meaningful text {number}".encode()).hexdigest(),
            "published_at": "2026-11-12T10:00:00Z"}


def snapshot(count=1):
    return {"schema_version": 1,
            "task": {"name": "Research", "topic": "general", "time_window_hours": 24},
            "captured_at": "2026-11-12T11:00:00Z",
            "coverage": {"archive": "partial", "radar": "complete", "memory": "partial",
                         "missing": ["Only supplied sources are available."]},
            "radar_signals": [signal(i + 1) for i in range(count)],
            "archive_articles": [{"title": "Archive context"}], "memory_records": []}


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.db_path = self.root / "state" / "research.sqlite3"
        self.ledger = Ledger(self.db_path)
        self.consumer = "doknews-research-v2"

    def tearDown(self):
        self.temporary.cleanup()

    def enqueue(self, value=None, request_id="request-1", limit=3):
        return self.ledger.enqueue(self.consumer, request_id, value or snapshot(), "gpt-6.1-sol", limit)

    def dossier(self, job, phase="completed", text="Verified dossier\n"):
        run_dir = self.root / "runs" / job["job_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        self.ledger.bind_run(job["job_id"], run_dir, run_id=job["job_id"])
        path = run_dir / "dossier.md"
        path.write_text(text, encoding="utf-8")
        receipt = {"schema_version": 1, "run_id": job["job_id"], "phase": phase,
                   "result_status": "available", "dossier_file": "dossier.md",
                   "dossier_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        return receipt, path

    def complete(self, job, phase="completed"):
        self.ledger.claim(job["job_id"])
        receipt, path = self.dossier(job, phase)
        return self.ledger.finish(job["job_id"], receipt, path)

    def test_plan_does_not_reserve_or_advance(self):
        value = snapshot(5)
        before = deepcopy(value)
        plan = self.ledger.plan(self.consumer, value, 3)
        self.assertEqual(plan["selected_count"], 3)
        self.assertEqual(plan["pending_count"], 5)
        self.assertEqual(plan["pending_remaining"], 2)
        self.assertIsNone(plan["blocked_job_id"])
        self.assertIsNone(self.ledger.next_queued())
        self.assertEqual(value, before)
        self.assertEqual(plan["coverage"]["radar"], "partial")
        self.assertEqual(plan["input_coverage"]["radar"], "complete")
        self.assertEqual(plan["snapshot"]["coverage"]["radar"], "partial")
        with sqlite3.connect(self.db_path) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM requests").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT count(*) FROM researched_signals").fetchone()[0], 0)

    def test_same_request_replays_existing_job_and_rejects_changed_payload(self):
        job = self.enqueue()
        self.assertEqual(self.enqueue()["job_id"], job["job_id"])
        changed = snapshot()
        changed["captured_at"] = "2026-11-12T12:00:00Z"
        with self.assertRaisesRegex(LedgerError, "different payload"):
            self.enqueue(changed)
        with self.assertRaisesRegex(LedgerError, "different payload"):
            self.ledger.enqueue(self.consumer, "request-1", snapshot(), "gpt-6-astra")
        with self.assertRaisesRegex(LedgerError, "different payload"):
            self.enqueue(limit=2)

    def test_read_only_replay_returns_old_snapshot_job_without_reserving(self):
        value = snapshot()
        value["captured_at"] = "2020-01-01T00:00:00Z"
        self.assertIsNone(self.ledger.replay(self.consumer, "request-1", value, "gpt-6.1-sol"))
        self.assertIsNone(self.ledger.next_queued())
        job = self.enqueue(value)
        self.assertEqual(self.ledger.replay(self.consumer, "request-1", value, "gpt-6.1-sol"), job)
        self.assertIsNone(self.ledger.replay(self.consumer, "unseen-request", value, "gpt-6.1-sol"))
        with sqlite3.connect(self.db_path) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM jobs").fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT count(*) FROM requests").fetchone()[0], 1)

    def test_read_only_replay_has_same_payload_conflict_as_enqueue(self):
        self.enqueue()
        changed = snapshot()
        changed["captured_at"] = "2026-11-12T12:00:00Z"
        for value, model, limit in ((changed, "gpt-6.1-sol", 3),
                                    (snapshot(), "gpt-6-astra", 3),
                                    (snapshot(), "gpt-6.1-sol", 2)):
            with self.subTest(model=model, limit=limit):
                with self.assertRaisesRegex(LedgerError, "different payload"):
                    self.ledger.replay(self.consumer, "request-1", value, model, limit)

    def test_read_only_replay_returns_saved_no_signals(self):
        value = snapshot(0)
        result = self.enqueue(value)
        self.assertEqual(self.ledger.replay(self.consumer, "request-1", value, "gpt-6.1-sol"), result)
        self.assertIsNone(self.ledger.next_queued())

    def test_one_reservation_under_concurrent_distinct_requests(self):
        barrier = threading.Barrier(8)

        def reserve(index):
            barrier.wait()
            try:
                return self.ledger.enqueue(self.consumer, f"request-{index}", snapshot(), "gpt-6.1-sol")
            except LedgerBusy as exc:
                return exc

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(reserve, range(8)))
        successful = [x for x in results if isinstance(x, dict)]
        self.assertEqual(len(successful), 1)
        self.assertEqual(sum(isinstance(x, LedgerBusy) for x in results), 7)
        self.assertTrue(all(x.job_id == successful[0]["job_id"] for x in results if isinstance(x, LedgerBusy)))

    def test_same_request_concurrently_returns_one_job(self):
        barrier = threading.Barrier(6)

        def reserve(_):
            barrier.wait()
            return self.enqueue()["job_id"]

        with ThreadPoolExecutor(max_workers=6) as pool:
            job_ids = list(pool.map(reserve, range(6)))
        self.assertEqual(len(set(job_ids)), 1)

    def test_only_one_worker_can_claim(self):
        job = self.enqueue()
        barrier = threading.Barrier(8)

        def claim(_):
            barrier.wait()
            try:
                return self.ledger.claim(job["job_id"])["status"]
            except LedgerError:
                return "rejected"

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(claim, range(8)))
        self.assertEqual(results.count("running"), 1)
        self.assertEqual(results.count("rejected"), 7)
        self.assertIsNone(self.ledger.next_queued())

    def test_bounded_batch_preserves_remainder(self):
        value = snapshot(7)
        first = self.enqueue(value, limit=3)
        self.assertEqual(first["coverage"], first["snapshot"]["coverage"])
        self.assertEqual(first["coverage"]["radar"], "partial")
        self.assertEqual(first["input_coverage"]["radar"], "complete")
        self.assertEqual([x["signal_id"] for x in first["snapshot"]["radar_signals"]],
                         ["signal-1", "signal-2", "signal-3"])
        self.complete(first)
        second = self.enqueue(value, request_id="request-2", limit=3)
        self.assertEqual(second["researched_count"], 3)
        self.assertEqual(second["pending_count"], 4)
        self.assertEqual(second["pending_remaining"], 1)
        self.assertEqual([x["signal_id"] for x in second["snapshot"]["radar_signals"]],
                         ["signal-4", "signal-5", "signal-6"])
        self.complete(second)
        third = self.enqueue(value, request_id="request-3", limit=3)
        self.assertEqual([x["signal_id"] for x in third["snapshot"]["radar_signals"]], ["signal-7"])

    def test_no_advancement_until_valid_completion(self):
        value = snapshot()
        job = self.enqueue(value)
        self.assertEqual(self.ledger.plan(self.consumer, value)["pending_count"], 1)
        self.ledger.claim(job["job_id"])
        receipt, path = self.dossier(job)
        receipt["phase"] = "in_progress"
        with self.assertRaises(LedgerError):
            self.ledger.finish(job["job_id"], receipt, path)
        self.assertEqual(self.ledger.plan(self.consumer, value)["pending_count"], 1)
        receipt["phase"] = "completed"
        receipt["dossier_sha256"] = "0" * 64
        with self.assertRaisesRegex(LedgerError, "does not match"):
            self.ledger.finish(job["job_id"], receipt, path)
        self.assertEqual(self.ledger.get_job(job["job_id"])["status"], "running")
        self.assertEqual(self.ledger.plan(self.consumer, value)["researched_count"], 0)
        receipt["dossier_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(self.ledger.finish(job["job_id"], receipt, path)["status"], "completed")
        plan = self.ledger.plan(self.consumer, value)
        self.assertEqual(plan["researched_count"], 1)
        self.assertIsNone(plan["blocked_job_id"])
        self.assertEqual(self.ledger.finish(job["job_id"], receipt, path)["status"], "completed")

    def test_failed_and_unknown_jobs_keep_consumer_blocked_and_recover(self):
        for phase in ("failed", "creation_unknown"):
            with self.subTest(phase=phase):
                consumer = f"consumer-{phase}"
                job = self.ledger.enqueue(consumer, "request-1", snapshot(), "gpt-6.1-sol")
                self.ledger.claim(job["job_id"])
                failed = self.ledger.fail(job["job_id"], phase, "secret /private/path sk-verysecret")
                self.assertNotIn("sk-", failed["failure_message"])
                self.assertNotIn("/private", failed["failure_message"])
                self.assertEqual(self.ledger.plan(consumer, snapshot())["pending_count"], 1)
                with self.assertRaises(LedgerBusy):
                    self.ledger.enqueue(consumer, "request-2", snapshot(), "gpt-6.1-sol")
                with self.assertRaises(LedgerError):
                    self.ledger.claim(job["job_id"])
                receipt, path = self.dossier(job)
                self.assertEqual(self.ledger.finish(job["job_id"], receipt, path)["status"], "completed")

    def test_changed_version_is_new_even_with_identical_content(self):
        self.complete(self.enqueue())
        value = snapshot()
        value["radar_signals"][0]["version"] = 2
        self.assertEqual(self.ledger.plan(self.consumer, value)["selected_count"], 1)

    def test_changed_text_is_new_even_when_version_and_upstream_hash_are_reused(self):
        self.complete(self.enqueue())
        value = snapshot()
        value["radar_signals"][0]["text"] += " A meaningful correction."
        self.assertEqual(self.ledger.plan(self.consumer, value)["selected_count"], 1)

    def test_observational_timestamps_and_review_state_do_not_repeat_work(self):
        self.complete(self.enqueue())
        value = snapshot()
        value["captured_at"] = "2026-11-12T13:00:00Z"
        value["radar_signals"][0].update(
            fetched_at="2026-11-12T13:00:00Z", read_at="2026-11-12T13:01:00Z",
            first_seen_at="2026-11-12T12:00:00Z", discovered_at="2026-11-12T12:00:00Z",
            review={"status": "pending", "reviewed_at": "2026-11-12T12:00:00Z"},
            retrieval={"consumer": "different-editorial-consumer", "snapshot_seq": 100},
            full_text_read=True, current_version=5,
        )
        self.assertEqual(self.ledger.plan(self.consumer, value)["selected_count"], 0)
        self.assertEqual(self.ledger.plan("separate-research-consumer", value)["selected_count"], 1)

    def test_no_signals_skip_is_idempotent_without_queued_job(self):
        value = snapshot(0)
        first = self.enqueue(value)
        self.assertEqual(first["status"], "no_signals")
        self.assertIsNone(first["job_id"])
        self.assertEqual(first["coverage"], value["coverage"])
        self.assertEqual(self.enqueue(value), first)
        self.assertIsNone(self.ledger.next_queued())
        with self.assertRaisesRegex(LedgerError, "different payload"):
            self.enqueue(snapshot())
        self.assertEqual(self.enqueue(snapshot(), request_id="request-2")["status"], "queued")

    def test_completed_signals_return_no_signals_without_a_new_job(self):
        self.complete(self.enqueue())
        result = self.enqueue(request_id="request-2")
        self.assertEqual(result["status"], "no_signals")
        self.assertEqual(result["researched_count"], 1)
        self.assertEqual(result["pending_count"], 0)

    def test_snapshot_is_frozen_and_can_be_loaded_after_restart(self):
        value = snapshot(2)
        job = self.enqueue(value, limit=1)
        value["radar_signals"][0]["text"] = "caller mutated input"
        job["snapshot"]["radar_signals"][0]["text"] = "caller mutated result"
        reopened = Ledger(self.db_path)
        recovered = reopened.get_job(job["job_id"])
        self.assertEqual(recovered["snapshot"]["radar_signals"][0]["text"], "Meaningful text 1")
        self.assertEqual(recovered["pending_remaining"], 1)
        self.assertEqual(reopened.next_queued()["job_id"], job["job_id"])
        self.assertEqual(reopened.next_queued(self.consumer)["job_id"], job["job_id"])
        self.assertIsNone(reopened.next_queued("other-consumer"))

    def test_list_unresolved_only_reads_claimed_jobs(self):
        first = self.enqueue()
        self.assertEqual(self.ledger.list_unresolved(), [])
        self.ledger.claim(first["job_id"])
        second = self.ledger.enqueue("other-consumer", "request-2", snapshot(), "gpt-6.1-sol")
        self.assertEqual([x["job_id"] for x in self.ledger.list_unresolved()], [first["job_id"]])
        self.ledger.claim(second["job_id"])
        self.ledger.fail(second["job_id"], "creation_unknown", "Unknown submission outcome.")
        reopened = Ledger(self.db_path)
        self.assertEqual([x["status"] for x in reopened.list_unresolved()], ["running", "unknown"])
        self.assertEqual(len(reopened.list_unresolved(limit=1)), 1)
        with self.assertRaises(LedgerError):
            reopened.list_unresolved(limit=True)

    def test_binding_cannot_change_or_accept_another_runs_receipt(self):
        job = self.enqueue()
        self.ledger.claim(job["job_id"])
        receipt, path = self.dossier(job)
        with self.assertRaises(LedgerError):
            self.ledger.bind_run(job["job_id"], path.parent, "another-run")
        with self.assertRaises(LedgerError):
            self.ledger.bind_run(job["job_id"], self.root / "another-directory")
        changed = {**receipt, "run_id": "another-run"}
        with self.assertRaisesRegex(LedgerError, "does not match"):
            self.ledger.finish(job["job_id"], changed, path)
        other_path = self.root / "other-dossier.md"
        other_path.write_bytes(path.read_bytes())
        with self.assertRaisesRegex(LedgerError, "does not belong"):
            self.ledger.finish(job["job_id"], receipt, other_path)
        self.assertEqual(self.ledger.plan(self.consumer, snapshot())["researched_count"], 0)

    def test_unclaimed_or_unbound_job_cannot_finish(self):
        job = self.enqueue()
        receipt, path = self.dossier(job)
        with self.assertRaisesRegex(LedgerError, "not been claimed"):
            self.ledger.finish(job["job_id"], receipt, path)
        other = self.ledger.enqueue("other-consumer", "request-1", snapshot(), "gpt-6.1-sol")
        self.ledger.claim(other["job_id"])
        with self.assertRaisesRegex(LedgerError, "does not match"):
            self.ledger.finish(other["job_id"], receipt, path)

    def test_completed_without_available_final_does_not_advance(self):
        job = self.enqueue()
        self.ledger.claim(job["job_id"])
        receipt, path = self.dossier(job)
        for phase, result_status in (("completed", "missing_final"), ("completed", None),
                                     ("resultavailable", "available")):
            with self.subTest(phase=phase, result_status=result_status):
                invalid = {**receipt, "phase": phase, "result_status": result_status}
                with self.assertRaisesRegex(LedgerError, "available dossier"):
                    self.ledger.finish(job["job_id"], invalid, path)
                self.assertEqual(self.ledger.plan(self.consumer, snapshot())["researched_count"], 0)
                self.assertEqual(self.ledger.get_job(job["job_id"])["status"], "running")

    def test_receipt_requires_expected_dossier_filename(self):
        job = self.enqueue()
        self.ledger.claim(job["job_id"])
        receipt, path = self.dossier(job)
        for filename in (None, "other.md", "../dossier.md"):
            with self.subTest(filename=filename):
                with self.assertRaisesRegex(LedgerError, "identify dossier.md"):
                    self.ledger.finish(job["job_id"], {**receipt, "dossier_file": filename}, path)
        self.assertEqual(self.ledger.plan(self.consumer, snapshot())["researched_count"], 0)

    def test_dossier_must_be_bounded_utf8_text_even_when_hash_matches(self):
        job = self.enqueue()
        self.ledger.claim(job["job_id"])
        receipt, path = self.dossier(job)
        for content, error in ((b"\xff", "UTF-8"), (b"x" * (MAX_DOSSIER_BYTES + 1), "permitted size")):
            with self.subTest(error=error):
                path.write_bytes(content)
                receipt["dossier_sha256"] = hashlib.sha256(content).hexdigest()
                with self.assertRaisesRegex(LedgerError, error):
                    self.ledger.finish(job["job_id"], receipt, path)
                self.assertEqual(self.ledger.plan(self.consumer, snapshot())["researched_count"], 0)

    def test_concurrent_successful_finish_records_each_signal_once(self):
        job = self.enqueue(snapshot(3))
        self.ledger.claim(job["job_id"])
        receipt, path = self.dossier(job)
        barrier = threading.Barrier(6)

        def finish(_):
            barrier.wait()
            return self.ledger.finish(job["job_id"], receipt, path)["status"]

        with ThreadPoolExecutor(max_workers=6) as pool:
            self.assertEqual(list(pool.map(finish, range(6))), ["completed"] * 6)
        with sqlite3.connect(self.db_path) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM researched_signals").fetchone()[0], 3)
        self.assertEqual(self.ledger.plan(self.consumer, snapshot(3))["researched_count"], 3)

    def test_completed_job_cannot_fail_or_change_dossier(self):
        job = self.complete(self.enqueue())
        with self.assertRaises(LedgerError):
            self.ledger.fail(job["job_id"], "failed", "Worker failed.")
        receipt, path = self.dossier(job, text="Different dossier\n")
        with self.assertRaisesRegex(LedgerError, "replace a completed"):
            self.ledger.finish(job["job_id"], receipt, path)
        self.assertEqual(self.ledger.plan(self.consumer, snapshot())["researched_count"], 1)

    def test_invalid_or_duplicate_signal_versions_fail_before_enqueue(self):
        for version in (None, 0, -1, True, "1", 2**63):
            with self.subTest(version=version):
                value = snapshot()
                value["radar_signals"][0]["version"] = version
                with self.assertRaises(LedgerError):
                    self.enqueue(value)
        value = snapshot()
        value["radar_signals"].append(deepcopy(value["radar_signals"][0]))
        with self.assertRaisesRegex(LedgerError, "duplicate"):
            self.enqueue(value)
        value = snapshot()
        del value["radar_signals"][0]["source_id"]
        with self.assertRaises(LedgerError):
            self.enqueue(value)
        self.assertIsNone(self.ledger.next_queued())

    def test_batch_bounds_and_strict_snapshot_validation(self):
        for limit in (0, 31, True, 1.0, "3"):
            with self.subTest(limit=limit):
                with self.assertRaises(LedgerError):
                    self.enqueue(limit=limit)
        self.assertEqual(self.ledger.plan(self.consumer, snapshot(31), 30)["pending_remaining"], 1)
        for key, invalid in (("schema_version", True), ("radar_signals", {}),
                             ("captured_at", "2026-11-12T10:00:00"), ("coverage", {})):
            with self.subTest(key=key):
                value = snapshot()
                value[key] = invalid
                with self.assertRaises(LedgerError):
                    self.enqueue(value)

    @unittest.skipIf(os.name == "nt", "POSIX permissions")
    def test_database_and_new_state_directory_are_private(self):
        self.assertEqual(self.db_path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.db_path.parent.stat().st_mode & 0o777, 0o700)


if __name__ == "__main__":
    unittest.main()
