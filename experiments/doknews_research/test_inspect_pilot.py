"""Exercise inspection over the actual SDK's GET-only fake transport."""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import inspect_pilot
import pilot


@unittest.skipUnless(importlib.util.find_spec("openai"), "isolated SDK not installed")
class InspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        for name in pilot.POLICY_FILES:
            path = self.folder / "skill" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Offline inspection test policy", encoding="utf-8")
        snapshot = json.loads((pilot.BASE / "input.example.json").read_text())
        snapshot["captured_at"] = datetime.now(timezone.utc).isoformat()
        (self.folder / "snapshot.json").write_text(json.dumps(snapshot))
        self.run_dir = self.folder / "run"
        pilot.prepare(self.folder / "skill", self.folder / "snapshot.json", self.run_dir,
                      pilot.DEFAULT_MODEL, False)
        self.state = pilot.update_receipt(self.run_dir, phase="creation_unknown", submission_attempts=1,
                                          observer_error={"type": "BadRequestError", "http_status": 400})
        self.receipt_before = (self.run_dir / "receipt.json").read_bytes()

    def inspect(self, sessions=None, fail_sessions=False):
        import httpx2
        from openai import OpenAI
        calls = []

        def respond(request):
            calls.append(request)
            self.assertEqual(request.method, "GET", "Inspection must never create or change API work")
            self.assertEqual(request.url.host, "api.openai.com")
            if request.url.path == "/v1/agents/sessions":
                self.assertEqual(request.url.params["metadata[dok_research_run_id]"], self.state["run_id"])
                self.assertEqual(request.url.params["metadata[purpose]"], "doknews_manual_research_pilot")
                if fail_sessions:
                    return httpx2.Response(400, headers={"x-request-id": "req_inspection_test"},
                                           json={"error": {"code": "invalid_request_error", "type": "invalid_request_error",
                                                            "param": "metadata", "message": "Inspection filter rejected"}})
                return httpx2.Response(200, json={"object": "list", "data": sessions or [],
                                                  "has_more": False, "last_id": None})
            self.assertEqual(request.url.path, "/v1/models/" + pilot.DEFAULT_MODEL)
            return httpx2.Response(200, json={"id": pilot.DEFAULT_MODEL, "object": "model",
                                              "created": 0, "owned_by": "offline-test"})

        http = httpx2.Client(transport=httpx2.MockTransport(respond), trust_env=False)
        with OpenAI(api_key="offline-inspection-only", http_client=http,
                    base_url="https://api.openai.com/v1", max_retries=0) as client:
            result = inspect_pilot.inspect_attempt(client, self.run_dir)
        self.assertEqual(len(calls), 2)
        self.assertEqual((self.run_dir / "receipt.json").read_bytes(), self.receipt_before)
        self.assertFalse(result["receipt_modified"])
        self.assertFalse(result["new_research"])
        self.assertFalse(result["automatic_retry"])
        return result

    def test_empty_filtered_list_never_enables_another_submission(self):
        result = self.inspect()
        self.assertEqual(result["session_lookup"]["status"], "no_match_in_filtered_results")
        self.assertTrue(result["session_lookup"]["absence_is_not_proven"])
        self.assertEqual(result["model_lookup"]["status"], "available")
        self.assertEqual(pilot.receipt(self.run_dir)["submission_attempts"], 1)

    def test_only_exact_metadata_matches_are_saved_without_prompts(self):
        def session(sid, run_id):
            return {"id": sid, "object": "agent.session", "agent": {"model": pilot.DEFAULT_MODEL,
                       "instructions": "prompt-body-must-not-be-saved"}, "environment": {"type": "openai_hosted"},
                    "created_at": 1, "last_active_at": 2, "status": "idle",
                    "metadata": {"dok_research_run_id": run_id, "purpose": "doknews_manual_research_pilot"},
                    "spend_control": {"limit": 300, "consumed": 2}}
        result = self.inspect([session("sess_ours", self.state["run_id"]), session("sess_other", "unrelated")])
        self.assertEqual([s["id"] for s in result["session_lookup"]["sessions"]], ["sess_ours"])
        self.assertEqual(result["session_lookup"]["unmatched_records"], 1)
        self.assertNotIn("prompt-body-must-not-be-saved", json.dumps(result))
        self.assertNotIn("sess_other", json.dumps(result))

    def test_http_refusal_is_reported_and_does_not_stop_independent_model_get(self):
        result = self.inspect(fail_sessions=True)
        error = result["session_lookup"]["error"]
        self.assertEqual(error["http_status"], 400)
        self.assertEqual(error["request_id"], "req_inspection_test")
        self.assertEqual(error["error"]["param"], "metadata")
        self.assertEqual(error["error"]["message"], "Inspection filter rejected")
        self.assertEqual(result["model_lookup"]["status"], "available")


if __name__ == "__main__":
    unittest.main()
