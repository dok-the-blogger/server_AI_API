"""Exercise the installed SDK over a fake transport, never the network."""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pilot


@unittest.skipUnless(importlib.util.find_spec("openai"), "isolated SDK not installed")
class InstalledSDKContractTests(unittest.TestCase):
    def test_stream_to_saved_dossier_and_bounded_wire_request(self):
        import httpx2
        from openai import OpenAI

        sid, tid = "sess_offline", "turn_offline"
        usage = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
        root = {"id": tid, "agent_id": "agent_offline", "session_id": sid,
                "subagent_id": None, "object": "agent.session.turn", "created_at": 1,
                "status": "completed", "usage": usage}
        message = {"id": "msg_final", "type": "message", "turn_id": tid,
                   "role": "assistant", "phase": "final_answer", "status": "completed",
                   "content": [{"type": "output_text", "text": "Offline final dossier", "annotations": []}]}
        session = {"id": sid, "object": "agent.session", "agent": {"model": pilot.DEFAULT_MODEL},
                   "environment": {"type": "openai_hosted"}, "created_at": 1, "last_active_at": 2,
                   "status": "idle", "required_actions": [], "metadata": {}, "vault_ids": [],
                   "usage": usage, "spend_control": {"limit": 300, "consumed": 12}}
        events = [
            {"type": "agent.session.turn.created", "session_id": sid, "turn_id": tid,
             "turn": {**root, "status": "queued"}},
            {"type": "agent.session.turn.in_progress", "session_id": sid, "turn_id": tid,
             "turn": {**root, "status": "in_progress"}},
            {"type": "agent.session.turn.item.done", "session_id": sid, "turn_id": tid,
             "item": message, "output_index": 0},
            {"type": "agent.session.turn.completed", "session_id": sid, "turn_id": tid, "turn": root},
            {"type": "agent.session.idle", "session": session},
        ]
        body = b"".join(
            ("data: " + json.dumps({**event, "event_id": "event_" + str(i)}) + "\n\n").encode()
            for i, event in enumerate(events)
        )

        class IdleThenOpen(httpx2.SyncByteStream):
            def __iter__(self):
                yield body
                raise AssertionError("Do not wait forever after terminal root + idle")

        calls = []
        def respond(request):
            calls.append(request)
            self.assertEqual(request.url.host, "api.openai.com")
            self.assertEqual(request.headers["OpenAI-Beta"], "agents=v1")
            if request.method == "POST":
                self.assertEqual(request.url.path, "/v1/agents/sessions")
                sent = json.loads(request.content)
                self.assertEqual(sent["spend_control"], {"limit": 300})
                self.assertEqual(sent["agent"]["service_tier"], "default")
                self.assertEqual(sent["environment"], {"type": "openai_hosted"})
                self.assertEqual(sent["agent"]["tools"][0]["type"], "web_search")
                return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=IdleThenOpen())
            if request.url.path.endswith("/items"):
                return httpx2.Response(200, json={"object": "list", "data": [message],
                                                   "has_more": False, "last_id": "msg_final"})
            if request.url.path.endswith("/" + tid):
                return httpx2.Response(200, json=root)
            self.assertEqual(request.url.path, "/v1/agents/sessions/" + sid)
            return httpx2.Response(200, json=session)

        with TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in pilot.POLICY_FILES:
                path = folder / "skill" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("Offline policy fixture", encoding="utf-8")
            snapshot = json.loads((pilot.BASE / "input.example.json").read_text())
            snapshot["captured_at"] = datetime.now(timezone.utc).isoformat()
            (folder / "snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
            run_dir = folder / "run"
            pilot.prepare(folder / "skill", folder / "snapshot.json", run_dir, pilot.DEFAULT_MODEL, False)
            # No real credential and no network-capable transport are present.
            http = httpx2.Client(transport=httpx2.MockTransport(respond), trust_env=False)
            with OpenAI(api_key="offline-contract-only", http_client=http,
                        base_url="https://api.openai.com/v1", max_retries=0) as client:
                result = pilot.run(client, run_dir, 30)
            self.assertEqual(result["phase"], "completed")
            self.assertEqual(result["result_status"], "available")
            self.assertEqual(result["spend_control"]["consumed"], 12)
            self.assertEqual((run_dir / "dossier.md").read_text(), "Offline final dossier")
            self.assertEqual([r.method for r in calls], ["POST", "GET", "GET", "GET"])


if __name__ == "__main__":
    unittest.main()
