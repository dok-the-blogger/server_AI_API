"""Offline checks; SDK-specific error cases run when OpenAI is installed. No network."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import importlib.util
import os
import unittest
from unittest.mock import patch

import pilot


SID = "sess_test"
TID = "turn_test"
PROJECT_ID = "proj_offline_budget_test"
HAS_SDK = importlib.util.find_spec("openai") is not None


def bad_request(body, *, request_id="req_offline_400", content_type="application/json",
                message="Error code: 400 - DO_NOT_SAVE_SDK_WRAPPER"):
    import httpx2
    from openai import BadRequestError
    request = httpx2.Request("POST", "https://api.openai.com/v1/agents/sessions",
                            headers={"authorization": "Bearer DO_NOT_SAVE_REQUEST_HEADER"})
    response = httpx2.Response(400, request=request,
                              headers={"x-request-id": request_id, "content-type": content_type,
                                       "x-private": "DO_NOT_SAVE_RESPONSE_HEADER"})
    return BadRequestError(message, response=response, body=body)


def turn(status="completed"):
    return {"id": TID, "session_id": SID, "subagent_id": None, "status": status,
            "usage": {"input_tokens": 100, "output_tokens": 20}}


def message(text="Проверяемое досье", phase="final_answer", status="completed", tid=TID):
    return {"id": "msg_final", "type": "message", "role": "assistant", "phase": phase,
            "status": status, "turn_id": tid,
            "content": [{"type": "output_text", "text": text}]}


class FakeStream:
    def __init__(self, events, disconnect=False):
        self.events = events
        self.disconnect = disconnect

    def with_result_collection(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        yield from self.events
        if self.disconnect:
            raise ConnectionError("provider body must not be logged")

    def get_final_result(self):
        return SimpleNamespace(session_id=SID, output_text="Not the persisted final answer")


class FakeSessions:
    def __init__(self):
        self.creates = 0
        self.cancels = 0
        self.deletes = 0
        self.status = "completed"
        self.disconnect = False
        self.fail_create = False
        self.spend_control = {"limit": 300, "consumed": 12}
        self.item_calls = []
        self.item_pages = {None: {"data": [message()], "has_more": False, "last_id": None}}
        self.events = SimpleNamespace(create=self.cancel)
        self.turns = SimpleNamespace(retrieve=self.get_turn, list=self.list_turns,
                                     items=SimpleNamespace(list=self.list_items))

    def create(self, **request):
        self.creates += 1
        self.request = request
        if self.fail_create:
            raise ConnectionError("sk-proj-" + "S" * 40)
        events = [{"type": "agent.session.turn.in_progress", "session_id": SID,
                   "turn_id": TID, "turn": turn("in_progress")}]
        if not self.disconnect:
            events.append({"type": "agent.session.turn.completed", "session_id": SID,
                           "turn_id": TID, "turn": turn(self.status)})
        return FakeStream(events, self.disconnect)

    def retrieve(self, session_id):
        assert session_id == SID
        return {"id": SID, "status": "idle", "required_actions": [],
                "spend_control": self.spend_control}

    def get_turn(self, *, turn_id, session_id):
        assert (turn_id, session_id) == (TID, SID)
        return turn(self.status)

    def list_turns(self, **kwargs):
        return {"data": [turn(self.status)], "has_more": False, "last_id": None}

    def list_items(self, **kwargs):
        self.item_calls.append(kwargs)
        return deepcopy(self.item_pages[kwargs.get("after")])

    def cancel(self, session_id, *, events):
        assert session_id == SID
        assert events == [{"type": "agent.session.input.cancel"}]
        self.cancels += 1
        return {}

    def delete(self, session_id):
        assert session_id == SID
        self.deletes += 1
        return {"deleted": True}


class PilotProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.skill = self.root / "skill"
        for name in pilot.POLICY_FILES:
            path = self.skill / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"Test policy for {name}\n", encoding="utf-8")
        self.snapshot = self.root / "snapshot.json"
        self.sample = json.loads((pilot.BASE / "input.example.json").read_text())
        self.write_snapshot()
        self.run_dir = self.root / "run"
        self.sessions = FakeSessions()
        self.client = SimpleNamespace(beta=SimpleNamespace(agents=SimpleNamespace(sessions=self.sessions)))

    def write_snapshot(self):
        self.snapshot.write_text(json.dumps(self.sample), encoding="utf-8")

    def prepare(self, project_id=None, spend_limit_cents=300):
        return pilot.prepare(self.skill, self.snapshot, self.run_dir, pilot.DEFAULT_MODEL,
                             False, spend_limit_cents, project_id)

    def mark_submitted(self, phase="in_progress"):
        self.prepare()
        pilot.update_receipt(self.run_dir, phase=phase, session_id=SID, turn_id=TID)

    def test_prepare_provenance_and_no_network(self):
        manifest = self.prepare()
        self.assertEqual(len(manifest["policy_files"]), 4)
        self.assertEqual(manifest["request_sha256"], pilot.digest((self.run_dir / "request.json").read_bytes()))
        self.assertEqual(pilot.receipt(self.run_dir)["phase"], "prepared")
        self.assertEqual(self.sessions.creates, 0)
        request = json.loads((self.run_dir / "request.json").read_text())
        self.assertEqual(request["spend_control"], {"limit": 300})
        self.assertEqual(request["agent"]["service_tier"], "default")
        self.assertEqual(request["metadata"]["dok_research_run_id"], manifest["run_id"])
        self.assertNotIn("budget_control", manifest)
        self.assertNotIn("budget_control", pilot.receipt(self.run_dir))
        self.assertEqual(pilot.budget_policy(self.run_dir), {"kind": "session", "limit_cents": 300})

    def test_project_budget_is_explicit_and_preserved_without_session_spend_control(self):
        manifest = self.prepare(PROJECT_ID)
        state = pilot.receipt(self.run_dir)
        request = json.loads((self.run_dir / "request.json").read_text())
        expected = {"kind": "project_hard_limit", "project_id": PROJECT_ID, "monthly_limit_cents": 300}
        self.assertNotIn("spend_control", request)
        self.assertEqual(manifest["budget_control"], expected)
        self.assertEqual(state["budget_control"], expected)
        self.assertEqual(pilot.budget_policy(self.run_dir), expected)
        self.assertEqual(manifest["spend_limit_cents"], 300)
        self.assertEqual(state["spend_limit_cents"], 300)
        self.assertNotIn("project_limit_confirmation", state)
        self.assertEqual(self.sessions.creates, 0)

    def test_invalid_project_or_excessive_project_budget_is_never_prepared(self):
        for project_id, limit in (("", 300), ("not_a_project", 300), ("proj_", 300),
                                  ("proj_bad\nheader", 300), ("proj_" + "a" * 101, 300),
                                  (PROJECT_ID, 301), (PROJECT_ID, True)):
            with self.subTest(project_id=project_id, limit=limit), self.assertRaises(pilot.PilotError):
                self.prepare(project_id, limit)
            self.assertFalse(self.run_dir.exists())

    def test_project_confirmation_and_client_binding_fail_before_attempt_is_recorded(self):
        self.prepare(PROJECT_ID)
        for confirmed, bound_project in ((False, PROJECT_ID), (True, None),
                                          (True, "proj_another"), ("true", PROJECT_ID)):
            self.client.project = bound_project
            with self.subTest(confirmed=confirmed, bound_project=bound_project), self.assertRaises(pilot.PilotError):
                pilot.run(self.client, self.run_dir, project_limit_confirmed=confirmed)
            state = pilot.receipt(self.run_dir)
            self.assertEqual(state["phase"], "prepared")
            self.assertEqual(state["submission_attempts"], 0)
            self.assertNotIn("project_limit_confirmation", state)
        self.assertEqual(self.sessions.creates, 0)

    def test_project_confirmation_is_saved_before_single_submission(self):
        self.prepare(PROJECT_ID)
        self.client.project = PROJECT_ID
        self.sessions.spend_control = None
        original_create = self.sessions.create

        def confirmed_create(**request):
            state = pilot.receipt(self.run_dir)
            self.assertEqual(state["phase"], "submitting")
            self.assertEqual(state["submission_attempts"], 1)
            confirmation = state["project_limit_confirmation"]
            self.assertEqual(set(confirmation), {"confirmed_at", "verified_by_api", "project_id", "monthly_limit_cents"})
            self.assertIs(confirmation["verified_by_api"], False)
            self.assertEqual(confirmation["project_id"], PROJECT_ID)
            self.assertEqual(confirmation["monthly_limit_cents"], 300)
            self.assertIsNotNone(pilot.datetime.fromisoformat(confirmation["confirmed_at"]).tzinfo)
            self.assertNotIn("spend_control", request)
            return original_create(**request)

        with patch.object(self.sessions, "create", side_effect=confirmed_create):
            result = pilot.run(self.client, self.run_dir, project_limit_confirmed=True)
            self.assertEqual(result["phase"], "completed")
            self.assertEqual(result["result_status"], "available")
            with self.assertRaises(pilot.PilotError):
                pilot.run(self.client, self.run_dir, project_limit_confirmed=True)
        self.assertEqual(self.sessions.creates, 1)

    def test_project_confirmation_flag_cannot_relabel_a_session_budget(self):
        self.prepare()
        self.client.project = PROJECT_ID
        with self.assertRaisesRegex(pilot.PilotError, "only valid"):
            pilot.run(self.client, self.run_dir, project_limit_confirmed=True)
        state = pilot.receipt(self.run_dir)
        self.assertEqual(state["phase"], "prepared")
        self.assertEqual(state["submission_attempts"], 0)
        self.assertNotIn("project_limit_confirmation", state)
        self.assertEqual(self.sessions.creates, 0)

    def test_budget_policy_rejects_partial_or_changed_project_protection(self):
        self.prepare(PROJECT_ID)
        manifest = pilot.read_json(self.run_dir / "manifest.json")
        state = pilot.receipt(self.run_dir)
        request = json.loads((self.run_dir / "request.json").read_text())
        cases = []
        for target in ("manifest", "receipt"):
            changed = deepcopy(manifest if target == "manifest" else state)
            changed.pop("budget_control")
            cases.append((f"missing_{target}", changed if target == "manifest" else manifest,
                          changed if target == "receipt" else state, request))
        changed = deepcopy(state)
        changed["budget_control"]["project_id"] = "proj_another"
        cases.append(("different_project", manifest, changed, request))
        changed = deepcopy(manifest)
        changed["budget_control"]["extra"] = True
        cases.append(("extra_policy_field", changed, state, request))
        changed = deepcopy(state)
        changed["spend_limit_cents"] = 299
        cases.append(("legacy_limit_mismatch", manifest, changed, request))
        for bad_limit in (True, 0, 301):
            m, s = deepcopy(manifest), deepcopy(state)
            m["budget_control"]["monthly_limit_cents"] = s["budget_control"]["monthly_limit_cents"] = bad_limit
            m["spend_limit_cents"] = s["spend_limit_cents"] = bad_limit
            cases.append((f"invalid_limit_{bad_limit}", m, s, request))
        for value in (None, {"limit": 300}):
            cases.append((f"unexpected_spend_control_{value}", manifest, state, {**request, "spend_control": value}))
        m, s = deepcopy(manifest), deepcopy(state)
        m.pop("budget_control")
        s.pop("budget_control")
        cases.append(("cannot_downgrade_to_missing_session_limit", m, s, request))
        self.client.project = PROJECT_ID
        for name, m, s, packet in cases:
            with self.subTest(name=name):
                raw = pilot.encode(packet)
                m, s = deepcopy(m), deepcopy(s)
                m["request_sha256"] = s["request_sha256"] = pilot.digest(raw)
                pilot.atomic_bytes(self.run_dir / "manifest.json", pilot.encode(m))
                pilot.atomic_bytes(self.run_dir / "receipt.json", pilot.encode(s))
                pilot.atomic_bytes(self.run_dir / "request.json", raw)
                with self.assertRaises(pilot.PilotError):
                    pilot.run(self.client, self.run_dir, project_limit_confirmed=True)
                self.assertEqual(pilot.receipt(self.run_dir)["submission_attempts"], 0)
        self.assertEqual(self.sessions.creates, 0)

    def test_client_for_run_uses_saved_project_and_legacy_signature(self):
        self.prepare(PROJECT_ID)
        with patch.object(pilot, "make_client", return_value=self.client) as factory:
            self.assertIs(pilot.client_for_run(self.run_dir), self.client)
            factory.assert_called_once_with(project_id=PROJECT_ID)
        legacy_dir = self.root / "legacy"
        pilot.prepare(self.skill, self.snapshot, legacy_dir, pilot.DEFAULT_MODEL, False)
        with patch.object(pilot, "make_client", return_value=self.client) as factory:
            self.assertIs(pilot.client_for_run(legacy_dir), self.client)
            factory.assert_called_once_with()

    def test_manifest_identity_mismatch_blocks_submission(self):
        self.prepare(PROJECT_ID)
        self.client.project = PROJECT_ID
        manifest = pilot.read_json(self.run_dir / "manifest.json")
        for field, value in (("run_id", "different-run"), ("request_sha256", "0" * 64)):
            changed = {**manifest, field: value}
            pilot.atomic_bytes(self.run_dir / "manifest.json", pilot.encode(changed))
            with self.subTest(field=field), self.assertRaises(pilot.PilotError):
                pilot.run(self.client, self.run_dir, project_limit_confirmed=True)
            self.assertEqual(pilot.receipt(self.run_dir)["submission_attempts"], 0)
        self.assertEqual(self.sessions.creates, 0)

    def test_saved_project_routing_survives_damaged_inputs_without_enabling_run(self):
        self.prepare(PROJECT_ID)
        self.client.project = PROJECT_ID
        pilot.update_receipt(self.run_dir, phase="in_progress", session_id=SID, turn_id=TID)
        (self.run_dir / "request.json").write_text("damaged input", encoding="utf-8")
        (self.run_dir / "manifest.json").unlink()
        with patch.object(pilot, "make_client", return_value=self.client) as factory:
            client = pilot.client_for_run(self.run_dir)
            factory.assert_called_once_with(project_id=PROJECT_ID)
            pilot.request_cancel(client, self.run_dir)
            with self.assertRaises(pilot.PilotError):
                pilot.run(client, self.run_dir, project_limit_confirmed=True)
        self.assertEqual(self.sessions.cancels, 1)
        self.assertEqual(self.sessions.creates, 0)

    def test_invalid_saved_project_policy_never_falls_back_to_default_project(self):
        self.prepare(PROJECT_ID)
        policy = pilot.receipt(self.run_dir)["budget_control"]
        pilot.update_receipt(self.run_dir, budget_control={**policy, "monthly_limit_cents": 299})
        with patch.object(pilot, "make_client") as factory:
            with self.assertRaises(pilot.PilotError):
                pilot.client_for_run(self.run_dir)
            factory.assert_not_called()

    @unittest.skipUnless(HAS_SDK, "Requires the pinned local OpenAI SDK")
    def test_real_sdk_project_header_and_no_repeated_post(self):
        import httpx2
        from openai import OpenAI as SDKOpenAI
        self.prepare(PROJECT_ID)
        calls = []

        def respond(request):
            calls.append(request)
            self.assertEqual(request.method, "POST")
            self.assertEqual(request.url.path, "/v1/agents/sessions")
            self.assertEqual(request.headers["OpenAI-Project"], PROJECT_ID)
            self.assertEqual(request.headers["OpenAI-Beta"], "agents=v1")
            self.assertNotIn("spend_control", json.loads(request.content))
            state = pilot.receipt(self.run_dir)
            self.assertEqual(state["submission_attempts"], 1)
            self.assertIs(state["project_limit_confirmation"]["verified_by_api"], False)
            return httpx2.Response(400, json={"error": {"message": "Offline transport rejection.",
                                                        "type": "invalid_request_error", "code": "test_error"}})

        http = httpx2.Client(transport=httpx2.MockTransport(respond), trust_env=False)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-project-test-only", "OPENAI_PROJECT_ID": "proj_wrong_env"}), \
                patch("openai.OpenAI", side_effect=lambda **kwargs: SDKOpenAI(http_client=http, **kwargs)):
            with pilot.client_for_run(self.run_dir) as client:
                self.assertEqual(client.project, PROJECT_ID)
                self.assertEqual(client.max_retries, 0)
                with self.assertRaises(pilot.PilotError):
                    pilot.run(client, self.run_dir, project_limit_confirmed=True)
                with self.assertRaises(pilot.PilotError):
                    pilot.run(client, self.run_dir, project_limit_confirmed=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(pilot.receipt(self.run_dir)["phase"], "creation_rejected")

    def test_invalid_budget_never_prepares_or_submits(self):
        for limit in (None, 0, -1, True, 10_001):
            with self.assertRaises(pilot.PilotError):
                pilot.prepare(self.skill, self.snapshot, self.run_dir, pilot.DEFAULT_MODEL, False, limit)
        self.assertFalse(self.run_dir.exists())

    def test_spend_keeps_last_known_value_when_latest_unknown(self):
        self.mark_submitted()
        first = pilot.recover(self.client, self.run_dir)
        self.assertEqual(first["spend_control"], {"limit": 300, "consumed": 12})
        self.sessions.spend_control = {"limit": 300, "consumed": None}
        later = pilot.recover(self.client, self.run_dir)
        self.assertEqual(later["spend_control"]["consumed"], 12)
        self.assertTrue(later["latest_spend_unknown"])

    def test_missing_required_policy_fails_before_creating_run(self):
        (self.skill / pilot.POLICY_FILES[2]).unlink()
        with self.assertRaises(pilot.PilotError):
            self.prepare()
        self.assertFalse(self.run_dir.exists())

    def test_invalid_coverage_and_secret_input_are_rejected(self):
        self.sample["coverage"]["archive"] = "probably_full"
        self.write_snapshot()
        with self.assertRaises(pilot.PilotError):
            self.prepare()
        self.sample["coverage"]["archive"] = "partial"
        self.sample["radar_signals"] = [{"body": "sk-proj-" + "S" * 40}]
        self.write_snapshot()
        with self.assertRaises(pilot.PilotError):
            self.prepare()

    def test_create_unknown_cannot_repeat_paid_submission_or_log_secret(self):
        self.prepare()
        self.sessions.fail_create = True
        with self.assertRaises(pilot.PilotError):
            pilot.run(self.client, self.run_dir)
        state = pilot.receipt(self.run_dir)
        self.assertEqual(state["phase"], "creation_unknown")
        self.assertEqual(state["api_stage"], "create_request")
        self.assertFalse(state["create_accepted"])
        self.assertNotIn("sk-proj", json.dumps(state))
        with self.assertRaises(pilot.PilotError):
            pilot.run(self.client, self.run_dir)
        self.assertEqual(self.sessions.creates, 1)

    @unittest.skipUnless(HAS_SDK, "Requires the pinned local OpenAI SDK")
    def test_real_create_400_is_rejected_and_never_repeated(self):
        self.prepare()
        error = bad_request({"message": "Invalid tool field.", "type": "invalid_request_error",
                             "param": "agent.tools[0].type", "code": "invalid_request_error"})
        with patch.object(self.sessions, "create", side_effect=error) as create:
            with self.assertRaisesRegex(pilot.PilotError, "API rejected session creation.*HTTP 400") as caught:
                pilot.run(self.client, self.run_dir)
            self.assertIs(caught.exception.__cause__, error)
            state = pilot.receipt(self.run_dir)
            self.assertEqual(state["phase"], "creation_rejected")
            self.assertEqual(state["api_stage"], "create_request")
            self.assertEqual(state["submission_attempts"], 1)
            self.assertFalse(state["create_accepted"])
            self.assertIsNone(state["session_id"])
            self.assertEqual(state["observer_error"]["request_id"], "req_offline_400")
            self.assertEqual(state["observer_error"]["error"]["param"], "agent.tools[0].type")
            self.assertEqual(pilot.recover(self.client, self.run_dir)["phase"], "creation_rejected")
            with self.assertRaises(pilot.PilotError):
                pilot.run(self.client, self.run_dir)
            self.assertEqual(create.call_count, 1)
        self.assertEqual(self.sessions.cancels, 0)

    @unittest.skipUnless(HAS_SDK, "Requires the pinned local OpenAI SDK")
    def test_400_after_create_before_first_event_is_not_creation_rejected(self):
        self.prepare()
        error = bad_request({"message": "Stream failure.", "type": "invalid_request_error"})
        stream = FakeStream([])
        with patch.object(self.sessions, "create", return_value=stream) as create, \
                patch.object(FakeStream, "__enter__", side_effect=error):
            with self.assertRaisesRegex(pilot.PilotError, "observer stopped") as caught:
                pilot.run(self.client, self.run_dir)
            self.assertIs(caught.exception.__cause__, error)
            state = pilot.receipt(self.run_dir)
            self.assertEqual(state["phase"], "creation_unknown")
            self.assertEqual(state["api_stage"], "observe_stream")
            self.assertTrue(state["create_accepted"])
            self.assertEqual(state["submission_attempts"], 1)
            with self.assertRaises(pilot.PilotError):
                pilot.run(self.client, self.run_dir)
            self.assertEqual(create.call_count, 1)

    @unittest.skipUnless(HAS_SDK, "Requires the pinned local OpenAI SDK")
    def test_recovery_error_keeps_terminal_turn_and_recovery_stage(self):
        self.prepare()
        error = bad_request({"message": "Saved items unavailable.", "type": "invalid_request_error"})
        with patch.object(self.sessions, "retrieve", side_effect=error):
            with self.assertRaisesRegex(pilot.PilotError, "Retrieval of saved output failed") as caught:
                pilot.run(self.client, self.run_dir)
        self.assertIs(caught.exception.__cause__, error)
        state = pilot.receipt(self.run_dir)
        self.assertEqual(state["phase"], "completed")
        self.assertEqual(state["api_stage"], "recover_saved_output")
        self.assertEqual(state["session_id"], SID)
        self.assertEqual(self.sessions.creates, 1)
        self.assertEqual(self.sessions.cancels, 0)
        self.assertFalse((self.run_dir / "dossier.md").exists())

    @unittest.skipUnless(HAS_SDK, "Requires the pinned local OpenAI SDK")
    def test_receipt_diagnostic_failure_does_not_mask_create_error_or_allow_replay(self):
        self.prepare()
        error = bad_request({"message": "Invalid field.", "type": "invalid_request_error"})
        with patch.object(self.sessions, "create", side_effect=error) as create, \
                patch.object(pilot, "update_receipt", side_effect=OSError("DO_NOT_LOG_DISK_DETAILS")):
            with self.assertRaisesRegex(pilot.PilotError, "diagnostic could not be saved") as caught:
                pilot.run(self.client, self.run_dir)
            self.assertIs(caught.exception.__cause__, error)
            self.assertEqual(pilot.receipt(self.run_dir)["phase"], "submitting")
            with self.assertRaises(pilot.PilotError):
                pilot.run(self.client, self.run_dir)
            self.assertEqual(create.call_count, 1)

    def test_non_sdk_400_does_not_prove_rejection(self):
        self.prepare()
        error = ConnectionError("DO_NOT_LOG_ERROR_STRING")
        error.status_code = 400
        with patch.object(self.sessions, "create", side_effect=error):
            with self.assertRaisesRegex(pilot.PilotError, "creation could not be confirmed"):
                pilot.run(self.client, self.run_dir)
        self.assertEqual(pilot.receipt(self.run_dir)["phase"], "creation_unknown")

    def test_changed_prepared_payload_is_not_submitted(self):
        self.prepare()
        (self.run_dir / "request.json").write_text("{}")
        with self.assertRaises(pilot.PilotError):
            pilot.run(self.client, self.run_dir)
        self.assertEqual(self.sessions.creates, 0)

    def test_normal_run_uses_persisted_final_not_commentary(self):
        self.prepare()
        self.sessions.item_pages[None]["data"].insert(0, message("Intermediate notes", "commentary"))
        result = pilot.run(self.client, self.run_dir)
        self.assertEqual(result["phase"], "completed")
        self.assertEqual(result["result_status"], "available")
        self.assertEqual(result["api_stage"], "recover_saved_output")
        self.assertEqual((self.run_dir / "dossier.md").read_text(), "Проверяемое досье")
        self.assertEqual(self.sessions.creates, 1)

    def test_disconnect_then_resume_uses_same_session_without_new_input(self):
        self.prepare()
        self.sessions.disconnect = True
        with self.assertRaises(pilot.PilotError):
            pilot.run(self.client, self.run_dir)
        self.assertEqual(pilot.receipt(self.run_dir)["session_id"], SID)
        result = pilot.recover(self.client, self.run_dir)
        self.assertEqual(result["result_status"], "available")
        self.assertEqual(self.sessions.creates, 1)
        self.assertEqual(self.sessions.cancels, 1)

    def test_cancel_diagnostic_failure_does_not_replace_stream_error(self):
        self.prepare()
        self.sessions.disconnect = True
        with patch.object(pilot, "request_cancel", side_effect=OSError("DO_NOT_LOG_CANCEL_DETAILS")):
            with self.assertRaisesRegex(pilot.PilotError, "observer stopped") as caught:
                pilot.run(self.client, self.run_dir)
        self.assertIsInstance(caught.exception.__cause__, ConnectionError)
        state = pilot.receipt(self.run_dir)
        self.assertEqual(state["phase"], "observation_interrupted")
        self.assertEqual(state["api_stage"], "observe_stream")
        self.assertNotIn("DO_NOT_LOG", json.dumps(state))

    def test_pagination_uses_server_last_id_and_rejects_incomplete_final(self):
        self.mark_submitted()
        self.sessions.item_pages = {
            None: {"data": [message("Commentary", "commentary")], "has_more": True,
                   "last_id": "opaque_cursor_unrelated_to_item_id"},
            "opaque_cursor_unrelated_to_item_id": {"data": [message(), message("Wrong turn", tid="other")],
                                                   "has_more": False, "last_id": None},
        }
        result = pilot.recover(self.client, self.run_dir)
        self.assertEqual(result["result_status"], "available")
        self.assertEqual(self.sessions.item_calls[1]["after"], "opaque_cursor_unrelated_to_item_id")
        self.assertEqual((self.run_dir / "dossier.md").read_text(), "Проверяемое досье")

    def test_completed_without_final_is_explicitly_missing(self):
        self.mark_submitted()
        self.sessions.item_pages[None]["data"] = [message("Partial", status="incomplete"),
                                                 message("Notes", phase="commentary")]
        state = pilot.recover(self.client, self.run_dir)
        self.assertEqual(state["result_status"], "missing_final")
        self.assertFalse((self.run_dir / "dossier.md").exists())

    def test_no_idle_equals_success_and_cancel_needs_confirmation(self):
        self.mark_submitted()
        self.sessions.status = "in_progress"
        state = pilot.recover(self.client, self.run_dir)
        self.assertEqual(state["session_status"], "idle")
        self.assertEqual(state["phase"], "in_progress")
        state = pilot.request_cancel(self.client, self.run_dir)
        self.assertFalse(state["cancel_confirmed"])
        self.sessions.status = "cancelled"
        self.assertTrue(pilot.recover(self.client, self.run_dir)["cancel_confirmed"])
        self.assertFalse((self.run_dir / "dossier.md").exists())

    def test_subagent_completion_does_not_finish_root(self):
        self.mark_submitted()
        child = dict(turn(), id="turn_child", subagent_id="sub_child")
        pilot.record_event(self.run_dir, {"type": "agent.session.turn.completed", "session_id": SID,
                                          "turn_id": "turn_child", "turn": child})
        self.assertEqual(pilot.receipt(self.run_dir)["phase"], "in_progress")
        self.assertEqual(pilot.receipt(self.run_dir)["turn_id"], TID)

    def test_unknown_usage_does_not_erase_last_known_counts(self):
        self.mark_submitted()
        pilot.update_receipt(self.run_dir, usage={"input_tokens": 123, "output_tokens": 45})
        self.sessions.turns.retrieve = lambda **kw: dict(turn(), usage=None)
        state = pilot.recover(self.client, self.run_dir)
        self.assertEqual(state["usage"]["input_tokens"], 123)
        self.assertTrue(state["latest_usage_unknown"])

    def test_failure_code_is_retained_without_server_message(self):
        self.mark_submitted()
        self.sessions.turns.retrieve = lambda **kw: dict(
            turn("failed"), error={"code": "credit_balance_exhausted", "message": "private provider body"})
        state = pilot.recover(self.client, self.run_dir)
        self.assertEqual(state["failure_code"], "credit_balance_exhausted")
        self.assertNotIn("private provider", json.dumps(state))

    def test_close_requires_saved_result_and_does_not_claim_physical_cleanup(self):
        self.mark_submitted()
        with self.assertRaises(pilot.PilotError):
            pilot.close_session(self.client, self.run_dir)
        pilot.recover(self.client, self.run_dir)
        state = pilot.close_session(self.client, self.run_dir)
        self.assertTrue(state["session_deleted"])
        self.assertEqual(state["environment_cleanup"], "asynchronous")
        pilot.close_session(self.client, self.run_dir)
        self.assertEqual(self.sessions.deletes, 1)


@unittest.skipUnless(HAS_SDK, "Requires the pinned local OpenAI SDK")
class SafeSDKErrorTests(unittest.TestCase):
    def test_real_sdk_error_preserves_fields_and_redacts_every_secret_before_bounding(self):
        exact_key = "FAKE_EXACT_ENVIRONMENT_KEY_12345"
        generic_key = "sk-proj-" + "S" * 40
        text = (f"Invalid agent.tools[0].type. Key={exact_key}; {generic_key}; "
                "Bearer bearer-secret; proxy http://alice:proxy-password@proxy.invalid/path?token=query-secret#fragment-secret. "
                + "x" * 3000 + " " + exact_key)
        fields = {"message": text, "code": "invalid_request_error", "type": "invalid_request_error",
                  "param": "agent.tools[0].type", "request": "DO_NOT_SAVE_EXTRA_BODY"}
        for body in (fields, {"error": fields}):
            with self.subTest(nested="error" in body), patch.dict(os.environ, {"OPENAI_API_KEY": " " + exact_key + " "}):
                result = pilot.safe_error(bad_request(body))
            self.assertEqual(result["type"], "BadRequestError")
            self.assertEqual(result["http_status"], 400)
            self.assertEqual(result["request_id"], "req_offline_400")
            self.assertEqual(result["content_type"], "application/json")
            self.assertEqual(result["body_format"], "error_object")
            self.assertEqual(result["error"]["code"], "invalid_request_error")
            self.assertEqual(result["error"]["type"], "invalid_request_error")
            self.assertEqual(result["error"]["param"], "agent.tools[0].type")
            self.assertLessEqual(len(result["error"]["message"]), 2000)
            saved = json.dumps(result)
            for secret in (exact_key, generic_key, "bearer-secret", "alice", "proxy-password",
                           "query-secret", "fragment-secret", "DO_NOT_SAVE"):
                self.assertNotIn(secret, saved)

    def test_html_or_malformed_body_never_uses_sdk_body_wrapper(self):
        for body, expected in (("<html>DO_NOT_SAVE_HTML</html>", "html"),
                               (["DO_NOT_SAVE_ARRAY"], "json_array"),
                               ({"error": "DO_NOT_SAVE_STRING_ERROR"}, "json_object")):
            with self.subTest(body_format=expected):
                result = pilot.safe_error(bad_request(body, content_type="text/html; charset=utf-8"))
                self.assertEqual(result["body_format"], expected)
                self.assertNotIn("error", result)
                self.assertNotIn("DO_NOT_SAVE", json.dumps(result))
        result = pilot.safe_error(bad_request(None))
        self.assertNotIn("error", result)

    def test_identifier_fields_are_allowlisted_bounded_and_secret_redacted(self):
        key = "FAKE_EXACT_ENVIRONMENT_KEY_12345"
        with patch.dict(os.environ, {"OPENAI_API_KEY": key}):
            result = pilot.safe_error(bad_request({"message": "A useful message.", "code": key,
                                                   "type": "x" * 129, "param": "Bearer hidden-token"},
                                                  request_id=key))
        self.assertNotIn("request_id", result)
        self.assertEqual(result["error"], {"message": "A useful message."})
        self.assertNotIn(key, json.dumps(result))

    def test_plain_sdk_connection_message_is_safe_but_arbitrary_exception_text_is_ignored(self):
        import httpx2
        from openai import APIConnectionError
        error = APIConnectionError(request=httpx2.Request("POST", "https://api.openai.com/v1/agents/sessions"))
        result = pilot.safe_error(error)
        self.assertEqual(result["error"], {"message": "Connection error."})
        self.assertEqual(pilot.safe_error(ValueError("DO_NOT_SAVE_EXCEPTION")), {"type": "ValueError"})

    def test_sanitizer_failure_never_masks_original_exception(self):
        error = bad_request({"message": "A useful message."})
        with patch("bootstrap.redact", side_effect=RuntimeError("DO_NOT_SAVE_REDACTOR_FAILURE")):
            result = pilot.safe_error(error)
        self.assertEqual(result, {"type": "Exception"})


if __name__ == "__main__":
    unittest.main()
