import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

import main
import profiles
from config import settings
from completion_providers import MAX_RESPONSE_BYTES

AUTH = {"Authorization": "Bearer service-test-token"}
FLASH = "mimo-v2.6-flash"
PRO = "mimo-v2.6-pro"


@pytest.fixture
def service(monkeypatch):
    for name, value in {
        "API_TOKEN": "service-test-token", "MIMO_API_KEY": "mimo-only-secret",
        "DIGITALOCEAN_API_KEY": "do-only-secret", "GROK_API_KEY": "",
        "GIGACHAT_CREDENTIALS": "", "SUMMARIES_MODEL": "glm-5.3-flash",
        "MIMO_MODEL": FLASH, "MIMO_MAX_TOKENS": 1024, "MIMO_TIMEOUT_SECONDS": 60,
    }.items():
        monkeypatch.setattr(settings, name, value)
    state = {"requests": [], "reply": None, "status": 200, "delay": 0, "error": None}

    async def respond(request):
        state["requests"].append(request)
        if state["delay"]:
            await asyncio.sleep(state["delay"])
        if state["error"]:
            raise state["error"]
        sent = json.loads(request.content)
        reply = state["reply"]
        if reply is None:
            content = json.dumps({"tldr": "Краткая сводка."}) if "response_format" in sent else "Ответ MiMo."
            reply = {"model": sent["model"], "choices": [{"finish_reason": "stop", "message": {
                "role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        return httpx.Response(state["status"], json=reply,
                              headers={"Location": "https://unrelated.example/"})

    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: original(
        *a, transport=httpx.MockTransport(respond), **kw))
    with TestClient(main.app) as client:
        clients = list(main.app.state.summaries_client.providers.values())
        yield client, state
    assert all(p.http.is_closed for p in clients)


@pytest.mark.parametrize("path,model", [("/summaries", FLASH), ("/chat", FLASH), ("/chat", PRO)])
def test_mimo_routes_credentials_parameters_and_response(service, path, model):
    client, state = service
    body = {"body_text": "Текст новости", "preset": None} if path == "/summaries" else {
        "message": "Привет", "session_id": "session-1"}
    response = client.post(path, headers=AUTH, json={**body, "model": model})
    assert response.status_code == 200
    assert response.json()["model"] == model
    if path == "/summaries":
        assert response.json()["provider"] == "mimo"
        assert response.json()["prompt_version"] == "summary-v1"
    else:
        assert response.json()["response"] == "Ответ MiMo."
        assert response.json()["session_id"] == "session-1"
    assert len(state["requests"]) == 1
    sent = state["requests"][0]
    assert str(sent.url) == "https://api.xiaomimimo.com/v1/chat/completions"
    assert sent.headers["authorization"] == "Bearer mimo-only-secret"
    assert "do-only-secret" not in str(sent.headers)
    payload = json.loads(sent.content)
    assert payload["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in payload
    assert payload["max_completion_tokens"] == 1024
    assert payload["stream"] is False
    assert payload.get("response_format") == ({"type": "json_object"} if path == "/summaries" else None)


def test_summary_default_and_explicit_do_models_keep_do_key(service):
    client, state = service
    for selection, expected in [(None, "glm-5.3-flash"), ("deepseek-v4.1-flash", "deepseek-v4.1-flash")]:
        response = client.post("/summaries", headers=AUTH, json={"body_text": "Text", "model": selection})
        assert response.status_code == 200
        assert response.json()["provider"] == "digitalocean"
        assert response.json()["model"] == expected
        sent = state["requests"][-1]
        assert sent.url.host == "inference.do-ai.run"
        assert sent.headers["authorization"] == "Bearer do-only-secret"
        assert "thinking" not in json.loads(sent.content)


def test_unconfigured_mimo_never_uses_other_provider(service, monkeypatch):
    client, state = service
    with monkeypatch.context() as patch:
        patch.setattr(main.app.state, "mimo_client", None)
        patch.delitem(main.app.state.summaries_client.providers, "mimo")
        for path, body in [("/chat", {"message": "Text"}), ("/summaries", {"body_text": "Text"})]:
            response = client.post(path, headers=AUTH, json={**body, "model": FLASH})
            assert response.status_code == 503
            assert response.json()["detail"]["code"].endswith("not_configured")
        assert client.get("/models?provider=mimo", headers=AUTH).status_code == 503
    assert not state["requests"]


def test_mimo_summary_does_not_require_do_configuration(service, monkeypatch):
    client, state = service
    with monkeypatch.context() as patch:
        patch.delitem(main.app.state.summaries_client.providers, "digitalocean")
        assert client.post("/summaries", headers=AUTH, json={"body_text": "Text", "model": FLASH}).status_code == 200
        assert client.post("/summaries", headers=AUTH, json={"body_text": "Text"}).status_code == 503
    assert len(state["requests"]) == 1


@pytest.mark.parametrize("path,body", [
    ("/chat", {"message": "Text", "model": "arbitrary"}),
    ("/summaries", {"body_text": "Text", "model": PRO}),
])
def test_allowlists_before_provider(service, path, body):
    client, state = service
    assert client.post(path, headers=AUTH, json=body).status_code == 422
    assert not state["requests"]


def test_chat_profile_and_explicit_model_override_preserve_context(service, monkeypatch):
    client, state = service
    monkeypatch.setitem(profiles._profiles, "mimo-test", {
        "system_prompt": "Profile prompt", "provider": "mimo", "model": PRO,
        "user_template": "User: {message}"})
    request = {"message": "Now", "profile": "mimo-test", "context": {
        "system": "Context prompt", "history": [{"role": "assistant", "content": "Earlier"}]}}
    assert client.post("/chat", headers=AUTH, json=request).json()["model"] == PRO
    assert client.post("/chat", headers=AUTH, json={**request, "model": FLASH}).json()["model"] == FLASH
    for sent in state["requests"]:
        assert json.loads(sent.content)["messages"] == [
            {"role": "system", "content": "Context prompt"},
            {"role": "assistant", "content": "Earlier"},
            {"role": "user", "content": "User: Now"}]


@pytest.mark.parametrize("provider", ["gigachat", "grok"])
def test_legacy_chat_without_model_preserves_provider(service, monkeypatch, provider):
    client, state = service
    response = SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="stop", message=SimpleNamespace(content="Legacy answer"))])
    backend = AsyncMock()
    if provider == "gigachat":
        backend.achat.return_value = response
        monkeypatch.setattr(main.app.state, "gigachat_client", backend)
        request = {"message": "Text"}
    else:
        backend.chat.completions.create.return_value = response
        monkeypatch.setattr(main.app.state, "xai_client", backend)
        monkeypatch.setitem(profiles._profiles, "legacy", {"provider": "grok", "system_prompt": "System"})
        request = {"message": "Text", "profile": "legacy"}
    result = client.post("/chat", headers=AUTH, json=request)
    assert result.status_code == 200
    assert result.json()["model"] == provider
    assert not state["requests"]


def test_mimo_model_list_is_local_and_authenticated(service):
    client, state = service
    assert client.get("/models?provider=mimo").status_code == 401
    response = client.get("/models?provider=mimo", headers=AUTH)
    assert response.status_code == 200
    assert {m["id"] for m in response.json()["models"]} == {FLASH, PRO}
    assert response.json()["current_model"] == FLASH
    assert not state["requests"]


@pytest.mark.parametrize("status,code", [
    (401, "provider_authentication_failed"), (402, "provider_payment_required"),
    (403, "provider_access_denied"), (421, "provider_content_filtered"),
    (429, "provider_rate_limited"), (503, "provider_error"), (302, "provider_error"),
])
def test_mimo_summary_errors_are_safe_and_specific(service, status, code):
    client, state = service
    state.update(status=status, reply={"error": "mimo-only-secret private input"})
    result = client.post("/summaries", headers=AUTH, json={"body_text": "Text", "model": FLASH})
    assert result.json()["detail"]["code"] == code
    assert "DigitalOcean" not in result.text
    assert "mimo-only-secret" not in result.text and "private input" not in result.text
    assert len(state["requests"]) == 1


def test_mimo_chat_filter_maps_to_existing_filtered_response(service):
    client, state = service
    state.update(status=421, reply={"error": "private input"})
    result = client.post("/chat", headers=AUTH, json={"message": "Text", "model": PRO})
    assert result.status_code == 200
    assert result.json()["filtered"] is True and result.json()["response"] == ""
    assert len(state["requests"]) == 1


@pytest.mark.parametrize("reply", [[], {"model": "other", "choices": []},
    {"model": FLASH, "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": ""}}]},
    {"model": FLASH, "choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": "private"}}]},
    {"model": FLASH, "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "x" * MAX_RESPONSE_BYTES}}]},
])
def test_mimo_invalid_chat_response_is_sanitized(service, reply):
    client, state = service
    state["reply"] = reply
    result = client.post("/chat", headers=AUTH, json={"message": "Text", "model": FLASH})
    assert result.status_code == 502
    assert result.json()["detail"]["code"] == "invalid_provider_response"
    assert "private" not in result.text


def test_mimo_chat_timeout_transport_and_auth(service, monkeypatch):
    client, state = service
    body = {"message": "Text", "model": FLASH}
    assert client.post("/chat", json=body).status_code == 401
    assert not state["requests"]
    monkeypatch.setattr(settings, "MIMO_TIMEOUT_SECONDS", 0.01)
    state["delay"] = 0.1
    assert client.post("/chat", headers=AUTH, json=body).status_code == 504
    state.update(delay=0, error=httpx.ConnectError("private input"))
    result = client.post("/chat", headers=AUTH, json=body)
    assert result.json()["detail"]["code"] == "provider_unavailable"
    assert "private input" not in result.text
