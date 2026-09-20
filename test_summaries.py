import asyncio
import hashlib
import json

import httpx
import pytest
from fastapi.testclient import TestClient

import main
from config import settings
from summaries import MAX_BODY_CHARS, MAX_RESPONSE_BYTES

AUTH = {"Authorization": "Bearer client-test-token"}
ARTICLE = {"title": "Запуск перенесён", "body_text": "Компания обещала запуск 1 июня, но перенесла его на июль. Новая дата не подтверждена.",
           "body_format": "doknews-markup-v1", "publication_date": "2026-06-01"}


def provider_result():
    return {"model": "glm-5.3-flash", "choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": json.dumps({"tldr": "Компания перенесла запуск с 1 июня на июль; новая дата не подтверждена."})}}],
        "usage": {"prompt_tokens": 150, "completion_tokens": 40, "total_tokens": 190}}


@pytest.fixture
def service(monkeypatch):
    for name, value in {"API_TOKEN": "client-test-token", "DIGITALOCEAN_API_KEY": "provider-test-key",
                        "GROK_API_KEY": "", "GIGACHAT_CREDENTIALS": "",
                        "SUMMARIES_MODEL": "glm-5.3-flash"}.items():
        monkeypatch.setattr(settings, name, value)
    state = {"requests": [], "reply": None, "error": None, "delay": 0}

    async def respond(request):
        state["requests"].append(request)
        if state["error"]:
            raise state["error"]
        if state["delay"]:
            await asyncio.sleep(state["delay"])
        if callable(state["reply"]):
            return state["reply"](request)
        return state["reply"] if state["reply"] is not None else httpx.Response(200, json=provider_result())

    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: original(*a, transport=httpx.MockTransport(respond), **kw))
    with TestClient(main.app) as client:
        state["provider"] = main.app.state.summaries_client
        yield client, state


def test_real_router_provider_contract_and_provenance(service):
    client, state = service
    response = client.post("/summaries", headers=AUTH, json=ARTICLE)
    assert response.status_code == 200
    result = response.json()
    assert result["tldr"] == "Компания перенесла запуск с 1 июня на июль; новая дата не подтверждена."
    assert (result["model"], result["provider"], result["prompt_version"]) == (
        "glm-5.3-flash", "digitalocean", "doknews-tldr-v1")
    assert result["usage"]["total_tokens"] == 190
    assert len(state["requests"]) == 1
    sent = state["requests"][0]
    payload = json.loads(sent.content)
    assert str(sent.url) == "https://inference.do-ai.run/v1/chat/completions"
    assert sent.headers["authorization"] == "Bearer provider-test-key"
    assert json.loads(payload["messages"][1]["content"]) == ARTICLE
    assert payload["messages"][0]["role"] == "system"
    assert payload["max_completion_tokens"] == 1024
    assert payload["reasoning_effort"] == "none"


@pytest.mark.parametrize("change", [{"body_text": " "}, {"body_text": "x" * (MAX_BODY_CHARS + 1)},
    {"title": 5}, {"model": "expensive"}, {"body_text": "\ud800"}, {"prompt": "override"},
    {"preset": "unknown"}, {"instruction": " "}, {"instruction": "x" * 8001},
    {"max_completion_tokens": 99999}])
def test_invalid_inputs_never_call_provider(service, change):
    client, state = service
    response = client.post("/summaries", headers={**AUTH, "Content-Type": "application/json"},
                           content=json.dumps({**ARTICLE, **change}))
    assert response.status_code == 422
    assert state["requests"] == []
    assert "input" not in response.json()["detail"][0]


def test_auth_and_unconfigured_provider(service):
    client, state = service
    assert client.post("/summaries", json=ARTICLE).status_code == 401
    assert state["requests"] == []
    provider = main.app.state.summaries_client
    main.app.state.summaries_client = None
    try:
        response = client.post("/summaries", headers=AUTH, json=ARTICLE)
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "summaries_not_configured"
    finally:
        main.app.state.summaries_client = provider


@pytest.mark.parametrize("model", ["glm-5.3-flash", "deepseek-v4.1-flash"])
def test_selected_model_preset_and_additive_instruction_cross_real_router(service, model):
    client, state = service
    reply = provider_result(); reply["model"] = model
    state["reply"] = httpx.Response(200, json=reply)
    request = {**ARTICLE, "model": model, "preset": "doknews-tldr-v2", "instruction": "Сохрани новую дату."}
    response = client.post("/summaries", headers=AUTH, json=request)
    assert response.status_code == 200
    result = response.json()
    sent = json.loads(state["requests"][0].content)
    assert sent["model"] == result["model"] == model
    assert result["prompt_version"] == result["preset"] == "doknews-tldr-v2"
    assert result["prompt_hash"] == hashlib.sha256(sent["messages"][0]["content"].encode()).hexdigest()
    assert result["elapsed_ms"] >= 0
    assert "Сохрани новую дату." in sent["messages"][0]["content"]
    assert json.loads(sent["messages"][1]["content"]) == ARTICLE
    assert len(state["requests"]) == 1


def test_general_summary_without_article_title_or_preset(service):
    client, state = service
    response = client.post("/summaries", headers=AUTH, json={
        "body_text": "Произвольный текст", "preset": None, "instruction": "Одно предложение."})
    assert response.status_code == 200
    assert response.json()["prompt_version"] == "summary-v1"
    assert response.json()["preset"] is None
    assert len(state["requests"]) == 1


@pytest.mark.parametrize("default_model,requested_model,selected_model,effort", [
    ("glm-5.3-flash", "deepseek-v4.1-flash", "deepseek-v4.1-flash", "low"),
    ("deepseek-v4.1-flash", None, "deepseek-v4.1-flash", "low"),
    ("deepseek-v4.1-flash", "glm-5.3-flash", "glm-5.3-flash", "none"),
    ("glm-5.3-flash", None, "glm-5.3-flash", "none"),
])
def test_provider_reasoning_contract_for_effective_model(
        service, default_model, requested_model, selected_model, effort):
    client, state = service
    state["provider"].model = default_model

    def provider_reply(request):
        sent = json.loads(request.content)
        # Reproduce DigitalOcean's observed rejection, including its allowed levels.
        if sent["model"] == "deepseek-v4.1-flash" and sent["reasoning_effort"] not in {
                "low", "high", "xhigh", "max"}:
            return httpx.Response(400, json={"error": {
                "message": "reasoning_effort must be one of [low high xhigh max] for this model"}})
        reply = provider_result()
        reply["model"] = selected_model
        return httpx.Response(200, json=reply)

    state["reply"] = provider_reply
    response = client.post("/summaries", headers=AUTH, json={
        **ARTICLE, "model": requested_model, "preset": "doknews-tldr-v2"})
    assert response.status_code == 200
    assert response.json()["model"] == selected_model
    assert len(state["requests"]) == 1
    sent = json.loads(state["requests"][0].content)
    assert sent["reasoning_effort"] == effort
    assert sent["response_format"] == {"type": "json_object"}
    assert sent["max_completion_tokens"] == (2048 if selected_model == "deepseek-v4.1-flash" else 1024)
    assert sent["stream"] is False
    assert json.loads(sent["messages"][1]["content"]) == ARTICLE


@pytest.mark.parametrize("status,code", [(402, "provider_payment_required"), (429, "provider_rate_limited"),
    (401, "provider_authentication_failed"), (400, "provider_input_rejected"),
    (503, "provider_error"), (302, "provider_error")])
def test_provider_failures_are_safe_and_not_retried(service, status, code):
    client, state = service
    state["reply"] = httpx.Response(status, json={"error": "provider-test-key private text"},
                                    headers={"Location": "https://other.example/"})
    response = client.post("/summaries", headers=AUTH, json=ARTICLE)
    assert response.json()["detail"]["code"] == code
    assert "provider-test-key" not in response.text
    assert "private text" not in response.text
    assert len(state["requests"]) == 1


@pytest.mark.parametrize("invalid", ["length", "blank", "fields", "model", "usage", "tool", "refusal", "json", "huge", "long_tldr", "message_type", "choices_type", "root_type"])
def test_invalid_model_output_never_becomes_summary(service, invalid, caplog):
    client, state = service
    payload = provider_result()
    choice = payload["choices"][0]
    if invalid == "length": choice["finish_reason"] = "length"
    elif invalid == "blank": choice["message"]["content"] = '{"tldr":"  "}'
    elif invalid == "fields": choice["message"]["content"] = '{"tldr":"ok","extra":"bad"}'
    elif invalid == "model": payload["model"] = "another-model"
    elif invalid == "usage": payload["usage"]["total_tokens"] = 0
    elif invalid == "tool": choice["message"]["tool_calls"] = [{"name": "call"}]
    elif invalid == "refusal": choice["message"]["refusal"] = "refused"
    elif invalid == "json": choice["message"]["content"] = "not JSON"
    elif invalid == "huge": choice["message"]["content"] = "x" * MAX_RESPONSE_BYTES
    elif invalid == "long_tldr": choice["message"]["content"] = json.dumps({"tldr": "x" * 901})
    elif invalid == "message_type": choice["message"] = []
    elif invalid == "choices_type": payload["choices"] = {"0": choice}
    elif invalid == "root_type": payload = []
    state["reply"] = httpx.Response(200, json=payload)
    response = client.post("/summaries", headers=AUTH, json=ARTICLE)
    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "invalid_provider_response"
    assert "Summary provider response rejected: model=glm-5.3-flash stage=" in caplog.text
    if invalid == "length":
        assert "stage=output_limit" in caplog.text
    if invalid == "long_tldr":
        assert "stage=summary_schema validation_type=string_too_long" in caplog.text
    assert str(choice.get("message")) not in caplog.text


def test_total_timeout_and_transport(service):
    client, state = service
    state["provider"].timeout = 0.01
    state["delay"] = 0.1
    assert client.post("/summaries", headers=AUTH, json=ARTICLE).status_code == 504
    state["delay"] = 0
    state["error"] = httpx.ConnectError("private")
    response = client.post("/summaries", headers=AUTH, json=ARTICLE)
    assert response.json()["detail"]["code"] == "provider_unavailable"
    assert len(state["requests"]) == 2
