import asyncio
import copy
import json

import httpx
import pytest
from fastapi.testclient import TestClient

import main
from config import settings

AUTH = {"Authorization": "Bearer classifier-test-token"}
MODEL = "typesafe-jev-1.13.0"


def provider_result(intent="news", mode="search"):
    def answer(choice, options):
        return {"type": "choice", "choice": choice, "confidence": 0.98,
                "probabilities": {option: int(option == choice) for option in options}}
    return {"model": MODEL, "answers": {
        "intent": answer(intent, ["news", "login", "assistant"]),
        "news_mode": answer(mode, ["latest", "search"]),
    }, "usage": {"input_tokens": 520, "output_tokens": 60}}


@pytest.fixture
def service(monkeypatch):
    for name, value in {"API_TOKEN": "classifier-test-token", "DIGITALOCEAN_API_KEY": "provider-test-key",
                        "GROK_API_KEY": "", "GIGACHAT_CREDENTIALS": "", "JEV_MODEL": MODEL}.items():
        monkeypatch.setattr(settings, name, value)
    state = {"requests": [], "status": 200, "result": provider_result(), "delay": 0}

    async def respond(request):
        state["requests"].append(request)
        if state["delay"]:
            await asyncio.sleep(state["delay"])
        return httpx.Response(state["status"], json=state["result"])

    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: original(*a, transport=httpx.MockTransport(respond), **kw))
    with TestClient(main.app) as client:
        yield client, state


@pytest.mark.parametrize("intent,mode", [("news", "latest"), ("news", "search"), ("login", "latest"), ("assistant", "latest")])
def test_authenticated_route_calls_systemone_once(service, intent, mode):
    client, state = service
    state["result"] = provider_result(intent, mode)
    text = "Покажи свежие новости про ИИ"
    response = client.post("/classify/dokbot", headers=AUTH, json={"text": text})
    assert response.status_code == 200
    assert (response.json()["intent"], response.json()["news_mode"]) == (intent, mode)
    assert response.json()["model"] == MODEL
    assert response.json()["prompt_version"] == "dokbot-intents-v1"
    assert len(state["requests"]) == 1
    request = state["requests"][0]
    assert str(request.url) == "https://inference.do-ai.run/v1/systemone"
    assert request.headers["authorization"] == "Bearer provider-test-key"
    sent = json.loads(request.content)
    assert sent["state"] == {"message": text}
    assert set(sent["questions"]["intent"]["criteria"]) == {"news", "login", "assistant"}
    assert "messages" not in sent


def test_client_auth_denies_before_provider_and_loopback_mode_preserved(service, monkeypatch):
    client, state = service
    for headers in ({}, {"Authorization": "Bearer wrong"}):
        assert client.post("/classify/dokbot", headers=headers, json={"text": "Новости"}).status_code == 401
    assert not state["requests"]
    monkeypatch.setattr(settings, "API_TOKEN", "")
    assert client.post("/classify/dokbot", json={"text": "Новости"}).status_code == 200


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": " "}, {"text": "x" * 4097}, {"text": 123},
                                     {"text": "\ud800"}, {"text": "hi", "model": "other"}])
def test_invalid_input_never_reaches_provider(service, payload):
    client, state = service
    response = client.post("/classify/dokbot", headers={**AUTH, "Content-Type": "application/json"}, content=json.dumps(payload))
    assert response.status_code == 422
    assert not state["requests"]
    assert "input" not in response.json()["detail"][0]


@pytest.mark.parametrize("status,code", [(402, "provider_payment_required"), (403, "provider_access_denied"),
    (401, "provider_authentication_failed"), (429, "provider_rate_limited"), (500, "provider_error"), (302, "provider_error")])
def test_provider_errors_are_not_an_assistant_decision(service, status, code):
    client, state = service
    state.update(status=status, result={"secret": "upstream-body-must-not-leak"})
    response = client.post("/classify/dokbot", headers=AUTH, json={"text": "Хочу войти"})
    assert response.status_code == (429 if status == 429 else 502)
    assert response.json()["detail"]["code"] == code
    assert "upstream-body" not in response.text
    assert len(state["requests"]) == 1


@pytest.mark.parametrize("change", ["unknown_intent", "nan", "missing", "model", "not_object", "distribution"])
def test_malformed_provider_decision_is_rejected(service, change):
    client, state = service
    result = copy.deepcopy(state["result"])
    if change == "unknown_intent": result["answers"]["intent"]["choice"] = "run_shell"
    if change == "nan": result["answers"]["intent"]["confidence"] = "NaN"
    if change == "missing": del result["answers"]["news_mode"]
    if change == "model": result["model"] = "other"
    if change == "not_object": result = []
    if change == "distribution": result["answers"]["intent"]["probabilities"]["login"] = 1
    state["result"] = result
    response = client.post("/classify/dokbot", headers=AUTH, json={"text": "hello"})
    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "invalid_provider_response"


def test_timeout_and_missing_provider(service):
    client, state = service
    provider = main.app.state.jev_client
    provider.timeout = 0.01
    state["delay"] = 0.2
    response = client.post("/classify/dokbot", headers=AUTH, json={"text": "Привет"})
    assert response.status_code == 504
    assert len(state["requests"]) == 1
    main.app.state.jev_client = None
    try:
        response = client.post("/classify/dokbot", headers=AUTH, json={"text": "Привет"})
        assert response.status_code == 503
    finally:
        main.app.state.jev_client = provider
