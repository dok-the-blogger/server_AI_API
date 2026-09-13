import asyncio
import copy
import json

import httpx
import pytest
from fastapi.testclient import TestClient

import main
from config import settings
from embeddings import DigitalOceanEmbeddings, MAX_BATCH_CHARS, MAX_INPUT_CHARS


MODEL = "qwen3-embedding-0.6b"
AUTH = {"Authorization": "Bearer client-test-token"}


def provider_result(count=1):
    return {
        "object": "list",
        "model": MODEL,
        "data": [
            {"object": "embedding", "index": i, "embedding": [0.01 * (i + 1)] * 1024}
            for i in range(count)
        ],
        "usage": {"prompt_tokens": 12, "total_tokens": 12},
    }


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr(settings, "API_TOKEN", "client-test-token")
    monkeypatch.setattr(settings, "DIGITALOCEAN_API_KEY", "provider-test-key")
    monkeypatch.setattr(settings, "EMBEDDINGS_MODEL", MODEL)
    monkeypatch.setattr(settings, "EMBEDDINGS_DIMENSIONS", 1024)
    monkeypatch.setattr(settings, "GROK_API_KEY", "")
    monkeypatch.setattr(settings, "GIGACHAT_CREDENTIALS", "")
    state = {"requests": [], "reply": None, "error": None, "delay": 0}

    async def respond(request):
        state["requests"].append(request)
        if state["error"]:
            raise state["error"]
        if state["delay"]:
            await asyncio.sleep(state["delay"])
        if state["reply"] is not None:
            return state["reply"]
        count = len(json.loads(request.content)["input"])
        return httpx.Response(200, json=provider_result(count))

    original_http_client = httpx.AsyncClient

    def initialize_with_transport(*args, **kwargs):
        return original_http_client(*args, transport=httpx.MockTransport(respond), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", initialize_with_transport)
    with TestClient(main.app) as client:
        state["provider"] = main.app.state.embeddings_client
        yield client, state


@pytest.mark.parametrize("inputs", ["  Текст\nс пробелами.  ", ["Документ один", "Документ два", "Документ один"]])
def test_document_contract(service, inputs):
    client, state = service
    response = client.post("/embeddings", headers=AUTH, json={"input": inputs})
    assert response.status_code == 200
    texts = [inputs] if isinstance(inputs, str) else inputs
    result = response.json()
    assert [row["index"] for row in result["data"]] == list(range(len(texts)))
    assert all(len(row["embedding"]) == 1024 for row in result["data"])
    assert result["data"][0]["embedding"][0] == 0.01
    assert result["model"] == MODEL
    assert result["provider"] == "digitalocean"
    assert result["dimensions"] == 1024
    assert result["input_type"] == "document"
    assert result["preprocessing_version"] == "qwen3-retrieval-v1"
    assert result["usage"] == {"prompt_tokens": 12, "total_tokens": 12}
    assert len(state["requests"]) == 1
    upstream = state["requests"][0]
    assert str(upstream.url) == "https://inference.do-ai.run/v1/embeddings"
    assert upstream.headers["Authorization"] == "Bearer provider-test-key"
    assert json.loads(upstream.content) == {"model": MODEL, "input": texts, "encoding_format": "float"}


def test_queries_use_qwen_instruction(service):
    client, state = service
    response = client.post("/embeddings", headers=AUTH, json={"input": ["Первый?", "Второй?"], "input_type": "query"})
    assert response.status_code == 200
    assert response.json()["input_type"] == "query"
    prefix = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
    assert json.loads(state["requests"][0].content)["input"] == [prefix + "Первый?", prefix + "Второй?"]


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong-token"}])
def test_auth_prevents_provider_call(service, headers):
    client, state = service
    response = client.post("/embeddings", headers=headers, json={"input": "текст"})
    assert response.status_code == 401
    assert state["requests"] == []


def test_existing_optional_auth_policy(service, monkeypatch):
    client, _ = service
    monkeypatch.setattr(settings, "API_TOKEN", "")
    assert client.post("/embeddings", json={"input": "текст"}).status_code == 200


@pytest.mark.parametrize("payload", [
    {}, {"input": ""}, {"input": " \n "}, {"input": []}, {"input": ["нормально", ""]},
    {"input": 123}, {"input": ["нормально", 1]}, {"input": [[1, 2]]},
    {"input": "текст", "input_type": "search"}, {"input": "текст", "model": "another-model"},
    {"input": ["текст"] * 65}, {"input": "x" * (MAX_INPUT_CHARS + 1)},
])
def test_invalid_input_never_reaches_provider(service, payload):
    client, state = service
    assert client.post("/embeddings", headers=AUTH, json=payload).status_code == 422
    assert state["requests"] == []


def test_invalid_unicode(service):
    client, state = service
    response = client.post("/embeddings", headers={**AUTH, "Content-Type": "application/json"}, content=b'{"input":"\\ud800"}')
    assert response.status_code == 422
    assert state["requests"] == []


def test_prepared_character_limits_and_batch_boundary(service):
    client, state = service
    assert client.post("/embeddings", headers=AUTH, json={"input": "x" * MAX_INPUT_CHARS}).status_code == 200
    assert len(json.loads(state["requests"][-1].content)["input"][0]) == MAX_INPUT_CHARS
    response = client.post("/embeddings", headers=AUTH, json={"input": "x" * MAX_INPUT_CHARS, "input_type": "query"})
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "input_too_large"
    assert len(state["requests"]) == 1
    assert client.post("/embeddings", headers=AUTH, json={"input": ["x"] * 64}).status_code == 200
    texts = ["x" * MAX_INPUT_CHARS] * (MAX_BATCH_CHARS // MAX_INPUT_CHARS)
    assert client.post("/embeddings", headers=AUTH, json={"input": texts}).status_code == 200
    before = len(state["requests"])
    assert client.post("/embeddings", headers=AUTH, json={"input": texts + ["x"]}).status_code == 422
    assert len(state["requests"]) == before


@pytest.mark.parametrize("status,expected,code", [
    (400, 422, "provider_input_rejected"), (413, 422, "provider_input_rejected"),
    (422, 422, "provider_input_rejected"), (429, 429, "provider_rate_limited"),
    (401, 502, "provider_authentication_failed"), (403, 502, "provider_authentication_failed"),
    (500, 502, "provider_error"), (404, 502, "provider_error"), (302, 502, "provider_error"),
])
def test_provider_errors_no_retry_or_leak(service, status, expected, code):
    client, state = service
    state["reply"] = httpx.Response(status, json={"error": "provider-test-key private text"}, headers={"Location": "https://other.example/"})
    response = client.post("/embeddings", headers=AUTH, json={"input": "текст"})
    assert response.status_code == expected
    assert response.json()["detail"]["code"] == code
    assert "provider-test-key" not in response.text
    assert "private text" not in response.text
    assert len(state["requests"]) == 1


@pytest.mark.parametrize("error,expected,code", [
    (httpx.ReadTimeout("provider-test-key"), 504, "provider_timeout"),
    (httpx.ConnectError("provider-test-key"), 502, "provider_unavailable"),
])
def test_transport_errors(service, error, expected, code):
    client, state = service
    state["error"] = error
    response = client.post("/embeddings", headers=AUTH, json={"input": "текст"})
    assert response.status_code == expected
    assert response.json()["detail"]["code"] == code
    assert "provider-test-key" not in response.text
    assert len(state["requests"]) == 1


def test_total_timeout(service):
    client, state = service
    state["provider"].timeout = 0.01
    state["delay"] = 0.1
    assert client.post("/embeddings", headers=AUTH, json={"input": "текст"}).status_code == 504
    assert len(state["requests"]) == 1


@pytest.mark.parametrize("invalid", ["missing_row", "duplicate_index", "wrong_dimension", "nan", "boolean", "zero", "wrong_model", "wrong_usage", "missing_usage"])
def test_invalid_provider_results_are_not_success(service, invalid):
    client, state = service
    payload = provider_result(2)
    if invalid == "missing_row":
        payload["data"].pop()
    elif invalid == "duplicate_index":
        payload["data"][1]["index"] = 0
    elif invalid == "wrong_dimension":
        payload["data"][1]["embedding"].pop()
    elif invalid == "nan":
        payload["data"][0]["embedding"][0] = float("nan")
    elif invalid == "boolean":
        payload["data"][0]["embedding"][0] = True
    elif invalid == "zero":
        payload["data"][0]["embedding"] = [0.0] * 1024
    elif invalid == "wrong_model":
        payload["model"] = "other-model"
    elif invalid == "wrong_usage":
        payload["usage"]["total_tokens"] = -1
    elif invalid == "missing_usage":
        del payload["usage"]
    state["reply"] = httpx.Response(200, content=json.dumps(payload))
    response = client.post("/embeddings", headers=AUTH, json={"input": ["один", "два"]})
    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "invalid_provider_response"
    assert "data" not in response.json()


def test_invalid_json_and_ordered_results(service):
    client, state = service
    state["reply"] = httpx.Response(200, text="not JSON")
    assert client.post("/embeddings", headers=AUTH, json={"input": "текст"}).status_code == 502
    payload = provider_result(2)
    expected = copy.deepcopy(payload["data"])
    payload["data"].reverse()
    state["reply"] = httpx.Response(200, json=payload)
    response = client.post("/embeddings", headers=AUTH, json={"input": ["один", "два"]})
    assert response.status_code == 200
    assert response.json()["data"] == expected


def test_without_provider_key_and_health(monkeypatch):
    for name in ("DIGITALOCEAN_API_KEY", "GROK_API_KEY", "GIGACHAT_CREDENTIALS"):
        monkeypatch.setattr(settings, name, "")
    monkeypatch.setattr(settings, "API_TOKEN", "client-test-token")
    with TestClient(main.app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        response = client.post("/embeddings", headers=AUTH, json={"input": "текст"})
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "embeddings_not_configured"


def test_client_is_closed_on_shutdown(service):
    client, state = service
    provider = state["provider"]
    assert not provider.http.is_closed
    client.__exit__(None, None, None)
    assert provider.http.is_closed
