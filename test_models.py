import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock
from main import app
from config import settings

class MockModelInfo:
    def __init__(self, id_, owned_by):
        self.id_ = id_
        self.owned_by = owned_by

class MockModelsResponse:
    def __init__(self, data):
        self.data = data

@pytest.fixture
def client():
    # Setup test app state
    settings.API_TOKEN = "test-token"
    settings.GIGACHAT_MODEL = "GigaChat-Plus"
    return TestClient(app)

def test_get_models_no_token(client):
    response = client.get("/models")
    assert response.status_code == 401
    assert response.json()["detail"] == "Unauthorized"

def test_get_models_wrong_token(client):
    response = client.get("/models", headers={"Authorization": "Bearer wrong-token"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Unauthorized"

def test_get_models_no_client(client):
    app.state.gigachat_client = None
    response = client.get("/models", headers={"Authorization": f"Bearer {settings.API_TOKEN}"})
    assert response.status_code == 500
    assert response.json()["detail"] == "GigaChat client is not initialized"

def test_get_models_success(client):
    mock_client = AsyncMock()
    mock_models_response = MockModelsResponse([
        MockModelInfo(id_="model-1", owned_by="sber"),
        MockModelInfo(id_="model-2", owned_by="sber")
    ])
    mock_client.aget_models.return_value = mock_models_response
    app.state.gigachat_client = mock_client

    response = client.get("/models", headers={"Authorization": f"Bearer {settings.API_TOKEN}"})
    assert response.status_code == 200

    data = response.json()
    assert data["provider"] == "gigachat"
    assert data["current_model"] == settings.GIGACHAT_MODEL
    assert len(data["models"]) == 2

    assert data["models"][0]["id"] == "model-1"
    assert data["models"][0]["provider"] == "gigachat"
    assert data["models"][0]["owned_by"] == "sber"

    assert data["models"][1]["id"] == "model-2"
    assert data["models"][1]["provider"] == "gigachat"
    assert data["models"][1]["owned_by"] == "sber"

    mock_client.aget_models.assert_awaited_once()

def test_get_models_exception(client):
    mock_client = AsyncMock()
    mock_client.aget_models.side_effect = Exception("API error")
    app.state.gigachat_client = mock_client

    response = client.get("/models", headers={"Authorization": f"Bearer {settings.API_TOKEN}"})
    assert response.status_code == 500
    assert response.json()["detail"] == "API error"
