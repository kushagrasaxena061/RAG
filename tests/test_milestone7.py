import pytest
from fastapi.testclient import TestClient
from adaptive_rag.api.app import app

client = TestClient(app)

def test_api_health_reset():
    response = client.post("/reset")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
