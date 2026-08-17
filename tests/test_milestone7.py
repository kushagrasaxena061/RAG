import pytest
from fastapi.testclient import TestClient
from adaptive_rag.api.app import app

client = TestClient(app)

def test_api_health():
    """Verify the API framework mounts correctly."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_end_to_end_query_and_telemetry():
    """Verify that a query returns an answer and perfectly tracks token efficiency."""
    response = client.post("/ask", json={"query": "What is our Q3 revenue?", "mock_mode": True})
    
    assert response.status_code == 200
    data = response.json()
    
    assert "answer" in data
    assert "telemetry" in data
    
    telemetry = data["telemetry"]
    
    assert telemetry["total_tokens"] > 0
    assert telemetry["retrieved_chunks"] == 10
    assert telemetry["final_chunks_used"] == 2
    assert telemetry["compression_ratio"] == 0.2
    assert telemetry["latency_ms"] >= 0
