"""
tests/integration/test_health.py
──────────────────────────────────
Day 1: Smoke tests — verify the app starts and health endpoint responds.
"""

from fastapi.testclient import TestClient


def test_health_returns_ok(client: TestClient):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "environment" in body


def test_docs_endpoint_accessible(client: TestClient):
    response = client.get("/docs")
    assert response.status_code == 200


def test_unknown_route_returns_404(client: TestClient):
    response = client.get("/api/v1/nonexistent")
    assert response.status_code == 404
