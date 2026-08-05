"""API shell smoke tests."""

from fastapi.testclient import TestClient

from services.api.app import create_app


def make_client() -> TestClient:
    return TestClient(create_app())


def test_health_is_alive_without_model() -> None:
    with make_client() as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.1.0"}


def test_ready_is_503_without_model() -> None:
    with make_client() as client:
        response = client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["ready"] is False
    assert body["model_loaded"] is False


def test_unknown_route_is_bounded_404() -> None:
    with make_client() as client:
        response = client.get("/definitely-not-a-route")
    assert response.status_code == 404
    assert len(response.content) < 1024
    text = response.text.lower()
    assert "traceback" not in text
    assert "/home/" not in text
    assert "d:\\" not in text
