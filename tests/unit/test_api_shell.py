"""API shell smoke tests."""

from fastapi.testclient import TestClient

from services.api.app import create_app


def make_client(tmp_path) -> TestClient:
    # Point the app at an empty models dir: no current.json -> honest 503s.
    return TestClient(create_app(models_dir=tmp_path))


def test_health_is_alive_without_model(tmp_path) -> None:
    with make_client(tmp_path) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"]


def test_ready_is_503_without_model(tmp_path) -> None:
    with make_client(tmp_path) as client:
        resp = client.get("/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["ready"] is False
    assert body["model_loaded"] is False
    assert "current.json missing" in body["detail"]


def test_model_info_503_without_model(tmp_path) -> None:
    with make_client(tmp_path) as client:
        resp = client.get("/v1/model")
    assert resp.status_code == 503
    assert resp.json()["model_id"] is None


def test_prediction_503_without_model(tmp_path) -> None:
    with make_client(tmp_path) as client:
        resp = client.post("/v1/predictions")
    assert resp.status_code == 503
    assert resp.json()["error_code"] == "model_unavailable"


def test_unknown_route_is_bounded_404(tmp_path) -> None:
    with make_client(tmp_path) as client:
        resp = client.get("/definitely-not-a-route")
    assert resp.status_code == 404
    text = resp.text.lower()
    assert "traceback" not in text
    assert "/home/" not in text
