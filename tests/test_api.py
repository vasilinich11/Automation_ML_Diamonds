from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from diamonds_mlops.api.app import create_app
from diamonds_mlops.config import Settings
from diamonds_mlops.model_training import TrainingResult


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_root_endpoint(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"


def test_health_endpoint_does_not_require_model(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False
    assert set(body["infrastructure"]) == {"cpu_percent", "ram_percent", "disk_percent"}


def test_predict_endpoint_returns_prediction(
    trained: TrainingResult,
    client: TestClient,
    diamond_payload: dict[str, float | str],
) -> None:
    response = client.post("/predict", json=diamond_payload)

    assert response.status_code == 200
    assert response.json()["predicted_price"] > 0
    assert response.json()["message"] == "Prediction completed successfully."


def test_predict_endpoint_returns_503_without_model(
    client: TestClient, diamond_payload: dict[str, float | str]
) -> None:
    response = client.post("/predict", json=diamond_payload)

    assert response.status_code == 503
    assert "Model is not trained yet" in response.json()["detail"]


def test_predict_endpoint_validates_positive_numeric_fields(
    client: TestClient, diamond_payload: dict[str, float | str]
) -> None:
    response = client.post("/predict", json={**diamond_payload, "carat": -1.0})

    assert response.status_code == 422


def test_model_info_without_model(client: TestClient) -> None:
    response = client.get("/model/info")

    assert response.status_code == 200
    body = response.json()
    assert body["model_exists"] is False
    assert body["metrics"] is None
    assert "carat" in body["features"]


def test_model_info_with_trained_model(trained: TrainingResult, client: TestClient) -> None:
    response = client.get("/model/info")

    assert response.status_code == 200
    body = response.json()
    assert body["model_exists"] is True
    assert body["metrics"] == pytest.approx(trained.metrics)
