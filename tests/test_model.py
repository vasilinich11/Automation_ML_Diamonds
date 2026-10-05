from __future__ import annotations

import pytest

from diamonds_mlops.config import Settings
from diamonds_mlops.data_processing import DataSplit
from diamonds_mlops.features import TARGET
from diamonds_mlops.model_training import (
    TrainingResult,
    calculate_metrics,
    load_model,
    load_processed_data,
)


def test_calculate_metrics_returns_all_required_values() -> None:
    metrics = calculate_metrics([100, 200, 300], [110, 190, 310])

    assert set(metrics) == {"rmse", "mae", "r2", "mape"}
    assert metrics["rmse"] == pytest.approx(10.0)
    assert metrics["mae"] == pytest.approx(10.0)
    assert metrics["r2"] > 0.9


def test_train_model_saves_artifacts(trained: TrainingResult, settings: Settings) -> None:
    assert settings.model_path.exists()
    assert settings.metrics_path.exists()
    assert settings.baseline_path.exists()
    assert trained.metrics["rmse"] > 0


def test_load_model_roundtrip(
    trained: TrainingResult, settings: Settings, data_split: DataSplit
) -> None:
    loaded = load_model(settings.model_path)

    assert loaded is not None
    features = data_split.test.drop(columns=[TARGET])
    assert loaded.predict(features) == pytest.approx(trained.model.predict(features))


def test_load_model_returns_none_when_missing(settings: Settings) -> None:
    assert load_model(settings.model_path) is None


def test_trained_model_handles_unknown_categories(
    trained: TrainingResult, data_split: DataSplit
) -> None:
    sample = data_split.test.iloc[[0]].copy()
    sample["cut"] = "Unknown Cut"

    prediction = trained.model.predict(sample.drop(columns=[TARGET]))

    assert prediction[0] > 0


def test_load_processed_data_runs_etl_when_missing(settings: Settings) -> None:
    split = load_processed_data(settings=settings)

    assert not split.train.empty
    assert not split.test.empty
