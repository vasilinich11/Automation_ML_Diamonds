from __future__ import annotations

from pathlib import Path

import pandas as pd

from diamonds_mlops.monitoring import (
    detect_data_drift,
    detect_degradation,
    get_infrastructure_metrics,
    load_baseline,
    save_baseline,
)


def test_drift_detection_flags_large_mean_shift(
    featured_data: pd.DataFrame, tmp_path: Path
) -> None:
    current_data = featured_data.copy()
    current_data["carat"] = current_data["carat"] * 2
    baseline = save_baseline(featured_data, {"rmse": 100.0, "r2": 0.9}, tmp_path / "baseline.json")

    result = detect_data_drift(current_data, baseline, relative_threshold=0.1)

    assert result["drift_detected"] is True
    assert "carat" in result["drifted_features"]


def test_drift_detection_returns_false_for_similar_data(
    featured_data: pd.DataFrame, tmp_path: Path
) -> None:
    baseline = save_baseline(featured_data, {"rmse": 100.0, "r2": 0.9}, tmp_path / "baseline.json")

    result = detect_data_drift(featured_data.copy(), baseline, relative_threshold=0.1)

    assert result["drift_detected"] is False
    assert result["drifted_features"] == {}


def test_baseline_roundtrip(featured_data: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    saved = save_baseline(featured_data, {"rmse": 100.0}, path)

    assert load_baseline(path) == saved
    assert load_baseline(tmp_path / "missing.json") is None


def test_degradation_detection_uses_real_metrics() -> None:
    result = detect_degradation(
        {"rmse": 130.0, "r2": 0.75},
        {"rmse": 100.0, "r2": 0.9},
        rmse_threshold=0.1,
        r2_drop_threshold=0.05,
    )

    assert result["degradation_detected"] is True
    assert result["rmse_degraded"] is True
    assert result["r2_degraded"] is True


def test_no_degradation_for_stable_metrics() -> None:
    result = detect_degradation({"rmse": 101.0, "r2": 0.89}, {"rmse": 100.0, "r2": 0.9})

    assert result["degradation_detected"] is False


def test_infrastructure_metrics_have_expected_keys() -> None:
    metrics = get_infrastructure_metrics()

    assert set(metrics) == {"cpu_percent", "ram_percent", "disk_percent"}
    assert all(0.0 <= value <= 100.0 for value in metrics.values())
