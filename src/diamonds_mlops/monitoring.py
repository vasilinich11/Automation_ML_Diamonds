"""Лёгкий мониторинг модели: baseline, дрейф данных, деградация метрик и ресурсы хоста."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
import psutil

from diamonds_mlops.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Порог, ниже которого базовое среднее считается нулевым.
_ZERO_MEAN_TOLERANCE = 1e-8


class FeatureStats(TypedDict):
    """Статистики числового признака."""

    mean: float
    std: float
    min: float
    max: float


class FeatureDrift(TypedDict):
    """Изменение среднего значения признака относительно baseline."""

    baseline_mean: float
    current_mean: float
    relative_change: float


class DriftReport(TypedDict):
    """Результат проверки дрейфа данных."""

    drift_detected: bool
    drifted_features: dict[str, FeatureDrift]
    checked_features: list[str]


class DegradationReport(TypedDict):
    """Результат проверки деградации качества модели."""

    degradation_detected: bool
    rmse_degraded: bool
    r2_degraded: bool


class InfrastructureMetrics(TypedDict):
    """Загрузка ресурсов хоста в процентах."""

    cpu_percent: float
    ram_percent: float
    disk_percent: float


def numeric_profile(data: pd.DataFrame) -> dict[str, FeatureStats]:
    """Посчитать mean/std/min/max для всех числовых колонок."""
    numeric_data = data.select_dtypes(include="number")
    return {
        str(column): FeatureStats(
            mean=float(series.mean()),
            std=float(series.std(ddof=0)),
            min=float(series.min()),
            max=float(series.max()),
        )
        for column, series in numeric_data.items()
    }


def save_baseline(
    data: pd.DataFrame,
    metrics: Mapping[str, float],
    path: Path | str,
) -> dict[str, Any]:
    """Сохранить метрики модели и профиль признаков как baseline для мониторинга."""
    baseline: dict[str, Any] = {
        "metrics": dict(metrics),
        "numeric_profile": numeric_profile(data),
        "rows": len(data),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    logger.info("Saved monitoring baseline to %s", path)
    return baseline


def load_baseline(path: Path | str) -> dict[str, Any] | None:
    """Загрузить сохранённый baseline или вернуть ``None``, если файла нет."""
    path = Path(path)
    if not path.exists():
        return None
    baseline: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return baseline


def detect_data_drift(
    current_data: pd.DataFrame,
    baseline: Mapping[str, Any],
    relative_threshold: float | None = None,
) -> DriftReport:
    """Сравнить средние значения числовых признаков с baseline.

    Признак считается дрейфующим, если относительное изменение среднего
    превышает ``relative_threshold``.
    """
    threshold = (
        relative_threshold if relative_threshold is not None else get_settings().drift_threshold
    )
    current_profile = numeric_profile(current_data)
    baseline_profile: Mapping[str, Mapping[str, float]] = baseline.get("numeric_profile", {})
    drifted_features: dict[str, FeatureDrift] = {}

    for column, current_stats in current_profile.items():
        base_stats = baseline_profile.get(column)
        if not base_stats:
            continue
        base_mean = float(base_stats["mean"])
        current_mean = current_stats["mean"]
        denominator = abs(base_mean) if abs(base_mean) > _ZERO_MEAN_TOLERANCE else 1.0
        relative_change = abs(current_mean - base_mean) / denominator
        if relative_change > threshold:
            drifted_features[column] = FeatureDrift(
                baseline_mean=base_mean,
                current_mean=current_mean,
                relative_change=float(relative_change),
            )

    if drifted_features:
        logger.warning("Data drift detected in features: %s", sorted(drifted_features))
    return DriftReport(
        drift_detected=bool(drifted_features),
        drifted_features=drifted_features,
        checked_features=sorted(current_profile),
    )


def detect_degradation(
    current_metrics: Mapping[str, float],
    baseline_metrics: Mapping[str, float],
    rmse_threshold: float | None = None,
    r2_drop_threshold: float | None = None,
    *,
    settings: Settings | None = None,
) -> DegradationReport:
    """Проверить рост RMSE и падение R2 относительно baseline."""
    settings = settings or get_settings()
    rmse_limit = (
        rmse_threshold if rmse_threshold is not None else settings.rmse_degradation_threshold
    )
    r2_limit = r2_drop_threshold if r2_drop_threshold is not None else settings.r2_drop_threshold

    base_rmse = baseline_metrics.get("rmse")
    current_rmse = current_metrics.get("rmse")
    base_r2 = baseline_metrics.get("r2")
    current_r2 = current_metrics.get("r2")

    rmse_degraded = bool(
        base_rmse
        and current_rmse is not None
        and (current_rmse - base_rmse) / base_rmse > rmse_limit
    )
    r2_degraded = bool(
        base_r2 is not None and current_r2 is not None and base_r2 - current_r2 > r2_limit
    )
    return DegradationReport(
        degradation_detected=rmse_degraded or r2_degraded,
        rmse_degraded=rmse_degraded,
        r2_degraded=r2_degraded,
    )


def get_infrastructure_metrics() -> InfrastructureMetrics:
    """Вернуть текущую загрузку CPU, RAM и диска в процентах.

    ``cpu_percent`` вызывается с коротким интервалом, чтобы не получать
    устаревшее значение ``0.0`` при первом вызове.
    """
    return InfrastructureMetrics(
        cpu_percent=float(psutil.cpu_percent(interval=0.1)),
        ram_percent=float(psutil.virtual_memory().percent),
        disk_percent=float(psutil.disk_usage("/").percent),
    )
