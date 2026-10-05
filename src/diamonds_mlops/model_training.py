"""Обучение, оценка, сохранение и загрузка модели предсказания цены."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import NamedTuple

import joblib
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from diamonds_mlops.config import Settings, get_settings
from diamonds_mlops.data_processing import TEST_FILE, TRAIN_FILE, DataSplit, run_data_pipeline
from diamonds_mlops.features import (
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET,
)
from diamonds_mlops.monitoring import save_baseline

logger = logging.getLogger(__name__)

# Защита MAPE от деления на ноль.
_MAPE_EPSILON = 1e-8


class TrainingResult(NamedTuple):
    """Обученный пайплайн и его метрики на тестовой выборке."""

    model: Pipeline
    metrics: dict[str, float]


def build_model(settings: Settings | None = None) -> Pipeline:
    """Собрать sklearn-пайплайн: препроцессинг признаков + RandomForestRegressor."""
    settings = settings or get_settings()
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES + ENGINEERED_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    regressor = RandomForestRegressor(
        n_estimators=settings.n_estimators,
        min_samples_leaf=settings.min_samples_leaf,
        random_state=settings.random_state,
        n_jobs=settings.n_jobs,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", regressor)])


def calculate_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Посчитать RMSE, MAE, R2 и MAPE (в процентах)."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    denominator = np.where(actual == 0, _MAPE_EPSILON, actual)
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
        "mape": float(np.mean(np.abs((actual - predicted) / denominator)) * 100),
    }


def load_processed_data(
    processed_dir: Path | str | None = None,
    *,
    settings: Settings | None = None,
) -> DataSplit:
    """Загрузить train/test после ETL; при их отсутствии запустить ETL."""
    settings = settings or get_settings()
    processed_dir = Path(processed_dir) if processed_dir is not None else settings.processed_dir
    train_path = processed_dir / TRAIN_FILE
    test_path = processed_dir / TEST_FILE
    if not train_path.exists() or not test_path.exists():
        logger.info("Processed data not found in %s, running ETL", processed_dir)
        return run_data_pipeline(output_dir=processed_dir, settings=settings)
    return DataSplit(train=pd.read_csv(train_path), test=pd.read_csv(test_path))


def save_model(model: Pipeline, path: Path | str) -> Path:
    """Сериализовать пайплайн на диск."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved model to %s", path)
    return path


def load_model(path: Path | str | None = None) -> Pipeline | None:
    """Загрузить обученный пайплайн или вернуть ``None``, если артефакта нет."""
    path = Path(path) if path is not None else get_settings().model_path
    if not path.exists():
        logger.warning("Model artifact not found: %s", path)
        return None
    model: Pipeline = joblib.load(path)
    logger.info("Loaded model from %s", path)
    return model


def train_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_path: Path | str | None = None,
    metrics_path: Path | str | None = None,
    baseline_path: Path | str | None = None,
    *,
    settings: Settings | None = None,
) -> TrainingResult:
    """Обучить модель, оценить её и сохранить модель, метрики и baseline.

    Args:
        train_df: Обучающая выборка с признаками и целевой переменной.
        test_df: Тестовая выборка с признаками и целевой переменной.
        model_path: Куда сохранить модель (по умолчанию из настроек).
        metrics_path: Куда сохранить метрики (по умолчанию из настроек).
        baseline_path: Куда сохранить baseline мониторинга (по умолчанию из настроек).
        settings: Настройки проекта.

    Returns:
        Обученный пайплайн и метрики на тестовой выборке.
    """
    settings = settings or get_settings()
    model = build_model(settings)

    logger.info("Training model on %d rows", len(train_df))
    model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET])
    predictions = model.predict(test_df[FEATURE_COLUMNS])
    metrics = calculate_metrics(test_df[TARGET], predictions)
    logger.info("Test metrics: %s", {name: round(value, 4) for name, value in metrics.items()})

    save_model(model, model_path if model_path is not None else settings.model_path)

    metrics_file = Path(metrics_path) if metrics_path is not None else settings.metrics_path
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    save_baseline(
        train_df[FEATURE_COLUMNS],
        metrics,
        baseline_path if baseline_path is not None else settings.baseline_path,
    )
    return TrainingResult(model=model, metrics=metrics)
