"""Оркестрация полного ML-пайплайна: ETL + обучение + сохранение артефактов."""

from __future__ import annotations

import logging

from diamonds_mlops.config import Settings, get_settings
from diamonds_mlops.data_processing import run_data_pipeline
from diamonds_mlops.model_training import TrainingResult, train_model

logger = logging.getLogger(__name__)


def run_pipeline(settings: Settings | None = None) -> TrainingResult:
    """Запустить ETL и обучение модели, вернуть модель и её метрики."""
    settings = settings or get_settings()
    split = run_data_pipeline(settings=settings)
    result = train_model(split.train, split.test, settings=settings)
    logger.info("Pipeline completed")
    return result
