"""Общие фикстуры: изолированные настройки, синтетические данные и обученная модель."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from diamonds_mlops.config import Settings, get_settings
from diamonds_mlops.data_processing import DataSplit, generate_sample_data
from diamonds_mlops.features import add_features
from diamonds_mlops.model_training import TrainingResult, train_model


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> Iterator[None]:
    """Сбрасывать кеш настроек, чтобы тесты с переменными окружения не влияли друг на друга."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Настройки, у которых все артефакты лежат во временном каталоге теста."""
    return Settings(
        _env_file=None,
        project_root=tmp_path,
        sample_rows=120,
        n_estimators=20,
    )


@pytest.fixture
def featured_data() -> pd.DataFrame:
    """Небольшой синтетический датасет с производными признаками."""
    return add_features(generate_sample_data(rows=80))


@pytest.fixture
def data_split(featured_data: pd.DataFrame) -> DataSplit:
    """Детерминированное разбиение 60/20."""
    return DataSplit(
        train=featured_data.iloc[:60].reset_index(drop=True),
        test=featured_data.iloc[60:].reset_index(drop=True),
    )


@pytest.fixture
def trained(data_split: DataSplit, settings: Settings) -> TrainingResult:
    """Модель, обученная на синтетике и сохранённая в каталог из ``settings``."""
    return train_model(data_split.train, data_split.test, settings=settings)


@pytest.fixture
def diamond_payload() -> dict[str, float | str]:
    """Корректное тело запроса к ``/predict``."""
    return {
        "carat": 1.0,
        "cut": "Ideal",
        "color": "E",
        "clarity": "VS1",
        "depth": 61.0,
        "table": 57.0,
        "x": 6.0,
        "y": 6.1,
        "z": 3.8,
    }
