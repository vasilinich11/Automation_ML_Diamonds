"""ETL-пайплайн: загрузка, валидация, очистка, feature engineering и train/test split."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Final, NamedTuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from diamonds_mlops.config import Settings, get_settings
from diamonds_mlops.features import (
    CLARITY_ORDER,
    COLOR_ORDER,
    CUT_ORDER,
    NUMERIC_FEATURES,
    REQUIRED_COLUMNS,
    TARGET,
    add_features,
)

logger = logging.getLogger(__name__)

POSITIVE_COLUMNS: Final[list[str]] = ["carat", "price", "x", "y", "z", "depth", "table"]
DEPTH_RANGE: Final[tuple[float, float]] = (40.0, 80.0)
TABLE_RANGE: Final[tuple[float, float]] = (40.0, 90.0)
CUT_PROBABILITIES: Final[tuple[float, ...]] = (0.05, 0.15, 0.25, 0.25, 0.30)

TRAIN_FILE: Final[str] = "train.csv"
TEST_FILE: Final[str] = "test.csv"
FULL_FILE: Final[str] = "diamonds_processed.csv"


class DataValidationError(ValueError):
    """Входные данные не соответствуют ожидаемой схеме."""


class DataSplit(NamedTuple):
    """Результат разбиения данных на обучающую и тестовую выборки."""

    train: pd.DataFrame
    test: pd.DataFrame


def _ordinal_score(values: np.ndarray, order: tuple[str, ...]) -> np.ndarray:
    mapping = {name: idx for idx, name in enumerate(order, start=1)}
    return pd.Series(values).map(mapping).to_numpy()


def generate_sample_data(rows: int = 500, random_state: int = 42) -> pd.DataFrame:
    """Сгенерировать детерминированный diamonds-like датасет для демо и тестов.

    Args:
        rows: Количество строк.
        random_state: Seed генератора случайных чисел.

    Returns:
        Датафрейм с теми же колонками, что и Kaggle Diamonds Dataset.
    """
    rng = np.random.default_rng(random_state)
    carat = rng.uniform(0.2, 2.5, rows).round(2)
    x = (4.0 + carat * 2.2 + rng.normal(0, 0.15, rows)).clip(3.0, None)
    y = (4.0 + carat * 2.1 + rng.normal(0, 0.15, rows)).clip(3.0, None)
    z = (2.4 + carat * 1.3 + rng.normal(0, 0.1, rows)).clip(1.5, None)
    depth = rng.normal(61.5, 2.0, rows).clip(55.0, 70.0)
    table = rng.normal(57.5, 2.5, rows).clip(50.0, 68.0)

    cut = rng.choice(np.array(CUT_ORDER), rows, p=list(CUT_PROBABILITIES))
    color = rng.choice(np.array(COLOR_ORDER), rows)
    clarity = rng.choice(np.array(CLARITY_ORDER), rows)

    price = (
        500
        + carat * 4200
        + _ordinal_score(cut, CUT_ORDER) * 180
        + _ordinal_score(color, COLOR_ORDER) * 130
        + _ordinal_score(clarity, CLARITY_ORDER) * 160
        + rng.normal(0, 350, rows)
    ).clip(300, None)

    return pd.DataFrame(
        {
            "carat": carat,
            "cut": cut,
            "color": color,
            "clarity": clarity,
            "depth": depth.round(2),
            "table": table.round(2),
            "price": price.round(2),
            "x": x.round(2),
            "y": y.round(2),
            "z": z.round(2),
        }
    )


def load_raw_data(
    path: Path | str | None = None,
    *,
    create_if_missing: bool = True,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Загрузить исходный CSV или создать sample dataset, если файла нет.

    Args:
        path: Путь к CSV. По умолчанию берётся из настроек.
        create_if_missing: Сгенерировать и сохранить sample dataset, если файла нет.
        settings: Настройки проекта.

    Raises:
        FileNotFoundError: Файл отсутствует и ``create_if_missing=False``.
    """
    settings = settings or get_settings()
    path = Path(path) if path is not None else settings.raw_data_path
    if path.exists():
        logger.info("Loading raw dataset from %s", path)
        return pd.read_csv(path)
    if not create_if_missing:
        msg = f"Raw dataset not found: {path}"
        raise FileNotFoundError(msg)

    logger.warning("Raw dataset %s not found, generating a sample dataset", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = generate_sample_data(rows=settings.sample_rows, random_state=settings.random_state)
    data.to_csv(path, index=False)
    return data


def validate_data(data: pd.DataFrame, required_columns: Iterable[str] = REQUIRED_COLUMNS) -> None:
    """Проверить наличие обязательных колонок, непустоту и числовые типы.

    Raises:
        DataValidationError: Если данные не проходят проверку.
    """
    missing = set(required_columns) - set(data.columns)
    if missing:
        msg = f"Missing required columns: {sorted(missing)}"
        raise DataValidationError(msg)

    if data.empty:
        msg = "Dataset is empty"
        raise DataValidationError(msg)

    non_numeric = [
        column
        for column in [*NUMERIC_FEATURES, TARGET]
        if not pd.api.types.is_numeric_dtype(data[column])
    ]
    if non_numeric:
        msg = f"Columns must be numeric: {non_numeric}"
        raise DataValidationError(msg)


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Удалить дубликаты, пропуски и физически некорректные измерения."""
    cleaned = data.drop_duplicates().dropna(subset=REQUIRED_COLUMNS)

    is_positive = (cleaned[POSITIVE_COLUMNS] > 0).all(axis=1)
    depth_ok = cleaned["depth"].between(*DEPTH_RANGE)
    table_ok = cleaned["table"].between(*TABLE_RANGE)
    cleaned = cleaned[is_positive & depth_ok & table_ok].reset_index(drop=True)

    dropped = len(data) - len(cleaned)
    if dropped:
        logger.info("Dropped %d invalid rows out of %d", dropped, len(data))
    return cleaned


def split_and_save(
    data: pd.DataFrame,
    output_dir: Path | str | None = None,
    *,
    test_size: float | None = None,
    random_state: int | None = None,
    settings: Settings | None = None,
) -> DataSplit:
    """Разбить данные на train/test и сохранить их в ``output_dir``."""
    settings = settings or get_settings()
    output_dir = Path(output_dir) if output_dir is not None else settings.processed_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test_split(
        data,
        test_size=test_size if test_size is not None else settings.test_size,
        random_state=random_state if random_state is not None else settings.random_state,
    )
    train_df.to_csv(output_dir / TRAIN_FILE, index=False)
    test_df.to_csv(output_dir / TEST_FILE, index=False)
    data.to_csv(output_dir / FULL_FILE, index=False)
    logger.info("Saved processed data to %s: train=%d, test=%d", output_dir, len(train_df), len(test_df))
    return DataSplit(train=train_df, test=test_df)


def run_data_pipeline(
    raw_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    *,
    settings: Settings | None = None,
) -> DataSplit:
    """Выполнить полный ETL: load -> validate -> clean -> features -> split."""
    settings = settings or get_settings()
    raw_data = load_raw_data(raw_path, settings=settings)
    validate_data(raw_data)
    cleaned = clean_data(raw_data)
    if cleaned.empty:
        msg = "No valid rows left after cleaning"
        raise DataValidationError(msg)
    featured = add_features(cleaned)
    return split_and_save(featured, output_dir, settings=settings)

