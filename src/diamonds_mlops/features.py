"""Схема данных и feature engineering.

Модуль используется и при обучении, и при inference в API, поэтому
признаки гарантированно считаются одинаково (нет training/serving skew).
"""

from __future__ import annotations

from typing import Final

import pandas as pd

NUMERIC_FEATURES: Final[list[str]] = ["carat", "depth", "table", "x", "y", "z"]
CATEGORICAL_FEATURES: Final[list[str]] = ["cut", "color", "clarity"]
ENGINEERED_FEATURES: Final[list[str]] = ["volume", "density", "depth_to_width"]
TARGET: Final[str] = "price"

FEATURE_COLUMNS: Final[list[str]] = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ENGINEERED_FEATURES
REQUIRED_COLUMNS: Final[list[str]] = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES, TARGET]

# Категории упорядочены от худшей к лучшей (как в Kaggle Diamonds Dataset).
CUT_ORDER: Final[tuple[str, ...]] = ("Fair", "Good", "Very Good", "Premium", "Ideal")
COLOR_ORDER: Final[tuple[str, ...]] = ("J", "I", "H", "G", "F", "E", "D")
CLARITY_ORDER: Final[tuple[str, ...]] = ("I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF")

# Защита от деления на ноль при расчёте производных признаков.
EPSILON: Final[float] = 0.001


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Добавить производные признаки ``volume``, ``density`` и ``depth_to_width``.

    Args:
        data: Датафрейм с колонками ``carat``, ``depth``, ``x``, ``y``, ``z``.

    Returns:
        Копия датафрейма с тремя новыми колонками.
    """
    featured = data.copy()
    featured["volume"] = featured["x"] * featured["y"] * featured["z"]
    featured["density"] = featured["carat"] / (featured["volume"] + EPSILON)
    featured["depth_to_width"] = featured["depth"] / (featured["x"] + EPSILON)
    return featured
