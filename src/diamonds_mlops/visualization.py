"""Графики для отчёта: распределение цены, predicted vs actual, важность признаков.

Требует группу зависимостей ``viz`` (matplotlib), которая не нужна API в production.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")  # без GUI: графики строятся на сервере и в CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from diamonds_mlops.config import Settings, get_settings
from diamonds_mlops.features import FEATURE_COLUMNS, TARGET
from diamonds_mlops.model_training import load_model, load_processed_data, train_model

logger = logging.getLogger(__name__)

DPI = 150
PRICE_COLOR = "#2f80ed"
SCATTER_COLOR = "#27ae60"
IDEAL_LINE_COLOR = "#eb5757"
IMPORTANCE_COLOR = "#9b51e0"


def plot_price_distribution(data: pd.DataFrame, output_path: Path) -> None:
    """Построить гистограмму распределения цены."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(data[TARGET], bins=30, color=PRICE_COLOR, edgecolor="white")
    ax.set_title("Diamond Price Distribution")
    ax.set_xlabel("Price")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)


def plot_predicted_vs_actual(test_df: pd.DataFrame, model: Pipeline, output_path: Path) -> None:
    """Построить scatter фактической и предсказанной цены с линией идеального прогноза."""
    actual = test_df[TARGET].to_numpy()
    predicted = model.predict(test_df[FEATURE_COLUMNS])
    min_value = float(min(actual.min(), predicted.min()))
    max_value = float(max(actual.max(), predicted.max()))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        actual, predicted, alpha=0.75, color=SCATTER_COLOR, edgecolor="white", linewidth=0.4
    )
    ax.plot(
        [min_value, max_value],
        [min_value, max_value],
        color=IDEAL_LINE_COLOR,
        linestyle="--",
        label="Ideal prediction",
    )
    ax.set_title("Predicted vs Actual Price")
    ax.set_xlabel("Actual price")
    ax.set_ylabel("Predicted price")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)


def plot_feature_importance(model: Pipeline, output_path: Path, top_n: int = 12) -> None:
    """Построить топ-``top_n`` признаков по важности RandomForest."""
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = model.named_steps["model"].feature_importances_

    order = np.argsort(importances)[-top_n:]
    names = [
        str(feature_names[index]).removeprefix("numeric__").removeprefix("categorical__")
        for index in order
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(names, importances[order], color=IMPORTANCE_COLOR)
    ax.set_title("RandomForest Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)


def generate_report_figures(
    output_dir: Path | str | None = None,
    *,
    settings: Settings | None = None,
) -> list[Path]:
    """Сгенерировать все графики отчёта; при отсутствии модели обучить её."""
    settings = settings or get_settings()
    output_dir = Path(output_dir) if output_dir is not None else settings.figures_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    split = load_processed_data(settings=settings)
    model = load_model(settings.model_path)
    if model is None:
        model = train_model(split.train, split.test, settings=settings).model

    outputs = {
        "price_distribution": output_dir / "price_distribution.png",
        "predicted_vs_actual": output_dir / "predicted_vs_actual.png",
        "feature_importance": output_dir / "feature_importance.png",
    }
    plot_price_distribution(
        pd.concat([split.train, split.test], ignore_index=True), outputs["price_distribution"]
    )
    plot_predicted_vs_actual(split.test, model, outputs["predicted_vs_actual"])
    plot_feature_importance(model, outputs["feature_importance"])

    for path in outputs.values():
        logger.info("Saved figure %s", path)
    return list(outputs.values())
