from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from diamonds_mlops.config import Settings
from diamonds_mlops.data_processing import (
    DataValidationError,
    clean_data,
    generate_sample_data,
    load_raw_data,
    run_data_pipeline,
    validate_data,
)
from diamonds_mlops.features import ENGINEERED_FEATURES, REQUIRED_COLUMNS, add_features


def _diamonds(**overrides: list[object]) -> pd.DataFrame:
    base: dict[str, list[object]] = {
        "carat": [1.0],
        "cut": ["Ideal"],
        "color": ["E"],
        "clarity": ["VS1"],
        "depth": [61.0],
        "table": [57.0],
        "price": [5000],
        "x": [6.0],
        "y": [6.1],
        "z": [3.8],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_validate_data_requires_expected_columns() -> None:
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_data(pd.DataFrame({"carat": [1.0]}))


def test_validate_data_rejects_empty_dataset() -> None:
    with pytest.raises(DataValidationError, match="Dataset is empty"):
        validate_data(pd.DataFrame(columns=REQUIRED_COLUMNS))


def test_validate_data_rejects_non_numeric_columns() -> None:
    with pytest.raises(DataValidationError, match="must be numeric"):
        validate_data(_diamonds(carat=["heavy"]))


def test_validation_error_is_value_error() -> None:
    assert issubclass(DataValidationError, ValueError)


def test_clean_data_removes_invalid_rows() -> None:
    data = pd.DataFrame(
        {
            "carat": [1.0, 0.0, 1.1],
            "cut": ["Ideal", "Good", "Ideal"],
            "color": ["E", "F", "E"],
            "clarity": ["VS1", "SI1", "VS1"],
            "depth": [61.0, 62.0, 100.0],
            "table": [57.0, 58.0, 57.0],
            "price": [5000, 4000, 5100],
            "x": [6.0, 0.0, 6.1],
            "y": [6.1, 6.0, 6.1],
            "z": [3.8, 3.7, 3.8],
        }
    )

    cleaned = clean_data(data)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["carat"] == 1.0


def test_clean_data_can_return_empty_dataframe_for_invalid_input() -> None:
    assert clean_data(_diamonds(carat=[0.0], x=[0.0])).empty


def test_add_features_creates_expected_columns() -> None:
    result = add_features(_diamonds(depth=[60.0], x=[5.0], y=[4.0], z=[3.0]))

    assert result.loc[0, "volume"] == 60.0
    assert result.loc[0, "density"] == pytest.approx(1.0 / 60.001)
    assert result.loc[0, "depth_to_width"] == pytest.approx(60.0 / 5.001)


def test_generate_sample_data_is_deterministic() -> None:
    pd.testing.assert_frame_equal(generate_sample_data(rows=30), generate_sample_data(rows=30))


def test_load_raw_data_can_fail_explicitly_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_raw_data(tmp_path / "missing.csv", create_if_missing=False)


def test_load_raw_data_generates_sample_when_missing(settings: Settings) -> None:
    data = load_raw_data(settings=settings)

    assert settings.raw_data_path.exists()
    assert len(data) == settings.sample_rows


def test_run_data_pipeline_saves_split(settings: Settings) -> None:
    split = run_data_pipeline(settings=settings)

    assert len(split.train) + len(split.test) == settings.sample_rows
    assert set(ENGINEERED_FEATURES) <= set(split.train.columns)
    for name in ("train.csv", "test.csv", "diamonds_processed.csv"):
        assert (settings.processed_dir / name).exists()
