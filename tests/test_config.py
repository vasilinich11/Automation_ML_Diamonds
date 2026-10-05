from __future__ import annotations

from pathlib import Path

import pytest

from diamonds_mlops.config import Settings, get_settings


def test_relative_paths_are_resolved_from_project_root(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, project_root=tmp_path)

    assert settings.model_path == tmp_path / "models" / "diamond_price_model.joblib"
    assert settings.processed_dir == tmp_path / "data" / "processed"


def test_absolute_paths_are_kept(tmp_path: Path) -> None:
    model_file = tmp_path / "elsewhere" / "model.joblib"
    settings = Settings(_env_file=None, model_file=model_file)

    assert settings.model_path == model_file


def test_settings_are_read_from_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DIAMONDS_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("DIAMONDS_N_ESTIMATORS", "7")

    settings = get_settings()

    assert settings.project_root == tmp_path
    assert settings.n_estimators == 7


def test_invalid_settings_are_rejected() -> None:
    with pytest.raises(ValueError, match="test_size"):
        Settings(_env_file=None, test_size=1.5)
