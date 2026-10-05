from __future__ import annotations

import json
from pathlib import Path

import pytest

from diamonds_mlops.cli import build_parser, main


@pytest.fixture
def project_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("DIAMONDS_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("DIAMONDS_N_ESTIMATORS", "20")
    return tmp_path


def test_pipeline_command_trains_model(
    project_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["pipeline"]) == 0

    metrics = json.loads(capsys.readouterr().out)
    assert set(metrics) == {"rmse", "mae", "r2", "mape"}
    assert (project_root / "models" / "diamond_price_model.joblib").exists()
    assert (project_root / "reports" / "monitoring" / "baseline.json").exists()


def test_monitor_command_prints_metrics(
    project_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["monitor"]) == 0

    assert set(json.loads(capsys.readouterr().out)) == {
        "cpu_percent",
        "ram_percent",
        "disk_percent",
    }


def test_figures_command_creates_images(
    project_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("matplotlib")

    assert main(["figures"]) == 0

    paths = [Path(path) for path in json.loads(capsys.readouterr().out)]
    assert len(paths) == 3
    assert all(
        path.exists() and path.parent == project_root / "reports" / "figures" for path in paths
    )


def test_command_is_required() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
