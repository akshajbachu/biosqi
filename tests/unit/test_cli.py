from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

import biosqi.cli as cli
from biosqi.cli import app

runner = CliRunner()


def _signal_file(tmp_path: Path) -> Path:
    path = tmp_path / "signal.npy"
    time = np.arange(400) / 40.0
    np.save(path, np.sin(2 * np.pi * time))
    return path


def test_score_json_and_file_output(tmp_path: Path) -> None:
    source = _signal_file(tmp_path)
    result = runner.invoke(app, ["score", str(source), "-r", "40", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["n_samples"] == 400

    output = tmp_path / "score.json"
    result = runner.invoke(app, ["score", str(source), "-r", "40", "-o", str(output)])
    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == "1.0"
    result = runner.invoke(app, ["score", str(source), "-r", "40"])
    assert result.exit_code == 0
    assert "Leading artifact" in result.output


def test_mask_and_html_report_commands(tmp_path: Path) -> None:
    source = _signal_file(tmp_path)
    mask = tmp_path / "mask.npy"
    result = runner.invoke(app, ["mask", str(source), "-r", "40", "-o", str(mask)])
    assert result.exit_code == 0, result.output
    assert np.load(mask).shape == (400,)

    report = tmp_path / "report.html"
    result = runner.invoke(app, ["report", str(source), "-r", "40", "-o", str(report)])
    assert result.exit_code == 0, result.output
    assert report.read_text(encoding="utf-8").startswith("<!doctype html>")

    csv_mask = tmp_path / "mask.csv"
    result = runner.invoke(app, ["mask", str(source), "-r", "40", "-o", str(csv_mask)])
    assert result.exit_code == 0
    assert csv_mask.read_text(encoding="utf-8").startswith("# usable")

    json_report = tmp_path / "report.json"
    result = runner.invoke(app, ["report", str(source), "-r", "40", "-o", str(json_report)])
    assert result.exit_code == 0
    assert json.loads(json_report.read_text(encoding="utf-8"))["schema_version"] == "1.0"


def test_fit_command(tmp_path: Path) -> None:
    training = tmp_path / "training.csv"
    training.write_text(
        "a,b,label\n"
        "0,0,0\n0.1,0.2,0\n0.2,0.1,0\n0.4,0.3,0\n"
        "0.6,0.7,1\n0.8,0.6,1\n0.9,0.9,1\n1,1,1\n",
        encoding="utf-8",
    )
    output = tmp_path / "fitted.json"
    result = runner.invoke(app, ["fit", str(training), "-o", str(output)])
    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text(encoding="utf-8"))["features"] == ["a", "b"]


def test_bad_signal_type_is_reported(tmp_path: Path) -> None:
    source = _signal_file(tmp_path)
    result = runner.invoke(app, ["score", str(source), "-r", "40", "-t", "unknown"])
    assert result.exit_code != 0
    assert "choose one of" in result.output


def test_fit_and_view_errors_are_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training = tmp_path / "bad.csv"
    training.write_text("a,target\n1,0\n", encoding="utf-8")
    result = runner.invoke(app, ["fit", str(training), "-o", str(tmp_path / "model.json")])
    assert result.exit_code != 0
    assert "label" in result.output

    def missing(_module: str) -> None:
        raise ImportError

    monkeypatch.setattr(cli.importlib, "import_module", missing)
    result = runner.invoke(app, ["view"])
    assert result.exit_code != 0
    assert "UI extra" in result.output
