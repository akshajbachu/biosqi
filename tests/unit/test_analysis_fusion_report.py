from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from biosqi.analysis import analyze
from biosqi.attribution import attribute_artifacts
from biosqi.fusion import FusionModel, fit_fusion_model, load_default_model
from biosqi.report import render_html_report, result_to_dict, result_to_json, write_html_report
from biosqi.types import IndicatorResult, Recording, SignalType


def test_default_model_prediction_and_attribution_are_transparent() -> None:
    model = load_default_model()
    indicators = tuple(
        IndicatorResult(feature, 0.8, 0.8, f"artifact-{index}")
        for index, feature in enumerate(model.features)
    )
    prediction = model.predict(indicators)
    artifacts = attribute_artifacts(indicators, model)
    assert 0.0 <= prediction.raw_score <= 1.0
    assert 0.0 <= prediction.calibrated_score <= 1.0
    assert set(prediction.contributions) == set(model.features)
    assert all(value == pytest.approx(0.2) for value in artifacts.values())


def test_model_round_trip_and_validation(tmp_path: Path) -> None:
    model = load_default_model()
    path = tmp_path / "model.json"
    model.save(path)
    assert FusionModel.load(path) == model
    with pytest.raises(ValueError, match="equally sized"):
        FusionModel("bad", ("a",), (1.0, 2.0), 0.0)
    with pytest.raises(ValueError, match="equal length"):
        FusionModel("bad", ("a",), (1.0,), 0.0, (0.0, 1.0), (0.0,))
    with pytest.raises(ValueError, match="calibration_x"):
        FusionModel("bad", ("a",), (1.0,), 0.0, (1.0, 0.0), (0.0, 1.0))
    with pytest.raises(ValueError, match="calibration_y"):
        FusionModel("bad", ("a",), (1.0,), 0.0, (0.0, 1.0), (1.0, 0.0))

    negative = FusionModel("negative", ("a",), (1.0,), -10.0)
    prediction = negative.predict(())
    assert prediction.raw_score < 0.01


def test_fit_fusion_model_learns_and_calibrates() -> None:
    rng = np.random.default_rng(8)
    x = rng.normal(size=(100, 2))
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(int)
    model = fit_fusion_model(x, y, ("one", "two"))
    assert model.weights[0] > 0
    assert len(model.calibration_x) >= 2
    with pytest.raises(ValueError, match="both binary classes"):
        fit_fusion_model(x, np.zeros(100), ("one", "two"))
    with pytest.raises(ValueError, match="2D array"):
        fit_fusion_model(x[:, 0], y, ("one",))
    with pytest.raises(ValueError, match="finite"):
        nonfinite = x.copy()
        nonfinite[0, 0] = np.nan
        fit_fusion_model(nonfinite, y, ("one", "two"))


def test_multichannel_analysis_and_reports(tmp_path: Path) -> None:
    sample_rate = 100.0
    time = np.arange(1200) / sample_rate
    good = np.sin(2 * np.pi * 1.2 * time)
    poor = np.zeros_like(good)
    recording = Recording(
        np.vstack((good, poor)),
        sample_rate,
        ("pulse", "disconnected"),
        SignalType.PPG,
        "patient<unsafe>.npy",
    )
    result = analyze(recording, window_seconds=5.0, step_seconds=5.0)
    assert len(result.windows) == 3
    assert all(len(window.channels) == 2 for window in result.windows)
    assert result.scores.shape == (3,)
    assert result.mask.dtype == np.bool_
    payload = result_to_dict(result)
    assert payload["schema_version"] == "1.0"
    assert payload["summary"]["window_count"] == 3
    assert json.loads(result_to_json(result))["signal_type"] == "ppg"
    document = render_html_report(result)
    assert "<!doctype html>" in document
    assert "patient&lt;unsafe&gt;.npy" in document
    assert "https://" not in document
    output = tmp_path / "report.html"
    assert write_html_report(result, output) == output
    assert output.read_text(encoding="utf-8") == document


def test_empty_and_invalid_analysis_options() -> None:
    empty = Recording(np.empty((1, 0)), 100.0)
    result = analyze(empty)
    assert result.windows == ()
    assert "no samples" in result.warnings[-1]
    with pytest.raises(ValueError, match="threshold"):
        analyze(empty, threshold=1.1)
