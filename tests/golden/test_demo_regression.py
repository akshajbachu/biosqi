from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from biosqi.analysis import analyze
from biosqi.io import read_recording
from biosqi.types import SignalType

ROOT = Path(__file__).resolve().parents[2]


def test_demo_recording_matches_release_model_snapshot() -> None:
    expected = json.loads((Path(__file__).with_name("demo_expected.json")).read_text())
    recording = read_recording(
        ROOT / "examples" / "demo_ecg.npz",
        signal_type=SignalType.ECG,
    )
    result = analyze(recording, threshold=expected["threshold"])

    assert result.model_name == expected["model"]
    assert [window.start for window in result.windows] == expected["starts"]
    assert [window.usable for window in result.windows] == expected["usable"]
    np.testing.assert_allclose(result.scores, expected["scores"], atol=1e-8, rtol=0)
