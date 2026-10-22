from __future__ import annotations

import numpy as np
import pytest

from biosqi.validation.synthetic import Artifact, inject_artifact


@pytest.fixture
def clean_signal() -> np.ndarray:
    time = np.arange(5000) / 250.0
    return np.sin(2 * np.pi * 1.2 * time)


@pytest.mark.parametrize("artifact", list(Artifact))
def test_each_artifact_preserves_shape_and_reports_mask(
    clean_signal: np.ndarray, artifact: Artifact
) -> None:
    result = inject_artifact(clean_signal, 250.0, artifact, snr_db=6.0, seed=4)

    assert result.signal.shape == clean_signal.shape
    assert result.affected_mask.shape == clean_signal.shape
    assert result.affected_mask.dtype == np.bool_
    assert 0 < result.affected_mask.sum() < clean_signal.size
    assert np.all(np.isfinite(result.signal))


@pytest.mark.parametrize(
    "artifact",
    [
        Artifact.BASELINE_WANDER,
        Artifact.LINE_NOISE,
        Artifact.MOTION,
        Artifact.ELECTRODE_POP,
    ],
)
def test_additive_artifacts_hit_requested_snr(clean_signal: np.ndarray, artifact: Artifact) -> None:
    result = inject_artifact(clean_signal, 250.0, artifact, snr_db=3.0, seed=7)
    assert result.measured_snr_db == pytest.approx(3.0, abs=1e-10)


def test_injection_is_deterministic(clean_signal: np.ndarray) -> None:
    first = inject_artifact(clean_signal, 250.0, Artifact.MOTION, seed=42)
    second = inject_artifact(clean_signal, 250.0, Artifact.MOTION, seed=42)
    np.testing.assert_array_equal(first.signal, second.signal)


def test_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="eight"):
        inject_artifact(np.ones(7), 250.0, Artifact.LINE_NOISE)
    with pytest.raises(ValueError, match="positive"):
        inject_artifact(np.ones(20), 0.0, Artifact.LINE_NOISE)
