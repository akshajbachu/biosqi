from __future__ import annotations

import numpy as np
import pytest

from biosqi.profiles import get_profile
from biosqi.types import Recording, SignalType
from biosqi.windowing import iter_windows, seconds_to_samples, window_bounds


def test_recording_normalizes_one_dimensional_input() -> None:
    recording = Recording(np.arange(20), 10.0, signal_type=SignalType.ECG)
    assert recording.data.shape == (1, 20)
    assert recording.channel_names == ("ch0",)
    assert recording.duration == 2.0


@pytest.mark.parametrize("sample_rate", [0.0, -1.0, np.inf, np.nan])
def test_recording_rejects_invalid_rate(sample_rate: float) -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        Recording(np.arange(5), sample_rate)


def test_recording_checks_channel_names() -> None:
    with pytest.raises(ValueError, match="channel_names"):
        Recording(np.zeros((2, 5)), 10.0, ("only-one",))
    with pytest.raises(ValueError, match="at least one channel"):
        Recording(np.empty((0, 5)), 10.0)


def test_window_bounds_are_half_open_and_deterministic() -> None:
    assert window_bounds(25, 10.0, 1.0, 0.5) == (
        (0, 10),
        (5, 15),
        (10, 20),
        (15, 25),
    )
    assert window_bounds(5, 10.0, 1.0, 0.5, include_partial=False) == ()
    assert window_bounds(0, 10.0, 1.0, 0.5) == ()


def test_iter_windows_preserves_channel_major_views() -> None:
    recording = Recording(np.arange(20).reshape(2, 10), 2.0)
    windows = list(iter_windows(recording, 2.0, 2.0))
    assert [(window.start, window.stop) for window in windows] == [(0, 4), (4, 8), (8, 10)]
    np.testing.assert_array_equal(windows[1].data, recording.data[:, 4:8])


def test_seconds_to_samples_uses_half_up_rounding() -> None:
    assert seconds_to_samples(0.25, 10.0) == 3
    with pytest.raises(ValueError):
        seconds_to_samples(0.001, 10.0)


def test_profiles_accept_strings_and_reject_unknowns() -> None:
    assert get_profile("ECG").signal_type is SignalType.ECG
    with pytest.raises(ValueError, match="unsupported"):
        get_profile("magnetocardiogram")
