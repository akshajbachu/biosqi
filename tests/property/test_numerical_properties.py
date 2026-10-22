from __future__ import annotations

from itertools import pairwise

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from biosqi.fusion import load_default_model
from biosqi.indicators import compute_indicators
from biosqi.profiles import get_profile
from biosqi.types import IndicatorResult, SignalType
from biosqi.windowing import window_bounds


@given(
    n_samples=st.integers(min_value=0, max_value=10_000),
    sample_rate=st.floats(min_value=1, max_value=1000, allow_nan=False, allow_infinity=False),
    window_seconds=st.floats(min_value=0.01, max_value=30, allow_nan=False, allow_infinity=False),
    step_seconds=st.floats(min_value=0.01, max_value=30, allow_nan=False, allow_infinity=False),
)
def test_window_bounds_are_ordered_and_in_range(
    n_samples: int, sample_rate: float, window_seconds: float, step_seconds: float
) -> None:
    try:
        bounds = window_bounds(n_samples, sample_rate, window_seconds, step_seconds)
    except ValueError:
        # Sub-sample durations are intentionally rejected.
        assert window_seconds * sample_rate < 0.5 or step_seconds * sample_rate < 0.5
        return
    assert all(0 <= start < stop <= n_samples for start, stop in bounds)
    assert all(left[0] < right[0] for left, right in pairwise(bounds))
    assert len(bounds) == len(set(bounds))


@given(
    values=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=0,
        max_size=300,
    ),
    sample_rate=st.floats(min_value=10, max_value=500, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=40, deadline=None)
def test_all_indicator_qualities_are_bounded(values: list[float], sample_rate: float) -> None:
    results = compute_indicators(values, sample_rate, get_profile(SignalType.ECG))
    assert len(results) == 8
    for result in results:
        assert result.quality is None or (
            np.isfinite(result.quality) and 0.0 <= result.quality <= 1.0
        )
        assert result.value is None or np.isfinite(result.value)


@given(
    low=st.floats(min_value=0, max_value=0.49, allow_nan=False, allow_infinity=False),
    high=st.floats(min_value=0.51, max_value=1, allow_nan=False, allow_infinity=False),
)
def test_positive_default_weights_make_quality_monotone(low: float, high: float) -> None:
    model = load_default_model()
    low_indicators = tuple(
        IndicatorResult(feature, low, low, "artifact") for feature in model.features
    )
    high_indicators = tuple(
        IndicatorResult(feature, high, high, "artifact") for feature in model.features
    )
    assert (
        model.predict(high_indicators).calibrated_score
        >= model.predict(low_indicators).calibrated_score
    )
