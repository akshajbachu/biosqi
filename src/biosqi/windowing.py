"""Deterministic sample-domain window generation."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from biosqi.types import Recording, SignalWindow


def seconds_to_samples(seconds: float, sample_rate: float, *, name: str = "duration") -> int:
    """Convert seconds to samples using stable half-up rounding."""

    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample_rate must be finite and positive")
    samples = int(np.floor(seconds * sample_rate + 0.5))
    if samples < 1:
        raise ValueError(f"{name} is shorter than one sample")
    return samples


def window_bounds(
    n_samples: int,
    sample_rate: float,
    window_seconds: float,
    step_seconds: float,
    *,
    include_partial: bool = True,
) -> tuple[tuple[int, int], ...]:
    """Return deterministic half-open bounds, without padding or duplicated tails."""

    if n_samples < 0:
        raise ValueError("n_samples cannot be negative")
    size = seconds_to_samples(window_seconds, sample_rate, name="window_seconds")
    step = seconds_to_samples(step_seconds, sample_rate, name="step_seconds")
    if n_samples == 0:
        return ()
    if n_samples <= size:
        return ((0, n_samples),) if include_partial or n_samples == size else ()

    starts = list(range(0, n_samples - size + 1, step))
    bounds = [(start, start + size) for start in starts]
    next_start = (starts[-1] + step) if starts else 0
    if include_partial and bounds[-1][1] < n_samples and next_start < n_samples:
        bounds.append((next_start, n_samples))
    return tuple(bounds)


def iter_windows(
    recording: Recording,
    window_seconds: float,
    step_seconds: float,
    *,
    include_partial: bool = True,
) -> Iterator[SignalWindow]:
    """Yield channel-major views over a recording in ascending sample order."""

    bounds = window_bounds(
        recording.n_samples,
        recording.sample_rate,
        window_seconds,
        step_seconds,
        include_partial=include_partial,
    )
    for index, (start, stop) in enumerate(bounds):
        yield SignalWindow(
            index=index,
            start=start,
            stop=stop,
            data=recording.data[:, start:stop],
            sample_rate=recording.sample_rate,
        )
