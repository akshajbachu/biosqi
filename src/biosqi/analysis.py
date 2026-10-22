"""Windowed signal-quality analysis orchestration."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from biosqi.attribution import attribute_artifacts
from biosqi.fusion import FusionModel, load_default_model
from biosqi.indicators import compute_indicators
from biosqi.profiles import SignalProfile, get_profile
from biosqi.types import AnalysisResult, ChannelResult, Recording, WindowResult
from biosqi.windowing import iter_windows


def _aggregate_artifacts(channels: tuple[ChannelResult, ...]) -> dict[str, float]:
    values: defaultdict[str, list[float]] = defaultdict(list)
    for channel in channels:
        for artifact, severity in channel.artifacts.items():
            values[artifact].append(severity)
    return dict(
        sorted(
            ((artifact, float(np.max(severities))) for artifact, severities in values.items()),
            key=lambda item: (-item[1], item[0]),
        )
    )


def analyze(
    recording: Recording,
    *,
    window_seconds: float | None = None,
    step_seconds: float | None = None,
    threshold: float = 0.5,
    model: FusionModel | None = None,
    include_partial: bool = True,
) -> AnalysisResult:
    """Analyze every channel and aggregate each window conservatively by median.

    Scores are calibrated estimates from the selected model, not guarantees of
    clinical usability. The threshold is an application policy and should be
    validated for the intended downstream task.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    profile: SignalProfile = get_profile(recording.signal_type)
    window_length = profile.window_seconds if window_seconds is None else window_seconds
    step_length = profile.step_seconds if step_seconds is None else step_seconds
    fusion = load_default_model() if model is None else model
    warnings: list[str] = []
    window_results: list[WindowResult] = []
    for window in iter_windows(
        recording,
        window_length,
        step_length,
        include_partial=include_partial,
    ):
        channels: list[ChannelResult] = []
        for name, values in zip(recording.channel_names, window.data, strict=True):
            indicators = compute_indicators(values, recording.sample_rate, profile)
            invalid_count = sum(not indicator.valid for indicator in indicators)
            if invalid_count:
                warnings.append(
                    f"window {window.index}, channel {name}: "
                    f"{invalid_count} indicator(s) unavailable"
                )
            prediction = fusion.predict(indicators)
            channels.append(
                ChannelResult(
                    channel=name,
                    score=prediction.raw_score,
                    calibrated_score=prediction.calibrated_score,
                    indicators=indicators,
                    artifacts=attribute_artifacts(indicators, fusion),
                )
            )
        channel_tuple = tuple(channels)
        raw_score = float(np.median([channel.score for channel in channel_tuple]))
        calibrated_score = float(np.median([channel.calibrated_score for channel in channel_tuple]))
        window_results.append(
            WindowResult(
                index=window.index,
                start=window.start,
                stop=window.stop,
                start_seconds=window.start_seconds,
                stop_seconds=window.stop_seconds,
                score=raw_score,
                calibrated_score=calibrated_score,
                usable=calibrated_score >= threshold,
                channels=channel_tuple,
                artifacts=_aggregate_artifacts(channel_tuple),
            )
        )
    if not window_results:
        warnings.append("recording contains no samples")
    return AnalysisResult(
        source=recording.source,
        signal_type=recording.signal_type,
        sample_rate=recording.sample_rate,
        channel_names=recording.channel_names,
        n_samples=recording.n_samples,
        window_seconds=window_length,
        step_seconds=step_length,
        threshold=threshold,
        windows=tuple(window_results),
        model_name=fusion.name,
        warnings=tuple(dict.fromkeys(warnings)),
    )
