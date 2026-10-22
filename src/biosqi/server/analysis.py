"""Adapt the typed biosqi core result to the dashboard response schema."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
from scipy import signal as scipy_signal

from biosqi.analysis import analyze
from biosqi.fusion import load_default_model
from biosqi.types import FloatArray, Recording, SignalType, WindowResult

ARTIFACT_LABELS = {
    "broadband_noise": "Broadband noise / out-of-band power",
    "spectral_disorder": "Unusual spectral distribution",
    "line_noise": "Mains interference",
    "baseline_wander": "Baseline wander / motion",
    "amplitude_failure": "Clipping, flatline, or unusual amplitude",
    "morphology_outlier": "Unusual waveform distribution",
    "aperiodicity": "Poor periodic consistency",
    "beat_inconsistency": "Beat disagreement",
}
ARTIFACT_COLORS = {
    "broadband_noise": "#567b8d",
    "spectral_disorder": "#7667a8",
    "line_noise": "#8f6aa8",
    "baseline_wander": "#c48636",
    "amplitude_failure": "#ba463a",
    "morphology_outlier": "#9a645c",
    "aperiodicity": "#3d7e6d",
    "beat_inconsistency": "#4f6d7a",
}


def core_available() -> bool:
    """Return true when the dashboard and numerical core are installed together."""
    return True


def _quality(score: float, threshold: float) -> str:
    if score >= threshold:
        return "good"
    if score < max(0.0, threshold - 0.15):
        return "poor"
    return "borderline"


def _downsample(
    samples: FloatArray,
    sample_rate: float,
    limit: int = 5000,
) -> list[dict[str, float]]:
    stride = max(1, math.ceil(samples.size / limit))
    indexes = np.arange(0, samples.size, stride)
    return [{"x": float(index / sample_rate), "y": float(samples[index])} for index in indexes]


def _psd(samples: FloatArray, sample_rate: float) -> list[dict[str, float]]:
    finite = samples[np.isfinite(samples)]
    if finite.size < 4:
        return []
    frequencies, powers = scipy_signal.welch(
        finite,
        fs=sample_rate,
        nperseg=min(finite.size, max(16, int(sample_rate * 4))),
    )
    valid = frequencies <= min(sample_rate / 2.0, 100.0)
    frequencies, powers = frequencies[valid], powers[valid]
    stride = max(1, math.ceil(frequencies.size / 300))
    return [
        {"x": float(frequencies[index]), "y": float(powers[index])}
        for index in range(0, frequencies.size, stride)
    ]


def _indicator_payload(window: WindowResult, threshold: float) -> list[dict[str, Any]]:
    channel = window.channels[0]
    weights = dict(zip(load_default_model().features, load_default_model().weights, strict=True))
    payload: list[dict[str, Any]] = []
    for indicator in channel.indicators:
        quality = 0.5 if indicator.quality is None else indicator.quality
        weight = float(weights.get(indicator.name, 0.0))
        payload.append(
            {
                "id": indicator.name,
                "label": indicator.name.replace("_", " ").title(),
                "value": indicator.value,
                "normalized": quality,
                "weight": weight,
                "contribution": weight * (quality - 0.5),
                "status": "unavailable" if not indicator.valid else _quality(quality, threshold),
                "description": indicator.reason
                or ARTIFACT_LABELS.get(indicator.artifact, indicator.artifact),
                "evidence": indicator.evidence,
            }
        )
    return payload


def _artifact_payload(window: WindowResult) -> list[dict[str, Any]]:
    significant = [
        (artifact, severity) for artifact, severity in window.artifacts.items() if severity >= 0.05
    ]
    total = sum(severity for _, severity in significant) or 1.0
    return [
        {
            "id": artifact,
            "label": ARTIFACT_LABELS.get(artifact, artifact.replace("_", " ").title()),
            "contribution": float(severity / total),
            "confidence": float(severity),
            "color": ARTIFACT_COLORS.get(artifact, "#5c788e"),
        }
        for artifact, severity in significant
    ]


def analyze_signal(
    samples: FloatArray,
    *,
    filename: str,
    sample_rate: float,
    profile: str,
    window_seconds: float,
    overlap: float,
    threshold: float,
    channel: str = "Channel 1",
) -> dict[str, Any]:
    """Analyze one uploaded channel using the same core used by Python and CLI."""
    raw = np.asarray(samples, dtype=np.float64).reshape(-1)
    if raw.size < 2:
        raise ValueError("The recording must contain at least two samples.")
    try:
        signal_type = SignalType(profile.lower())
    except ValueError as error:
        supported = ", ".join(item.value for item in SignalType)
        raise ValueError(f"Unknown profile {profile!r}; choose from {supported}.") from error

    finite = np.isfinite(raw)
    display = raw.copy()
    warnings: list[str] = []
    if not finite.all():
        missing = int((~finite).sum())
        valid_indexes = np.flatnonzero(finite)
        if not valid_indexes.size:
            raise ValueError("The recording does not contain any finite samples.")
        display = np.interp(np.arange(raw.size), valid_indexes, raw[valid_indexes])
        warnings.append(
            f"{missing} non-finite samples were interpolated for visualization only; "
            "quality indicators used the original values."
        )

    recording = Recording(
        data=raw,
        sample_rate=sample_rate,
        channel_names=(channel,),
        signal_type=signal_type,
        source=filename,
    )
    result = analyze(
        recording,
        window_seconds=window_seconds,
        step_seconds=window_seconds * (1.0 - overlap),
        threshold=threshold,
        include_partial=True,
    )
    warnings.extend(result.warnings)
    windows: list[dict[str, Any]] = []
    for window in result.windows:
        windows.append(
            {
                "index": window.index,
                "start": window.start_seconds,
                "end": window.stop_seconds,
                "score": window.calibrated_score,
                "raw_score": window.score,
                "quality": _quality(window.calibrated_score, threshold),
                "indicators": _indicator_payload(window, threshold),
                "artifacts": _artifact_payload(window),
                "psd": _psd(display[window.start : window.stop], sample_rate),
            }
        )

    good = sum(item["quality"] == "good" for item in windows)
    poor = sum(item["quality"] == "poor" for item in windows)
    borderline = len(windows) - good - poor
    overall = float(np.mean([item["score"] for item in windows]))
    return {
        "filename": filename,
        "profile": profile,
        "sample_rate": float(sample_rate),
        "channel": channel,
        "model": result.model_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "threshold": float(threshold),
        "waveform": _downsample(display, sample_rate),
        "psd": windows[0]["psd"] if windows else [],
        "windows": windows,
        "summary": {
            "overall_score": overall,
            "usable_percent": float(good / len(windows) * 100.0),
            "good_windows": good,
            "poor_windows": poor,
            "borderline_windows": borderline,
            "total_windows": len(windows),
            "duration_seconds": float(raw.size / sample_rate),
        },
        "warnings": list(dict.fromkeys(warnings)),
    }
