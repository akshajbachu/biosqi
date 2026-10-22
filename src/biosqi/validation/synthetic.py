"""Controlled artifact injection for ground-truth validation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class Artifact(str, Enum):
    """Artifact families produced by the synthetic validation harness."""

    BASELINE_WANDER = "baseline_wander"
    LINE_NOISE = "line_noise"
    MOTION = "motion"
    ELECTRODE_POP = "electrode_pop"
    CLIPPING = "clipping"
    DROPOUT = "dropout"


@dataclass(frozen=True)
class Corruption:
    """A corrupted waveform and its exact sample-level ground truth."""

    signal: FloatArray
    artifact: Artifact
    requested_snr_db: float
    measured_snr_db: float
    affected_mask: NDArray[np.bool_]


def _scale_noise(clean: FloatArray, noise: FloatArray, snr_db: float) -> FloatArray:
    signal_power = float(np.mean(np.square(clean - np.mean(clean))))
    noise_power = float(np.mean(np.square(noise)))
    if signal_power <= 0 or noise_power <= 0:
        return np.zeros_like(noise)
    wanted_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    return np.asarray(noise * np.sqrt(wanted_noise_power / noise_power), dtype=np.float64)


def inject_artifact(
    samples: NDArray[np.floating],
    sample_rate: float,
    artifact: Artifact | str,
    *,
    snr_db: float = 6.0,
    start_fraction: float = 0.25,
    duration_fraction: float = 0.5,
    seed: int = 0,
    mains_hz: float = 60.0,
) -> Corruption:
    """Inject one artifact with a known affected interval and measured SNR."""
    clean = np.asarray(samples, dtype=np.float64).reshape(-1)
    if clean.size < 8:
        raise ValueError("Synthetic corruption requires at least eight samples")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    kind = Artifact(artifact)
    start = int(np.clip(start_fraction, 0.0, 0.95) * clean.size)
    length = max(2, int(np.clip(duration_fraction, 0.01, 1.0) * clean.size))
    stop = min(clean.size, start + length)
    mask = np.zeros(clean.size, dtype=bool)
    mask[start:stop] = True
    time = np.arange(clean.size, dtype=float) / sample_rate
    rng = np.random.default_rng(seed)
    raw = np.zeros_like(clean)
    corrupted: FloatArray

    if kind is Artifact.BASELINE_WANDER:
        raw[mask] = np.sin(2 * np.pi * 0.25 * time[mask])
    elif kind is Artifact.LINE_NOISE:
        raw[mask] = np.sin(2 * np.pi * mains_hz * time[mask])
    elif kind is Artifact.MOTION:
        motion = rng.normal(size=clean.size)
        kernel = np.ones(max(3, int(sample_rate * 0.08)))
        raw[mask] = np.convolve(motion, kernel / kernel.sum(), mode="same")[mask]
    elif kind is Artifact.ELECTRODE_POP:
        event = min(stop - 1, start + max(1, (stop - start) // 3))
        decay = np.exp(-np.arange(stop - event) / max(1.0, sample_rate * 0.15))
        raw[event:stop] = decay
    elif kind is Artifact.CLIPPING:
        scaled = _scale_noise(clean[mask], rng.normal(size=mask.sum()), snr_db)
        preliminary = clean.copy()
        preliminary[mask] += scaled
        limit = float(np.quantile(np.abs(clean), 0.80))
        corrupted = preliminary.copy()
        corrupted[mask] = np.clip(preliminary[mask], -limit, limit)
        noise = corrupted - clean
        measured = _measured_snr(clean[mask], noise[mask])
        return Corruption(corrupted, kind, snr_db, measured, mask)
    elif kind is Artifact.DROPOUT:
        corrupted = clean.copy()
        corrupted[mask] = np.median(clean[max(0, start - 10) : start + 1])
        noise = corrupted - clean
        measured = _measured_snr(clean[mask], noise[mask])
        return Corruption(corrupted, kind, snr_db, measured, mask)

    scaled_noise = _scale_noise(clean[mask], raw[mask], snr_db)
    noise = np.zeros_like(clean)
    noise[mask] = scaled_noise
    corrupted = np.asarray(clean + noise, dtype=np.float64)
    measured = _measured_snr(clean[mask], noise[mask])
    return Corruption(corrupted, kind, snr_db, measured, mask)


def _measured_snr(signal: FloatArray, noise: FloatArray) -> float:
    signal_power = float(np.mean(np.square(signal - np.mean(signal))))
    noise_power = float(np.mean(np.square(noise)))
    if noise_power == 0:
        return float("inf")
    return float(10.0 * np.log10(signal_power / noise_power))
