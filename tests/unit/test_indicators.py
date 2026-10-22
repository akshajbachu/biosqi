from __future__ import annotations

import numpy as np

from biosqi.indicators import (
    baseline_wander_ratio,
    beat_agreement_template,
    clipping_flatline_rms,
    compute_indicators,
    detect_peaks_energy,
    detect_peaks_prominence,
    distribution_shape,
    line_noise_ratio,
    periodicity_autocorrelation,
    spectral_band_power_ratio,
    spectral_entropy,
)
from biosqi.profiles import get_profile
from biosqi.types import SignalType


def _pulse_train(sample_rate: float = 100.0, seconds: float = 12.0) -> np.ndarray:
    samples = int(sample_rate * seconds)
    impulses = np.zeros(samples)
    impulses[np.arange(int(sample_rate), samples, int(sample_rate))] = 1.0
    kernel_axis = np.arange(-8, 9)
    kernel = np.exp(-0.5 * (kernel_axis / 2.0) ** 2)
    return np.convolve(impulses, kernel, mode="same")


def test_indicators_return_bounded_finite_qualities() -> None:
    sample_rate = 100.0
    x = _pulse_train(sample_rate) + 0.01 * np.random.default_rng(3).normal(size=1200)
    results = compute_indicators(x, sample_rate, get_profile(SignalType.PPG))
    assert len(results) == 8
    assert all(result.quality is None or 0.0 <= result.quality <= 1.0 for result in results)
    assert sum(result.valid for result in results) >= 7


def test_nan_inf_and_short_inputs_do_not_raise() -> None:
    x = np.array([np.nan, np.inf, 1.0, -np.inf])
    results = compute_indicators(x, 100.0, get_profile(SignalType.ECG))
    assert all(not result.valid for result in results)
    assert all(result.reason for result in results)


def test_power_line_tone_is_detected() -> None:
    sample_rate = 250.0
    time = np.arange(2500) / sample_rate
    clean = np.sin(2 * np.pi * 8 * time) + 0.1 * np.sin(2 * np.pi * 48 * time)
    contaminated = clean + 2.0 * np.sin(2 * np.pi * 50 * time)
    clean_result = line_noise_ratio(clean, sample_rate)
    contaminated_result = line_noise_ratio(contaminated, sample_rate)
    assert contaminated_result.value is not None and clean_result.value is not None
    assert contaminated_result.value > clean_result.value
    assert contaminated_result.quality is not None and clean_result.quality is not None
    assert contaminated_result.quality < clean_result.quality


def test_clipping_and_flatline_penalize_quality() -> None:
    clean = np.sin(np.linspace(0, 20, 1000))
    clipped = np.clip(clean, -0.2, 0.2)
    flat = np.ones(1000)
    clean_result = clipping_flatline_rms(clean)
    clipped_result = clipping_flatline_rms(clipped)
    flat_result = clipping_flatline_rms(flat)
    assert clean_result.quality is not None and clipped_result.quality is not None
    assert clean_result.quality > clipped_result.quality
    assert flat_result.quality == 0.0


def test_baseline_wander_ratio_tracks_slow_power() -> None:
    sample_rate = 100.0
    time = np.arange(2000) / sample_rate
    physiological = np.sin(2 * np.pi * 2 * time)
    wandered = physiological + 4 * np.sin(2 * np.pi * 0.2 * time)
    clean = baseline_wander_ratio(physiological, sample_rate, (0.05, 0.5), (0.5, 10))
    poor = baseline_wander_ratio(wandered, sample_rate, (0.05, 0.5), (0.5, 10))
    assert clean.value is not None and poor.value is not None
    assert poor.value > clean.value


def test_periodicity_and_two_peak_detectors() -> None:
    sample_rate = 100.0
    x = _pulse_train(sample_rate)
    periodicity = periodicity_autocorrelation(x, sample_rate, (30, 220))
    first = detect_peaks_prominence(x, sample_rate, SignalType.PPG, (30, 220))
    second = detect_peaks_energy(x, sample_rate, SignalType.PPG, (30, 220))
    beat = beat_agreement_template(x, sample_rate, SignalType.PPG, (30, 220))
    assert periodicity.valid and periodicity.value is not None and periodicity.value > 0.5
    assert 8 <= first.size <= 14
    assert 8 <= second.size <= 14
    assert beat.valid and beat.quality is not None and beat.quality > 0.4


def test_individual_statistics_handle_degenerate_signals() -> None:
    assert not distribution_shape(np.ones(100)).valid
    assert spectral_entropy(np.ones(100), 50).quality is not None
    assert spectral_band_power_ratio(np.ones(100), 50, (1, 10)).quality == 0.0
