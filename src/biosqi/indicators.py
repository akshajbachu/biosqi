"""Numerically robust signal-quality indicators.

Normalized qualities and artifact thresholds in this module are transparent
engineering heuristics. They are not clinical decision limits and should be
recalibrated on a labelled, task-representative dataset.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from scipy import integrate, signal, stats

from biosqi.profiles import SignalProfile
from biosqi.types import FloatArray, IndicatorResult, SignalType

_EPS = float(np.finfo(np.float64).eps)


def _invalid(name: str, artifact: str, reason: str) -> IndicatorResult:
    return IndicatorResult(name, None, None, artifact, {}, False, reason)


def _prepare(values: npt.ArrayLike, minimum: int = 8) -> tuple[FloatArray | None, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size < minimum:
        return None, 0.0
    finite = np.isfinite(x)
    fraction = float(np.mean(finite))
    if np.count_nonzero(finite) < minimum:
        return None, fraction
    if not np.all(finite):
        positions = np.arange(x.size)
        x = np.interp(positions, positions[finite], x[finite])
    return x, fraction


def _welch(x: FloatArray, sample_rate: float) -> tuple[FloatArray, FloatArray]:
    segment = min(x.size, max(16, round(sample_rate * 4)))
    frequencies, power = signal.welch(
        x,
        fs=sample_rate,
        nperseg=segment,
        detrend="constant",
        scaling="density",
    )
    return frequencies.astype(np.float64), power.astype(np.float64)


def _band_power(frequencies: FloatArray, power: FloatArray, band: tuple[float, float]) -> float:
    low, high = band
    selected = (frequencies >= low) & (frequencies <= high)
    if np.count_nonzero(selected) < 2:
        return 0.0
    return float(integrate.trapezoid(power[selected], frequencies[selected]))


def spectral_band_power_ratio(
    values: npt.ArrayLike, sample_rate: float, band: tuple[float, float]
) -> IndicatorResult:
    """Return in-band/total spectral power.

    The Welch estimator follows Welch, IEEE Trans. Audio Electroacoustics
    15(2), 1967. Band limits are profile defaults, not universal physiology.
    """

    x, finite_fraction = _prepare(values)
    if x is None:
        return _invalid("spectral_power_ratio", "broadband_noise", "too few finite samples")
    frequencies, power = _welch(x, sample_rate)
    nyquist = sample_rate / 2
    effective_band = (max(0.0, band[0]), min(band[1], nyquist))
    total = _band_power(frequencies, power, (max(0.01, frequencies[1]), nyquist))
    ratio = _band_power(frequencies, power, effective_band) / max(total, _EPS)
    quality = float(np.clip(ratio * finite_fraction, 0.0, 1.0))
    return IndicatorResult(
        "spectral_power_ratio",
        ratio,
        quality,
        "broadband_noise",
        {"band_low_hz": effective_band[0], "band_high_hz": effective_band[1]},
    )


def spectral_entropy(values: npt.ArrayLike, sample_rate: float) -> IndicatorResult:
    """Estimate normalized spectral entropy.

    Spectral entropy as an SQI follows the information-theoretic approach used
    by Zhang et al., IEEE Trans. Biomed. Eng. 62(9), 2015. The quality mapping,
    centered on moderate entropy, is heuristic because either extreme can be
    poor depending on signal family.
    """

    x, finite_fraction = _prepare(values)
    if x is None:
        return _invalid("spectral_entropy", "spectral_disorder", "too few finite samples")
    _, power = _welch(x, sample_rate)
    mass = power / max(float(np.sum(power)), _EPS)
    entropy = -float(np.sum(mass * np.log2(mass + _EPS))) / max(np.log2(mass.size), _EPS)
    quality = 1.0 - abs(entropy - 0.6) / 0.6
    return IndicatorResult(
        "spectral_entropy",
        entropy,
        float(np.clip(quality * finite_fraction, 0.0, 1.0)),
        "spectral_disorder",
        {"interpretation": "extreme values are penalized heuristically"},
    )


def line_noise_ratio(values: npt.ArrayLike, sample_rate: float) -> IndicatorResult:
    """Measure 50/60 Hz power relative to nearby high-frequency power.

    Power-line interference assessment is consistent with Clifford et al.,
    Physiol. Meas. 33(9), 2012. The 1 Hz bands and exponential score mapping
    are transparent engineering choices, not validated clinical thresholds.
    """

    x, finite_fraction = _prepare(values)
    if x is None:
        return _invalid("line_noise_ratio", "line_noise", "too few finite samples")
    frequencies, power = _welch(x, sample_rate)
    nyquist = sample_rate / 2
    candidates = [frequency for frequency in (50.0, 60.0) if frequency + 1 <= nyquist]
    if not candidates:
        return _invalid("line_noise_ratio", "line_noise", "sample rate cannot resolve 50/60 Hz")
    ratios: dict[float, float] = {}
    for frequency in candidates:
        tone = _band_power(frequencies, power, (frequency - 1.0, frequency + 1.0))
        neighborhood = _band_power(frequencies, power, (frequency - 5.0, frequency + 5.0))
        ratios[frequency] = tone / max(neighborhood - tone, _EPS)
    mains, ratio = max(ratios.items(), key=lambda item: item[1])
    quality = float(np.exp(-2.0 * ratio) * finite_fraction)
    return IndicatorResult(
        "line_noise_ratio",
        ratio,
        float(np.clip(quality, 0.0, 1.0)),
        "line_noise",
        {"dominant_mains_hz": mains, "ratios": str(ratios)},
    )


def clipping_flatline_rms(values: npt.ArrayLike) -> IndicatorResult:
    """Measure clipping, flat-line occupancy, and RMS amplitude.

    Flat-line and saturation checks are common in Li et al., Physiol. Meas.
    29(1), 2008. Without ADC limits, clipping is inferred from repeated extrema;
    the 1% tolerance and score weights are dataset-agnostic heuristics.
    """

    x, finite_fraction = _prepare(values)
    if x is None:
        return _invalid("clipping_flatline_rms", "amplitude_failure", "too few finite samples")
    span = float(np.ptp(x))
    scale = max(span, float(np.nanstd(x)), _EPS)
    tolerance = max(scale * 1e-6, _EPS)
    at_minimum = np.abs(x - np.min(x)) <= tolerance
    at_maximum = np.abs(x - np.max(x)) <= tolerance
    clipping = float(np.mean(at_minimum | at_maximum))
    differences = np.abs(np.diff(x))
    flatline = float(np.mean(differences <= max(scale * 1e-5, _EPS)))
    rms = float(np.sqrt(np.mean(np.square(x - np.mean(x)))))
    dynamic = 0.0 if span <= _EPS else 1.0
    quality = dynamic * finite_fraction * np.exp(-8.0 * clipping - 12.0 * flatline)
    return IndicatorResult(
        "clipping_flatline_rms",
        rms,
        float(np.clip(quality, 0.0, 1.0)),
        "amplitude_failure",
        {"clipping_fraction": clipping, "flatline_fraction": flatline, "rms": rms},
    )


def baseline_wander_ratio(
    values: npt.ArrayLike,
    sample_rate: float,
    baseline_band: tuple[float, float],
    physiological_band: tuple[float, float],
) -> IndicatorResult:
    """Measure baseline-band power relative to physiological-band power.

    Baseline power ratios are motivated by Clifford et al., Physiol. Meas.
    33(9), 2012. Profile bands and the ratio-to-score transform are heuristic.
    """

    x, finite_fraction = _prepare(values)
    if x is None:
        return _invalid("baseline_wander_ratio", "baseline_wander", "too few finite samples")
    frequencies, power = _welch(x, sample_rate)
    baseline = _band_power(frequencies, power, baseline_band)
    physiological = _band_power(frequencies, power, physiological_band)
    ratio = baseline / max(baseline + physiological, _EPS)
    return IndicatorResult(
        "baseline_wander_ratio",
        ratio,
        float(np.clip((1.0 - ratio) * finite_fraction, 0.0, 1.0)),
        "baseline_wander",
        {"baseline_power": baseline, "physiological_power": physiological},
    )


def distribution_shape(values: npt.ArrayLike) -> IndicatorResult:
    """Return skewness and Pearson kurtosis with a conservative morphology score.

    Orphanidou et al., IEEE J. Biomed. Health Inform. 19(3), 2015 discuss
    statistical ECG quality features. No universal target distribution exists;
    the score only down-weights very extreme shape statistics.
    """

    x, finite_fraction = _prepare(values)
    if x is None:
        return _invalid("distribution_shape", "morphology_outlier", "too few finite samples")
    if float(np.ptp(x)) <= _EPS:
        return _invalid("distribution_shape", "morphology_outlier", "constant or degenerate signal")
    skewness = float(stats.skew(x, bias=False))
    kurtosis = float(stats.kurtosis(x, fisher=False, bias=False))
    if not np.isfinite(skewness) or not np.isfinite(kurtosis):
        return _invalid("distribution_shape", "morphology_outlier", "constant or degenerate signal")
    penalty = max(0.0, abs(skewness) - 2.0) + 0.25 * max(0.0, kurtosis - 12.0)
    quality = np.exp(-penalty) * finite_fraction
    return IndicatorResult(
        "distribution_shape",
        kurtosis,
        float(np.clip(quality, 0.0, 1.0)),
        "morphology_outlier",
        {"skewness": skewness, "pearson_kurtosis": kurtosis},
    )


def periodicity_autocorrelation(
    values: npt.ArrayLike,
    sample_rate: float,
    rate_bpm: tuple[float, float] | None,
) -> IndicatorResult:
    """Find the strongest normalized autocorrelation at plausible beat lags.

    Autocorrelation SQIs follow the periodicity principle in Li et al.,
    Physiol. Meas. 29(1), 2008. Plausible rate bounds come from the selected
    profile and are broad quality-analysis assumptions, not diagnostic limits.
    """

    x, finite_fraction = _prepare(values, minimum=16)
    if x is None:
        return _invalid("periodicity_autocorrelation", "aperiodicity", "too few finite samples")
    x = np.asarray(signal.detrend(x), dtype=np.float64)
    variance = float(np.dot(x, x))
    if variance <= _EPS:
        return _invalid("periodicity_autocorrelation", "aperiodicity", "constant signal")
    correlation = signal.correlate(x, x, mode="full", method="fft")[x.size - 1 :] / variance
    if rate_bpm is None:
        minimum_lag = max(1, round(0.25 * sample_rate))
        maximum_lag = min(x.size - 1, round(2.0 * sample_rate))
    else:
        minimum_lag = max(1, int(np.floor(60.0 * sample_rate / rate_bpm[1])))
        maximum_lag = min(x.size - 1, int(np.ceil(60.0 * sample_rate / rate_bpm[0])))
    if maximum_lag <= minimum_lag:
        return _invalid("periodicity_autocorrelation", "aperiodicity", "window is too short")
    section = correlation[minimum_lag : maximum_lag + 1]
    relative_lag = int(np.argmax(section))
    peak = float(section[relative_lag])
    lag = minimum_lag + relative_lag
    quality = np.clip(max(peak, 0.0) * finite_fraction, 0.0, 1.0)
    return IndicatorResult(
        "periodicity_autocorrelation",
        peak,
        float(quality),
        "aperiodicity",
        {"lag_samples": lag, "rate_bpm": 60.0 * sample_rate / lag},
    )


def _bandpass(x: FloatArray, sample_rate: float, band: tuple[float, float]) -> FloatArray:
    nyquist = sample_rate / 2
    low = max(0.05, band[0])
    high = min(band[1], nyquist * 0.95)
    if low >= high or x.size < 16:
        return np.asarray(signal.detrend(x), dtype=np.float64)
    sos = signal.butter(2, (low, high), btype="bandpass", fs=sample_rate, output="sos")
    pad_length = min(3 * (2 * len(sos) + 1), x.size - 1)
    return np.asarray(signal.sosfiltfilt(sos, x, padlen=pad_length), dtype=np.float64)


def detect_peaks_prominence(
    values: npt.ArrayLike,
    sample_rate: float,
    signal_type: SignalType,
    rate_bpm: tuple[float, float],
) -> npt.NDArray[np.int64]:
    """Detect beats using bandpass amplitude and robust prominence.

    This intentionally simple detector is independent of the energy detector.
    Its filtering/peak-search structure is inspired by Elgendi, PLoS ONE 8(10),
    2013 for PPG and Hamilton, Comput. Cardiol., 2002 for ECG.
    """

    x, _ = _prepare(values, minimum=16)
    if x is None:
        return np.empty(0, dtype=np.int64)
    band = (5.0, 18.0) if signal_type == SignalType.ECG else (0.5, 8.0)
    filtered = _bandpass(x, sample_rate, band)
    distance = max(1, int(np.floor(60.0 * sample_rate / rate_bpm[1])))
    prominence = max(float(stats.median_abs_deviation(filtered, scale="normal")), _EPS) * 0.8
    positive, properties = signal.find_peaks(filtered, distance=distance, prominence=prominence)
    negative, negative_properties = signal.find_peaks(
        -filtered, distance=distance, prominence=prominence
    )
    positive_strength = float(np.sum(properties.get("prominences", ())))
    negative_strength = float(np.sum(negative_properties.get("prominences", ())))
    selected = positive if positive_strength >= negative_strength else negative
    return np.asarray(selected, dtype=np.int64)


def detect_peaks_energy(
    values: npt.ArrayLike,
    sample_rate: float,
    signal_type: SignalType,
    rate_bpm: tuple[float, float],
) -> npt.NDArray[np.int64]:
    """Detect beats from derivative energy and adaptive local maxima.

    Squared derivative integration follows Pan and Tompkins, IEEE Trans.
    Biomed. Eng. BME-32(3), 1985. Parameters are simplified for quality
    agreement and must not be treated as a diagnostic QRS detector.
    """

    x, _ = _prepare(values, minimum=16)
    if x is None:
        return np.empty(0, dtype=np.int64)
    band = (5.0, 18.0) if signal_type == SignalType.ECG else (0.5, 8.0)
    filtered = _bandpass(x, sample_rate, band)
    derivative_energy = np.square(np.gradient(filtered))
    width = max(1, round(sample_rate * (0.12 if signal_type == SignalType.ECG else 0.20)))
    integrated = np.convolve(derivative_energy, np.ones(width) / width, mode="same")
    distance = max(1, int(np.floor(60.0 * sample_rate / rate_bpm[1])))
    threshold = float(np.median(integrated) + 1.5 * stats.median_abs_deviation(integrated))
    candidates, _ = signal.find_peaks(integrated, distance=distance, height=threshold)
    radius = max(1, round(0.12 * sample_rate))
    refined: list[int] = []
    for candidate in candidates:
        start, stop = max(0, candidate - radius), min(x.size, candidate + radius + 1)
        local = np.abs(filtered[start:stop])
        refined.append(start + int(np.argmax(local)))
    return np.asarray(sorted(set(refined)), dtype=np.int64)


def _beat_agreement(
    first: npt.NDArray[np.int64],
    second: npt.NDArray[np.int64],
    tolerance: int,
) -> float:
    if first.size == 0 or second.size == 0:
        return 0.0
    used: set[int] = set()
    matches = 0
    for peak in first:
        distances = np.abs(second - peak)
        for index in np.argsort(distances):
            candidate = int(index)
            if candidate not in used and distances[candidate] <= tolerance:
                used.add(candidate)
                matches += 1
                break
    return 2.0 * matches / (first.size + second.size)


def _template_correlation(x: FloatArray, peaks: npt.NDArray[np.int64], radius: int) -> float:
    beats = [
        x[peak - radius : peak + radius + 1] for peak in peaks if radius <= peak < x.size - radius
    ]
    if len(beats) < 2:
        return 0.0
    matrix = np.asarray(beats, dtype=np.float64)
    template = np.median(matrix, axis=0)
    template_std = float(np.std(template))
    if template_std <= _EPS:
        return 0.0
    correlations = [
        float(np.corrcoef(beat, template)[0, 1]) for beat in matrix if float(np.std(beat)) > _EPS
    ]
    finite = [value for value in correlations if np.isfinite(value)]
    return float(np.clip(np.median(finite), -1.0, 1.0)) if finite else 0.0


def beat_agreement_template(
    values: npt.ArrayLike,
    sample_rate: float,
    signal_type: SignalType,
    rate_bpm: tuple[float, float] | None,
) -> IndicatorResult:
    """Compare two independent beat detectors and beat-template morphology.

    Detector agreement as a quality concept follows Clifford et al., Physiol.
    Meas. 33(9), 2012; template correlation follows Orphanidou et al., IEEE
    J. Biomed. Health Inform. 19(3), 2015. The 150 ms tolerance and equal score
    weighting are explicit heuristics, not clinical thresholds.
    """

    if signal_type not in {SignalType.ECG, SignalType.PPG} or rate_bpm is None:
        return _invalid("beat_agreement_template", "beat_inconsistency", "not applicable")
    x, finite_fraction = _prepare(values, minimum=max(16, int(sample_rate)))
    if x is None:
        return _invalid("beat_agreement_template", "beat_inconsistency", "window is too short")
    first = detect_peaks_prominence(x, sample_rate, signal_type, rate_bpm)
    second = detect_peaks_energy(x, sample_rate, signal_type, rate_bpm)
    tolerance = max(1, round(0.15 * sample_rate))
    agreement = _beat_agreement(first, second, tolerance)
    template = _template_correlation(x, first, max(1, round(0.2 * sample_rate)))
    template_quality = max(0.0, template)
    quality = finite_fraction * (0.6 * agreement + 0.4 * template_quality)
    return IndicatorResult(
        "beat_agreement_template",
        agreement,
        float(np.clip(quality, 0.0, 1.0)),
        "beat_inconsistency",
        {
            "prominence_peak_count": int(first.size),
            "energy_peak_count": int(second.size),
            "agreement_f1": agreement,
            "template_correlation": template,
            "tolerance_ms": 1000.0 * tolerance / sample_rate,
        },
    )


def compute_indicators(
    values: npt.ArrayLike,
    sample_rate: float,
    profile: SignalProfile,
) -> tuple[IndicatorResult, ...]:
    """Compute all applicable indicators without allowing one failure to abort a window."""

    computations: tuple[Callable[[], IndicatorResult], ...] = (
        lambda: spectral_band_power_ratio(values, sample_rate, profile.physiological_band),
        lambda: spectral_entropy(values, sample_rate),
        lambda: line_noise_ratio(values, sample_rate),
        lambda: clipping_flatline_rms(values),
        lambda: baseline_wander_ratio(
            values, sample_rate, profile.baseline_band, profile.physiological_band
        ),
        lambda: distribution_shape(values),
        lambda: periodicity_autocorrelation(values, sample_rate, profile.expected_rate_bpm),
        lambda: beat_agreement_template(
            values, sample_rate, profile.signal_type, profile.expected_rate_bpm
        ),
    )
    results: list[IndicatorResult] = []
    for computation in computations:
        try:
            results.append(computation())
        except (ValueError, FloatingPointError, OverflowError) as exc:
            name = getattr(computation, "__name__", "indicator")
            results.append(_invalid(name, "numerical_failure", str(exc)))
    return tuple(results)
