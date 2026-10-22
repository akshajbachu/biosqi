# Signal-quality indicators

Every indicator returns a raw value, normalized quality in `[0, 1]`, an
artifact family, and diagnostic evidence. Thresholds are profile-specific
engineering priors and are superseded by fitted fusion weights.

## Spectral power ratio

Compares power in the profile’s expected physiological band with total power.
Welch’s estimator is used for sufficiently long windows. This is related to
the `pSQI` family described by Clifford et al. (2012).

## Spectral entropy

Computes normalized Shannon entropy of the power spectral density. Extremely
concentrated or diffuse spectra can both be implausible; profile thresholds
map the raw value to quality without hiding it.

## Mains interference

Measures power near 50 or 60 Hz and harmonics relative to neighboring or total
power. The analysis tests the configured mains frequency and reports the
evidence used.

## Clipping, flatline, and amplitude

Detects repeated samples near observed rails, long runs with negligible
first-difference, and implausible RMS. Because recording units differ by
hardware, amplitude evidence is treated cautiously unless the profile supplies
meaningful bounds.

## Baseline wander

Measures low-frequency power beneath the profile cutoff relative to total
power after removing the mean. It is evidence of drift or motion, not proof of
the physical source.

## Distribution shape

Skewness (`sSQI`) and kurtosis (`kSQI`) summarize waveform shape. These are
weak indicators individually and receive modest fusion weight.

## Periodicity and beat agreement

Autocorrelation peak prominence estimates quasi-periodic consistency. For ECG
and PPG, two independent peak detectors are compared using tolerance-matched
F1 agreement (`bSQI`), and aligned beat shapes are compared with their robust
template.

## References

- Clifford, G. D., Lopez, D., Li, Q., & Rezek, I. (2012). Signal quality
  indices and data fusion for determining clinical acceptability of
  electrocardiograms. *Physiological Measurement*, 33(9), 1419–1433.
- Li, Q., Mark, R. G., & Clifford, G. D. (2008). Robust heart rate estimation
  from multiple asynchronous noisy sources using signal quality indices and a
  Kalman filter. *Physiological Measurement*, 29(1), 15–32.
- Moody, G. B., Muldrow, W. E., & Mark, R. G. (1984). A noise stress test for
  arrhythmia detectors. *Computers in Cardiology*, 11, 381–384.
- Silva, I., Moody, G. B., & Celi, L. (2011). Improving the quality of ECGs
  collected using mobile phones: the PhysioNet/Computing in Cardiology
  Challenge 2011. *Computing in Cardiology*, 38, 273–276.
