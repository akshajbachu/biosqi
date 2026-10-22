# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-22

First public release.

### Added

- Window-aligned, calibrated quality scores for ECG, PPG, and EEG recordings
  using logistic fusion followed by isotonic calibration.
- Spectral, amplitude, drift, distribution, and periodicity indicators with
  per-window artifact attribution (mains interference, baseline wander/motion,
  clipping, flatline/dropout, abnormal amplitude, poor beat consistency).
- Python API, `biosqi` command-line interface, standalone HTML reports, and a
  local web dashboard (`biosqi view`).
- NumPy/NPZ and CSV input, plus WFDB and EDF/BDF through the `data` extra.
- Reproducible benchmark pipeline on PhysioNet/CinC 2011 Set A and MIT-BIH
  NSTDB.

[Unreleased]: https://github.com/akshajbachu/biosqi/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/akshajbachu/biosqi/releases/tag/v0.1.0
