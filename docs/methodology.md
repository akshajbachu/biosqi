# Methodology

## Pipeline

1. A reader converts the source into a channel-major `float64` recording and
   preserves sampling rate and channel names.
2. Deterministic half-open windows are generated in sample coordinates. No
   sample is silently interpolated.
3. Every channel is evaluated independently. Undefined indicators are marked
   unavailable with a reason instead of being converted to a perfect or failed
   score.
4. Normalized indicator qualities are fused by logistic regression.
5. Raw probabilities pass through a monotone isotonic calibration map learned
   from out-of-fold predictions.
6. Channel scores are aggregated by median. Artifact severities use the
   conservative maximum across channels.

## What “calibrated” means

Calibration is population-specific. In the released ECG model, logistic
weights are fitted using all labeled records from PhysioNet/CinC Challenge
2011 Set A. The isotonic map is fitted to predictions produced when each
record was held outside its training fold. This prevents the calibrator from
seeing in-sample predictions.

A score near 0.8 therefore means “among similar Set A recordings receiving
this score, approximately 80% were labeled acceptable,” subject to sampling
error and dataset shift. It does **not** mean the waveform is 80% clean, nor
does it imply clinical safety.

## Missing evidence

Some SQIs are meaningful only for particular signal families. Beat-template
agreement, for example, is appropriate for ECG/PPG but not generic EEG.
Missing indicator values use the model’s documented neutral value, and the
analysis report records a warning. The library never fills missing waveform
samples before calculating an SQI.

## Artifact attribution

Attribution is evidence-based and intentionally multi-label. A window may show
both line interference and clipping. Each artifact severity is a transparent
function of one or more normalized SQIs; it is not a claim that the physical
cause has been proven. The raw SQIs remain available for inspection.

## Reproducibility

Run the quick benchmark:

```bash
uv run python benchmarks/run_all.py --quick
```

The release benchmark omits `--quick`, downloads all public labeled records,
performs nested cross-validation, writes `benchmarks/results/results.json`,
fits the packaged model, and replaces the generated README metric block.
