# Validation

## PhysioNet/Computing in Cardiology Challenge 2011

Set A contains approximately 1,000 ten-second, 12-lead ECGs sampled at 500 Hz.
Each recording was reviewed independently by 3–18 annotators. The benchmark
uses the public acceptable and unacceptable lists; indeterminate records are
excluded, matching the Challenge scoring policy.

Reported generalization metrics come from five stratified outer folds.
Isotonic calibration is fitted using three inner folds within each outer
training split. ROC AUC includes a deterministic 2,000-sample stratified
bootstrap confidence interval. No score is reported for Set B or C because
their reference labels are withheld.

## MIT-BIH Noise Stress Test Database

The NSTDB test uses records 118 and 119 corrupted with real electrode-motion
noise at 24, 18, 12, 6, 0, and ?6 dB SNR. It checks whether mean quality tracks
increasing SNR across noisy intervals. This is a stress test of ordering, not
an independent expert-label classification test.

## Synthetic corruption

The controlled harness injects baseline wander, mains interference, smoothed
motion noise, electrode pops, clipping, and dropout into clean signals. For
additive artifacts, noise is scaled to a requested SNR and measured again
after injection. Every corruption includes an exact sample-level affected
mask.

Synthetic data tests whether an algorithm reacts to a known perturbation. It
does not substitute for validation on real acquisition failures.

## Limitations

- Challenge 2011 contains short, mobile 12-lead ECG acquisitions. Results may
  not transfer to long-term Holter, ICU, wearable single-lead, PPG, or EEG.
- Acceptability depends on the intended analysis. A segment adequate for heart
  rate may be inadequate for morphology.
- Artifact attribution is an interpretable hypothesis derived from SQIs, not a
  causal diagnosis.
- The PPG and EEG profiles are supported engineering configurations; only the
  ECG model receives the stated Challenge 2011 calibration.

Dataset licenses and provenance are recorded beside each downloaded cache.
