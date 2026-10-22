"""Create a small ECG-like NPZ recording with two known artifact intervals."""

from pathlib import Path

import numpy as np

SAMPLE_RATE = 250.0
DURATION = 40.0

time = np.arange(int(SAMPLE_RATE * DURATION)) / SAMPLE_RATE
phase = np.mod(time, 60.0 / 72.0)
signal = (
    0.08 * np.sin(2 * np.pi * 1.2 * time)
    + 1.0 * np.exp(-0.5 * ((phase - 0.10) / 0.018) ** 2)
    - 0.18 * np.exp(-0.5 * ((phase - 0.15) / 0.025) ** 2)
    + 0.22 * np.exp(-0.5 * ((phase - 0.34) / 0.06) ** 2)
)
rng = np.random.default_rng(20260820)
signal += rng.normal(0.0, 0.015, signal.size)
signal[int(12 * SAMPLE_RATE) : int(18 * SAMPLE_RATE)] += 0.35 * np.sin(
    2 * np.pi * 60 * time[int(12 * SAMPLE_RATE) : int(18 * SAMPLE_RATE)]
)
signal[int(28 * SAMPLE_RATE) : int(33 * SAMPLE_RATE)] = np.clip(
    signal[int(28 * SAMPLE_RATE) : int(33 * SAMPLE_RATE)], -0.15, 0.15
)

destination = Path(__file__).with_name("demo_ecg.npz")
np.savez_compressed(
    destination,
    data=signal,
    sample_rate=SAMPLE_RATE,
    channel_names=np.asarray(["synthetic_ecg"]),
)
print(destination)
