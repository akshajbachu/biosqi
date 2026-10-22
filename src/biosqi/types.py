"""Typed domain objects shared by the numerical core and command-line interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


class SignalType(str, Enum):
    """Supported biomedical signal families."""

    ECG = "ecg"
    PPG = "ppg"
    EEG = "eeg"
    RESP = "resp"
    GENERIC = "generic"


@dataclass(frozen=True)
class Recording:
    """A sampled, channel-major biomedical recording."""

    data: FloatArray
    sample_rate: float
    channel_names: tuple[str, ...] = ()
    signal_type: SignalType = SignalType.GENERIC
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        data = np.asarray(self.data, dtype=np.float64)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        if data.ndim != 2:
            raise ValueError("recording data must be one- or two-dimensional")
        if data.shape[0] < 1:
            raise ValueError("recording must contain at least one channel")
        if not np.isfinite(self.sample_rate) or self.sample_rate <= 0:
            raise ValueError("sample_rate must be finite and positive")
        signal_type = (
            self.signal_type
            if isinstance(self.signal_type, SignalType)
            else SignalType(str(self.signal_type).lower())
        )
        names = self.channel_names or tuple(f"ch{i}" for i in range(data.shape[0]))
        if len(names) != data.shape[0]:
            raise ValueError("channel_names must match the channel count")
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "channel_names", tuple(names))
        object.__setattr__(self, "signal_type", signal_type)

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[1])

    @property
    def duration(self) -> float:
        return self.n_samples / self.sample_rate


@dataclass(frozen=True)
class SignalWindow:
    """A deterministic half-open sample window into a recording."""

    index: int
    start: int
    stop: int
    data: FloatArray
    sample_rate: float

    @property
    def start_seconds(self) -> float:
        return self.start / self.sample_rate

    @property
    def stop_seconds(self) -> float:
        return self.stop / self.sample_rate


@dataclass(frozen=True)
class IndicatorResult:
    """One indicator's normalized quality and transparent diagnostic evidence."""

    name: str
    value: float | None
    quality: float | None
    artifact: str
    evidence: dict[str, float | int | str | bool | None] = field(default_factory=dict)
    valid: bool = True
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.quality is not None and not 0.0 <= self.quality <= 1.0:
            raise ValueError("indicator quality must lie in [0, 1]")


@dataclass(frozen=True)
class ChannelResult:
    """Quality result for one channel in one window."""

    channel: str
    score: float
    calibrated_score: float
    indicators: tuple[IndicatorResult, ...]
    artifacts: dict[str, float]


@dataclass(frozen=True)
class WindowResult:
    """Aggregated result for one recording window."""

    index: int
    start: int
    stop: int
    start_seconds: float
    stop_seconds: float
    score: float
    calibrated_score: float
    usable: bool
    channels: tuple[ChannelResult, ...]
    artifacts: dict[str, float]


@dataclass(frozen=True)
class AnalysisResult:
    """Complete, serializable output of a quality analysis."""

    source: str | None
    signal_type: SignalType
    sample_rate: float
    channel_names: tuple[str, ...]
    n_samples: int
    window_seconds: float
    step_seconds: float
    threshold: float
    windows: tuple[WindowResult, ...]
    model_name: str
    warnings: tuple[str, ...] = ()

    @property
    def scores(self) -> FloatArray:
        return np.asarray([window.calibrated_score for window in self.windows], dtype=np.float64)

    @property
    def mask(self) -> npt.NDArray[np.bool_]:
        return np.asarray([window.usable for window in self.windows], dtype=np.bool_)


def source_name(path: str | Path | None) -> str | None:
    """Return a stable display name for a source path."""

    return None if path is None else str(Path(path))
