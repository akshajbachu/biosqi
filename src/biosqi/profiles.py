"""Signal-family profiles used to select physiologically relevant SQIs."""

from __future__ import annotations

from dataclasses import dataclass

from biosqi.types import SignalType


@dataclass(frozen=True)
class SignalProfile:
    """Frequency and beat-rate assumptions for one signal family.

    Bounds are deliberately broad. They are engineering defaults for quality
    triage, not diagnostic limits and not claims about individual physiology.
    """

    signal_type: SignalType
    physiological_band: tuple[float, float]
    baseline_band: tuple[float, float]
    expected_rate_bpm: tuple[float, float] | None
    invertible: bool
    window_seconds: float = 10.0
    step_seconds: float = 5.0


PROFILES: dict[SignalType, SignalProfile] = {
    SignalType.ECG: SignalProfile(SignalType.ECG, (0.5, 40.0), (0.05, 0.5), (30.0, 220.0), True),
    SignalType.PPG: SignalProfile(SignalType.PPG, (0.4, 12.0), (0.05, 0.4), (30.0, 220.0), True),
    SignalType.EEG: SignalProfile(SignalType.EEG, (0.5, 45.0), (0.05, 0.5), None, True),
    SignalType.RESP: SignalProfile(
        SignalType.RESP, (0.05, 2.0), (0.01, 0.05), (3.0, 60.0), True, 30.0, 15.0
    ),
    SignalType.GENERIC: SignalProfile(SignalType.GENERIC, (0.1, 40.0), (0.01, 0.1), None, True),
}


def get_profile(signal_type: SignalType | str) -> SignalProfile:
    """Return the profile for an enum or case-insensitive string."""

    try:
        kind = (
            signal_type if isinstance(signal_type, SignalType) else SignalType(signal_type.lower())
        )
    except ValueError as exc:
        supported = ", ".join(item.value for item in SignalType)
        raise ValueError(
            f"unsupported signal type {signal_type!r}; choose from {supported}"
        ) from exc
    return PROFILES[kind]
