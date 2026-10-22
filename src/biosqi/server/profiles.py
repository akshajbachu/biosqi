"""Profile discovery for the local dashboard."""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping
from typing import Any

DEFAULT_PROFILES: list[dict[str, Any]] = [
    {
        "id": "ecg",
        "name": "ECG - General",
        "modality": "ECG",
        "description": "Balanced morphology, baseline, amplitude, and spectral checks.",
        "default_window_seconds": 10.0,
        "default_overlap": 0.5,
        "default_threshold": 0.65,
    },
    {
        "id": "ppg",
        "name": "PPG - General",
        "modality": "PPG",
        "description": "Pulse-shape and motion-aware assessment for optical recordings.",
        "default_window_seconds": 8.0,
        "default_overlap": 0.5,
        "default_threshold": 0.62,
    },
    {
        "id": "eeg",
        "name": "EEG - General",
        "modality": "EEG",
        "description": "Amplitude, flatline, line-noise, and high-frequency artifact checks.",
        "default_window_seconds": 4.0,
        "default_overlap": 0.5,
        "default_threshold": 0.60,
    },
]


def _value(source: Any, name: str, default: Any) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _normalize_profile(profile_id: str, source: Any) -> dict[str, Any]:
    signal_type = _value(source, "signal_type", profile_id)
    normalized_id = str(getattr(signal_type, "value", signal_type)).lower()
    modality = str(_value(source, "modality", normalized_id)).upper()
    window_seconds = float(
        _value(source, "default_window_seconds", _value(source, "window_seconds", 10.0))
    )
    step_seconds = float(_value(source, "step_seconds", window_seconds / 2))
    return {
        "id": str(_value(source, "id", _value(source, "name", normalized_id))).lower(),
        "name": str(
            _value(
                source,
                "display_name",
                _value(source, "name", f"{modality} - General"),
            )
        ),
        "modality": modality,
        "description": str(_value(source, "description", f"{modality} signal-quality profile")),
        "default_window_seconds": window_seconds,
        "default_overlap": float(
            _value(
                source,
                "default_overlap",
                _value(source, "overlap", 1.0 - step_seconds / window_seconds),
            )
        ),
        "default_threshold": float(
            _value(source, "default_threshold", _value(source, "threshold", 0.65))
        ),
    }


def available_profiles() -> list[dict[str, Any]]:
    """Return normalized profiles without requiring a specific core registry API."""
    profiles = {item["id"]: item for item in DEFAULT_PROFILES}
    try:
        module = importlib.import_module("biosqi.profiles")
    except (ImportError, AttributeError):
        return list(profiles.values())

    registry: Any = getattr(module, "PROFILES", None)
    if registry is None:
        list_profiles = getattr(module, "list_profiles", None)
        if callable(list_profiles):
            try:
                registry = list_profiles()
            except Exception:
                registry = None

    try:
        discovered: Iterable[dict[str, Any]]
        if isinstance(registry, Mapping):
            discovered = (
                _normalize_profile(
                    str(getattr(profile_id, "value", profile_id)),
                    profile,
                )
                for profile_id, profile in registry.items()
            )
        elif registry is not None:
            discovered = (
                _normalize_profile(str(_value(profile, "id", index)), profile)
                for index, profile in enumerate(registry)
            )
        else:
            discovered = ()
        for profile in discovered:
            profiles[profile["id"]] = profile
    except (TypeError, ValueError):
        pass
    return list(profiles.values())
