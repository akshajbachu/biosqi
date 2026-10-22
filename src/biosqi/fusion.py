"""Logistic SQI fusion with monotone isotonic calibration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from biosqi.types import IndicatorResult


@dataclass(frozen=True)
class FusionOutput:
    """One model prediction and its per-indicator log-odds contributions."""

    raw_score: float
    calibrated_score: float
    contributions: dict[str, float]


@dataclass(frozen=True)
class FusionModel:
    """Serializable logistic model followed by piecewise-linear isotonic calibration."""

    name: str
    features: tuple[str, ...]
    weights: tuple[float, ...]
    intercept: float
    calibration_x: tuple[float, ...] = (0.0, 1.0)
    calibration_y: tuple[float, ...] = (0.0, 1.0)
    missing_value: float = 0.5
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.features or len(self.features) != len(self.weights):
            raise ValueError("features and weights must be non-empty and equally sized")
        if len(self.calibration_x) != len(self.calibration_y) or len(self.calibration_x) < 2:
            raise ValueError("calibration arrays must have equal length and at least two points")
        if any(right < left for left, right in pairwise(self.calibration_x)):
            raise ValueError("calibration_x must be nondecreasing")
        if any(right < left for left, right in pairwise(self.calibration_y)):
            raise ValueError("calibration_y must be nondecreasing")

    def predict(self, indicators: tuple[IndicatorResult, ...]) -> FusionOutput:
        """Fuse valid indicator qualities; missing features use the documented neutral value."""

        qualities = {
            indicator.name: indicator.quality
            for indicator in indicators
            if indicator.valid and indicator.quality is not None
        }
        contributions: dict[str, float] = {}
        logit = self.intercept
        for feature, weight in zip(self.features, self.weights, strict=True):
            value = float(qualities.get(feature, self.missing_value))
            contribution = weight * value
            contributions[feature] = contribution
            logit += contribution
        raw = _sigmoid(logit)
        calibrated = float(
            np.interp(
                raw,
                np.asarray(self.calibration_x),
                np.asarray(self.calibration_y),
                left=self.calibration_y[0],
                right=self.calibration_y[-1],
            )
        )
        return FusionOutput(raw, float(np.clip(calibrated, 0.0, 1.0)), contributions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "features": list(self.features),
            "weights": list(self.weights),
            "intercept": self.intercept,
            "calibration": {"x": list(self.calibration_x), "y": list(self.calibration_y)},
            "missing_value": self.missing_value,
            "provenance": self.provenance,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FusionModel:
        calibration = payload.get("calibration", {})
        return cls(
            name=str(payload["name"]),
            features=tuple(str(item) for item in payload["features"]),
            weights=tuple(float(item) for item in payload["weights"]),
            intercept=float(payload["intercept"]),
            calibration_x=tuple(float(item) for item in calibration.get("x", (0.0, 1.0))),
            calibration_y=tuple(float(item) for item in calibration.get("y", (0.0, 1.0))),
            missing_value=float(payload.get("missing_value", 0.5)),
            provenance=dict(payload.get("provenance", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> FusionModel:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("model JSON root must be an object")
        return cls.from_dict(payload)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return float(1.0 / (1.0 + np.exp(-value)))
    exponential = np.exp(value)
    return float(exponential / (1.0 + exponential))


def load_default_model() -> FusionModel:
    """Load the package's explicitly versioned default heuristic model."""

    model_file = resources.files("biosqi").joinpath("models/default.json")
    payload = json.loads(model_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("bundled model JSON root must be an object")
    return FusionModel.from_dict(payload)


def fit_fusion_model(
    features: npt.ArrayLike,
    labels: npt.ArrayLike,
    feature_names: tuple[str, ...],
    *,
    name: str = "fitted-logistic-isotonic",
    random_state: int = 0,
) -> FusionModel:
    """Fit logistic fusion with isotonic calibration on out-of-fold predictions."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if x.ndim != 2 or x.shape[1] != len(feature_names):
        raise ValueError("features must be a 2D array matching feature_names")
    if x.shape[0] != y.size or y.size < 4:
        raise ValueError("labels must match rows and contain at least four observations")
    if not np.all(np.isfinite(x)):
        raise ValueError("training features must be finite")
    unique = np.unique(y)
    if not np.array_equal(unique, np.array([0, 1])):
        raise ValueError("labels must contain both binary classes 0 and 1")
    class_counts = np.bincount(y, minlength=2)
    if int(class_counts.min()) < 2:
        raise ValueError("each class needs at least two observations for calibration")
    folds = min(5, int(class_counts.min()))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    logistic = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    raw = cross_val_predict(
        logistic,
        x,
        y,
        cv=splitter,
        method="predict_proba",
    )[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw, y)
    logistic.fit(x, y)
    calibration_x = tuple(float(item) for item in calibrator.X_thresholds_)
    calibration_y = tuple(float(item) for item in calibrator.y_thresholds_)
    return FusionModel(
        name=name,
        features=feature_names,
        weights=tuple(float(item) for item in logistic.coef_[0]),
        intercept=float(logistic.intercept_[0]),
        calibration_x=calibration_x,
        calibration_y=calibration_y,
    )
