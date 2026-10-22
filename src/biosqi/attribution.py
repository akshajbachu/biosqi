"""Transparent conversion of indicator deficits into artifact likelihoods."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from biosqi.fusion import FusionModel
from biosqi.types import IndicatorResult


def attribute_artifacts(
    indicators: tuple[IndicatorResult, ...],
    model: FusionModel,
) -> dict[str, float]:
    """Aggregate ``1 - quality`` by artifact using absolute model weights.

    Values are relative diagnostic severities, not probabilities. Invalid or
    non-applicable indicators are omitted instead of being silently called bad.
    """

    weights = dict(zip(model.features, model.weights, strict=True))
    weighted_deficit: defaultdict[str, float] = defaultdict(float)
    total_weight: defaultdict[str, float] = defaultdict(float)
    for indicator in indicators:
        if not indicator.valid or indicator.quality is None:
            continue
        weight = abs(weights.get(indicator.name, 1.0))
        weighted_deficit[indicator.artifact] += weight * (1.0 - indicator.quality)
        total_weight[indicator.artifact] += weight
    severities = {
        artifact: float(np.clip(deficit / total_weight[artifact], 0.0, 1.0))
        for artifact, deficit in weighted_deficit.items()
        if total_weight[artifact] > 0
    }
    return dict(sorted(severities.items(), key=lambda item: (-item[1], item[0])))
