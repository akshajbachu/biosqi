"""Run every reproducible biosqi benchmark and regenerate published results."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from benchmarks.datasets import NSTDB_RECORDS, fetch_challenge_2011, fetch_nstdb  # noqa: E402
from biosqi.fusion import FusionModel, load_default_model  # noqa: E402
from biosqi.indicators import compute_indicators  # noqa: E402
from biosqi.io import read_wfdb  # noqa: E402
from biosqi.profiles import get_profile  # noqa: E402
from biosqi.types import SignalType  # noqa: E402
from biosqi.validation.synthetic import Artifact, inject_artifact  # noqa: E402

RESULTS_START = "<!-- benchmark-results:start -->"
RESULTS_END = "<!-- benchmark-results:end -->"


def _record_features(path: Path, feature_names: tuple[str, ...]) -> np.ndarray:
    recording = read_wfdb(path, signal_type=SignalType.ECG)
    profile = get_profile(SignalType.ECG)
    rows: list[list[float]] = []
    for channel in recording.data:
        indicators = compute_indicators(channel, recording.sample_rate, profile)
        quality = {
            item.name: item.quality
            for item in indicators
            if item.valid and item.quality is not None
        }
        rows.append([float(quality.get(name, 0.5)) for name in feature_names])
    return np.median(np.asarray(rows), axis=0)


def challenge_features(
    data_root: Path, *, limit: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Extract or load deterministic record-level Set A features."""
    acceptable = (data_root / "RECORDS-acceptable").read_text().split()
    unacceptable = (data_root / "RECORDS-unacceptable").read_text().split()
    labeled = [(name, 1) for name in acceptable] + [(name, 0) for name in unacceptable]
    labeled.sort()
    if limit is not None:
        positives = [item for item in labeled if item[1] == 1][:limit]
        negatives = [item for item in labeled if item[1] == 0][:limit]
        labeled = sorted((*positives, *negatives))
    cache = data_root / f"biosqi-features-{len(labeled)}.npz"
    if cache.exists():
        payload = np.load(cache)
        return np.asarray(payload["x"]), np.asarray(payload["y"])
    names = load_default_model().features
    x = np.vstack([_record_features(data_root / name, names) for name, _ in labeled])
    y = np.asarray([label for _, label in labeled], dtype=np.int64)
    np.savez_compressed(cache, x=x, y=y)
    return x, y


def _ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = labels.size
    value = 0.0
    for low, high in pairwise(edges):
        selected = (probabilities >= low) & (
            probabilities <= high if high == 1.0 else probabilities < high
        )
        if selected.any():
            value += (
                selected.sum()
                / total
                * abs(probabilities[selected].mean() - labels[selected].mean())
            )
    return float(value)


def _bootstrap_auc(
    labels: np.ndarray, probabilities: np.ndarray, *, iterations: int = 2000
) -> tuple[float, float]:
    rng = np.random.default_rng(20260820)
    estimates: list[float] = []
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    for _ in range(iterations):
        sample = np.concatenate(
            (
                rng.choice(positive, positive.size, replace=True),
                rng.choice(negative, negative.size, replace=True),
            )
        )
        estimates.append(float(roc_auc_score(labels[sample], probabilities[sample])))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def evaluate_challenge(x: np.ndarray, y: np.ndarray) -> tuple[dict[str, Any], FusionModel]:
    """Return leakage-resistant nested-CV metrics and the release model."""
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260820)
    estimator = CalibratedClassifierCV(
        estimator=LogisticRegression(max_iter=3000, class_weight="balanced", random_state=20260820),
        method="isotonic",
        cv=3,
    )
    probabilities = cross_val_predict(estimator, x, y, cv=outer, method="predict_proba", n_jobs=-1)[
        :, 1
    ]
    predictions = probabilities >= 0.5
    ci_low, ci_high = _bootstrap_auc(y, probabilities)
    metrics = {
        "records": int(y.size),
        "acceptable": int(y.sum()),
        "unacceptable": int((1 - y).sum()),
        "evaluation": "5-fold stratified outer CV; 3-fold isotonic calibration inside each fold",
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "roc_auc_95ci": [ci_low, ci_high],
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "ece_10_bin": _ece(y, probabilities),
    }

    # The deployed logistic model uses all labels. Its calibrator sees only
    # out-of-fold probabilities, preventing target leakage into calibration.
    base = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=20260820)
    oof_raw = cross_val_predict(base, x, y, cv=outer, method="predict_proba", n_jobs=-1)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(oof_raw, y)
    base.fit(x, y)
    model = FusionModel(
        name="biosqi-cinc2011-set-a-v1",
        features=load_default_model().features,
        weights=tuple(float(value) for value in base.coef_[0]),
        intercept=float(base.intercept_[0]),
        calibration_x=tuple(float(value) for value in calibrator.X_thresholds_),
        calibration_y=tuple(float(value) for value in calibrator.y_thresholds_),
        provenance={
            "dataset": "PhysioNet/Computing in Cardiology Challenge 2011 Set A",
            "dataset_version": "1.0.0",
            "labels": "RECORDS-acceptable and RECORDS-unacceptable",
            "fit": "class-balanced logistic regression on all labeled Set A records",
            "calibration": "isotonic regression on 5-fold out-of-fold probabilities",
            "warning": "Validated for this ECG acquisition domain; not a medical device.",
        },
    )
    return metrics, model


def evaluate_nstdb(data_root: Path, model: FusionModel, *, quick: bool) -> dict[str, Any]:
    """Check that mean quality falls as electrode-motion noise increases."""
    profile = get_profile(SignalType.ECG)
    by_snr: dict[int, list[float]] = {}
    for record_name in NSTDB_RECORDS:
        snr = -6 if record_name.endswith("_6") else int(record_name[-2:])
        recording = read_wfdb(data_root / record_name, signal_type=SignalType.ECG)
        # Noise begins at 5 min and alternates 2 min noisy / 2 min clean.
        starts = range(5 * 60, min(int(recording.duration), 29 * 60), 4 * 60)
        if quick:
            starts = list(starts)[:1]
        for start_seconds in starts:
            start = int(start_seconds * recording.sample_rate)
            stop = min(start + int(10 * recording.sample_rate), recording.n_samples)
            channel_scores: list[float] = []
            for channel in recording.data[:, start:stop]:
                prediction = model.predict(
                    compute_indicators(channel, recording.sample_rate, profile)
                )
                channel_scores.append(prediction.calibrated_score)
            by_snr.setdefault(snr, []).append(float(np.median(channel_scores)))
    means = {str(snr): float(np.mean(values)) for snr, values in sorted(by_snr.items())}
    ordered = [means[str(snr)] for snr in (-6, 0, 6, 12, 18, 24)]
    rho = float(np.corrcoef(np.arange(len(ordered)), ordered)[0, 1])
    return {
        "mean_score_by_snr_db": means,
        "pearson_r_with_increasing_snr": rho,
        "monotonic_non_decreasing": bool(np.all(np.diff(ordered) >= -0.02)),
        "segments_per_record": 1 if quick else 6,
    }


def _synthetic_ecg(sample_rate: float, seconds: float, heart_rate: float, seed: int) -> np.ndarray:
    time = np.arange(int(sample_rate * seconds)) / sample_rate
    phase = np.mod(time, 60.0 / heart_rate)
    rng = np.random.default_rng(seed)
    return (
        0.04 * np.sin(2 * np.pi * 0.25 * time)
        + 0.05 * np.sin(2 * np.pi * heart_rate / 60.0 * time)
        + 1.0 * np.exp(-0.5 * ((phase - 0.10) / 0.018) ** 2)
        - 0.16 * np.exp(-0.5 * ((phase - 0.15) / 0.026) ** 2)
        + 0.23 * np.exp(-0.5 * ((phase - 0.34) / 0.060) ** 2)
        + rng.normal(0.0, 0.01, time.size)
    )


def evaluate_synthetic(model: FusionModel, *, quick: bool) -> dict[str, Any]:
    """Measure artifact-detection AUC at controlled additive-noise levels."""
    sample_rate = 250.0
    profile = get_profile(SignalType.ECG)
    repeats = 6 if quick else 30
    levels = (-6, 0, 6, 12, 18, 24)
    artifacts: dict[str, dict[str, float]] = {}
    for artifact in Artifact:
        by_snr: dict[str, float] = {}
        for snr in levels:
            clean_anomaly: list[float] = []
            corrupted_anomaly: list[float] = []
            for repetition in range(repeats):
                clean = _synthetic_ecg(
                    sample_rate,
                    10.0,
                    heart_rate=55.0 + (repetition * 11) % 80,
                    seed=1000 + repetition,
                )
                corrupted = inject_artifact(
                    clean,
                    sample_rate,
                    artifact,
                    snr_db=float(snr),
                    start_fraction=0.0,
                    duration_fraction=1.0,
                    seed=2000 + repetition,
                ).signal
                clean_score = model.predict(
                    compute_indicators(clean, sample_rate, profile)
                ).calibrated_score
                corrupted_score = model.predict(
                    compute_indicators(corrupted, sample_rate, profile)
                ).calibrated_score
                clean_anomaly.append(1.0 - clean_score)
                corrupted_anomaly.append(1.0 - corrupted_score)
            labels = np.asarray([0] * repeats + [1] * repeats)
            scores = np.asarray(clean_anomaly + corrupted_anomaly)
            by_snr[str(snr)] = float(roc_auc_score(labels, scores))
        artifacts[artifact.value] = by_snr
    return {
        "examples_per_class_per_condition": repeats,
        "snr_db": list(levels),
        "artifact_detection_auc": artifacts,
        "note": "Synthetic AUC tests controlled sensitivity, not clinical generalization.",
    }


def _line_plot_svg(
    series: dict[str, list[float]],
    x_labels: list[str],
    *,
    title: str,
    y_label: str,
) -> str:
    width, height = 900, 480
    left, top, plot_width, plot_height = 72, 48, 780, 350
    palette = ("#2962ff", "#00a67d", "#e05252", "#9c5de5", "#f08c2e", "#4f6d7a")
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" '
        f'font-size="18">{title}</text>',
    ]
    for tick in range(6):
        y_value = tick / 5
        y = top + plot_height * (1 - y_value)
        lines.append(
            f'<line x1="{left}" y1="{y}" x2="{left + plot_width}" y2="{y}" stroke="#e8e8e8"/>'
        )
        lines.append(
            f'<text x="{left - 10}" y="{y + 4}" text-anchor="end" '
            f'font-family="sans-serif" font-size="11">{y_value:.1f}</text>'
        )
    count = max(1, len(x_labels) - 1)
    for index, label in enumerate(x_labels):
        x = left + plot_width * index / count
        lines.append(
            f'<text x="{x}" y="{top + plot_height + 22}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="11">{label}</text>'
        )
    for series_index, (name, values) in enumerate(series.items()):
        color = palette[series_index % len(palette)]
        points = " ".join(
            f"{left + plot_width * index / count:.1f},"
            f"{top + plot_height * (1 - np.clip(value, 0, 1)):.1f}"
            for index, value in enumerate(values)
        )
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        legend_y = top + 16 * series_index
        lines.append(
            f'<text x="{left + plot_width - 4}" y="{legend_y}" text-anchor="end" '
            f'font-family="sans-serif" font-size="11" fill="{color}">{name}</text>'
        )
    lines.append(
        f'<text x="18" y="{top + plot_height / 2}" text-anchor="middle" '
        f'transform="rotate(-90 18 {top + plot_height / 2})" '
        f'font-family="sans-serif" font-size="12">{y_label}</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines)


def write_plots(results: dict[str, Any], output: Path) -> None:
    synthetic = results["synthetic"]["artifact_detection_auc"]
    levels = [str(value) for value in results["synthetic"]["snr_db"]]
    (output / "synthetic_auc.svg").write_text(
        _line_plot_svg(
            {name: [values[level] for level in levels] for name, values in synthetic.items()},
            levels,
            title="Synthetic artifact detection by SNR",
            y_label="ROC AUC",
        )
        + "\n"
    )
    nstdb = results["nstdb"]["mean_score_by_snr_db"]
    nstdb_levels = ["-6", "0", "6", "12", "18", "24"]
    (output / "nstdb_scores.svg").write_text(
        _line_plot_svg(
            {"NSTDB": [nstdb[level] for level in nstdb_levels]},
            nstdb_levels,
            title="Quality score across MIT-BIH NSTDB noise levels",
            y_label="Mean calibrated quality",
        )
        + "\n"
    )


def _markdown(results: dict[str, Any]) -> str:
    challenge = results["challenge_2011"]
    nstdb = results["nstdb"]
    ci = challenge["roc_auc_95ci"]
    rows = [
        "## Reproducible validation",
        "",
        "Numbers below are generated by `python benchmarks/run_all.py`; they are not",
        "hand-entered. The Challenge metric uses nested calibration inside each outer",
        "fold. Set B/C labels remain withheld and are not claimed as test results.",
        "",
        "| Benchmark | Result |",
        "| --- | ---: |",
        f"| CinC 2011 Set A records | {challenge['records']} |",
        f"| 5-fold ROC AUC (95% bootstrap CI) | {challenge['roc_auc']:.3f} "
        f"({ci[0]:.3f}-{ci[1]:.3f}) |",
        f"| Balanced accuracy | {challenge['balanced_accuracy']:.3f} |",
        f"| Brier score | {challenge['brier_score']:.3f} |",
        f"| NSTDB score/SNR correlation | {nstdb['pearson_r_with_increasing_snr']:.3f} |",
    ]
    return "\n".join(rows)


def update_readme(results: dict[str, Any]) -> None:
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    generated = f"{RESULTS_START}\n{_markdown(results)}\n{RESULTS_END}"
    pattern = re.compile(rf"{re.escape(RESULTS_START)}.*?{re.escape(RESULTS_END)}", re.DOTALL)
    text = (
        pattern.sub(generated, text)
        if pattern.search(text)
        else f"{text.rstrip()}\n\n{generated}\n"
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="balanced subset and one NST segment")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--no-update-readme", action="store_true")
    arguments = parser.parse_args()
    challenge_root = (
        fetch_challenge_2011(limit=50 if arguments.quick else None)
        if not arguments.skip_download
        else Path.home() / ".cache" / "biosqi" / "challenge-2011" / "set-a"
    )
    nstdb_root = (
        fetch_nstdb()
        if not arguments.skip_download
        else Path.home() / ".cache" / "biosqi" / "nstdb"
    )
    x, y = challenge_features(challenge_root, limit=50 if arguments.quick else None)
    challenge_results, model = evaluate_challenge(x, y)
    nstdb_results = evaluate_nstdb(nstdb_root, model, quick=arguments.quick)
    synthetic_results = evaluate_synthetic(model, quick=arguments.quick)
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "biosqi_version": "0.1.0",
        "mode": "quick" if arguments.quick else "full",
        "challenge_2011": challenge_results,
        "nstdb": nstdb_results,
        "synthetic": synthetic_results,
    }
    output = ROOT / "benchmarks" / "results"
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    write_plots(results, output)
    model.save(output / "fitted-default.json")
    if not arguments.quick:
        model.save(ROOT / "src" / "biosqi" / "models" / "default.json")
    if not arguments.no_update_readme:
        update_readme(results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
