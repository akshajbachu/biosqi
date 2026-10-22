"""Typer command-line interface for biosqi."""

from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from biosqi.analysis import analyze
from biosqi.fusion import FusionModel, fit_fusion_model, load_default_model
from biosqi.io import OptionalDependencyError, read_recording
from biosqi.report import result_to_json, write_html_report, write_json_report
from biosqi.types import AnalysisResult, SignalType

app = typer.Typer(
    name="biosqi",
    no_args_is_help=True,
    help="Transparent, calibrated signal-quality assessment.",
)
console = Console()


def _kind(value: str) -> SignalType:
    try:
        return SignalType(value.lower())
    except ValueError as exc:
        choices = ", ".join(item.value for item in SignalType)
        raise typer.BadParameter(f"choose one of: {choices}") from exc


def _analyze_path(
    input_path: Path,
    sample_rate: float | None,
    signal_type: str,
    channel_axis: int,
    window_seconds: float | None,
    step_seconds: float | None,
    threshold: float,
    model_path: Path | None,
) -> AnalysisResult:
    try:
        recording = read_recording(
            input_path,
            sample_rate=sample_rate,
            signal_type=_kind(signal_type),
            channel_axis=channel_axis,
        )
        model = FusionModel.load(model_path) if model_path else load_default_model()
        return analyze(
            recording,
            window_seconds=window_seconds,
            step_seconds=step_seconds,
            threshold=threshold,
            model=model,
        )
    except (OSError, ValueError, OptionalDependencyError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def score(
    input_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    sample_rate: Annotated[
        float | None, typer.Option("--sample-rate", "-r", help="Hz; required for NPY/CSV.")
    ] = None,
    signal_type: Annotated[str, typer.Option("--signal-type", "-t")] = "generic",
    channel_axis: Annotated[int, typer.Option(help="Channel axis for NumPy arrays.")] = 0,
    window_seconds: Annotated[float | None, typer.Option("--window")] = None,
    step_seconds: Annotated[float | None, typer.Option("--step")] = None,
    threshold: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.5,
    model: Annotated[Path | None, typer.Option(exists=True, readable=True)] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print full JSON to stdout.")] = False,
) -> None:
    """Score an input recording window by window."""

    result = _analyze_path(
        input_path,
        sample_rate,
        signal_type,
        channel_axis,
        window_seconds,
        step_seconds,
        threshold,
        model,
    )
    if output:
        write_json_report(result, output)
        console.print(f"Wrote [bold]{output}[/bold]")
        return
    if json_output:
        typer.echo(result_to_json(result))
        return
    table = Table("Window", "Time (s)", "Quality", "Decision", "Leading artifact")
    for window in result.windows:
        artifact = next(iter(window.artifacts), "-")
        table.add_row(
            str(window.index),
            f"{window.start_seconds:.2f}-{window.stop_seconds:.2f}",
            f"{window.calibrated_score:.3f}",
            "[green]usable[/green]" if window.usable else "[red]reject[/red]",
            artifact,
        )
    console.print(table)


@app.command()
def mask(
    input_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    sample_rate: Annotated[float | None, typer.Option("--sample-rate", "-r")] = None,
    signal_type: Annotated[str, typer.Option("--signal-type", "-t")] = "generic",
    channel_axis: Annotated[int, typer.Option()] = 0,
    window_seconds: Annotated[float | None, typer.Option("--window")] = None,
    step_seconds: Annotated[float | None, typer.Option("--step")] = None,
    threshold: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.5,
    model: Annotated[Path | None, typer.Option(exists=True, readable=True)] = None,
) -> None:
    """Write a sample-level boolean mask using majority vote in overlapping windows."""

    result = _analyze_path(
        input_path,
        sample_rate,
        signal_type,
        channel_axis,
        window_seconds,
        step_seconds,
        threshold,
        model,
    )
    votes = np.zeros(result.n_samples, dtype=np.int64)
    coverage = np.zeros(result.n_samples, dtype=np.int64)
    for window in result.windows:
        coverage[window.start : window.stop] += 1
        votes[window.start : window.stop] += int(window.usable)
    sample_mask = (coverage > 0) & (votes * 2 >= coverage)
    if output.suffix.lower() == ".npy":
        np.save(output, sample_mask)
    else:
        np.savetxt(output, sample_mask.astype(np.uint8), delimiter=",", fmt="%d", header="usable")
    console.print(f"Wrote {sample_mask.size} mask values to [bold]{output}[/bold]")


@app.command()
def report(
    input_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    sample_rate: Annotated[float | None, typer.Option("--sample-rate", "-r")] = None,
    signal_type: Annotated[str, typer.Option("--signal-type", "-t")] = "generic",
    channel_axis: Annotated[int, typer.Option()] = 0,
    window_seconds: Annotated[float | None, typer.Option("--window")] = None,
    step_seconds: Annotated[float | None, typer.Option("--step")] = None,
    threshold: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.5,
    model: Annotated[Path | None, typer.Option(exists=True, readable=True)] = None,
) -> None:
    """Create a self-contained HTML report or a JSON report."""

    result = _analyze_path(
        input_path,
        sample_rate,
        signal_type,
        channel_axis,
        window_seconds,
        step_seconds,
        threshold,
        model,
    )
    if output.suffix.lower() == ".json":
        write_json_report(result, output)
    else:
        write_html_report(result, output)
    console.print(f"Wrote [bold]{output}[/bold]")


@app.command()
def fit(
    training_csv: Annotated[Path, typer.Argument(exists=True, readable=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    label: Annotated[str, typer.Option(help="Binary target column.")] = "label",
    name: Annotated[str, typer.Option(help="Model provenance name.")] = "biosqi-fitted",
) -> None:
    """Fit logistic fusion plus isotonic calibration from a feature CSV."""

    with training_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or label not in reader.fieldnames:
            raise typer.BadParameter(f"training CSV must include a {label!r} column")
        feature_names = tuple(column for column in reader.fieldnames if column != label)
        rows = list(reader)
    if not feature_names:
        raise typer.BadParameter("training CSV must include at least one feature column")
    try:
        x = np.asarray(
            [[float(row[column]) for column in feature_names] for row in rows], dtype=np.float64
        )
        y = np.asarray([int(row[label]) for row in rows], dtype=np.int64)
        fitted = fit_fusion_model(x, y, feature_names, name=name)
        fitted.save(output)
    except (KeyError, TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(
        f"Wrote [bold]{output}[/bold]. Validate calibration on held-out representative data."
    )


@app.command()
def view(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8000,
) -> None:
    """Launch the optional local report viewer."""

    try:
        uvicorn = importlib.import_module("uvicorn")
        module = importlib.import_module("biosqi.server.app")
        web_app = module.app
    except (ImportError, AttributeError) as exc:
        raise typer.BadParameter(
            "view requires the UI extra and server package; install biosqi[ui]"
        ) from exc
    uvicorn.run(web_app, host=host, port=port)


if __name__ == "__main__":
    app()
