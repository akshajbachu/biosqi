"""Machine-readable and self-contained HTML report serialization."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from biosqi.types import AnalysisResult, ChannelResult, IndicatorResult, WindowResult


def _indicator_dict(indicator: IndicatorResult) -> dict[str, Any]:
    return {
        "name": indicator.name,
        "value": indicator.value,
        "quality": indicator.quality,
        "artifact": indicator.artifact,
        "evidence": indicator.evidence,
        "valid": indicator.valid,
        "reason": indicator.reason,
    }


def _channel_dict(channel: ChannelResult) -> dict[str, Any]:
    return {
        "channel": channel.channel,
        "score": channel.score,
        "calibrated_score": channel.calibrated_score,
        "artifacts": channel.artifacts,
        "indicators": [_indicator_dict(indicator) for indicator in channel.indicators],
    }


def _window_dict(window: WindowResult) -> dict[str, Any]:
    return {
        "index": window.index,
        "start": window.start,
        "stop": window.stop,
        "start_seconds": window.start_seconds,
        "stop_seconds": window.stop_seconds,
        "score": window.score,
        "calibrated_score": window.calibrated_score,
        "usable": window.usable,
        "artifacts": window.artifacts,
        "channels": [_channel_dict(channel) for channel in window.channels],
    }


def result_to_dict(result: AnalysisResult) -> dict[str, Any]:
    """Convert an analysis to stable JSON-compatible primitives."""

    usable = sum(window.usable for window in result.windows)
    return {
        "schema_version": "1.0",
        "source": result.source,
        "signal_type": result.signal_type.value,
        "sample_rate": result.sample_rate,
        "channel_names": list(result.channel_names),
        "n_samples": result.n_samples,
        "duration_seconds": result.n_samples / result.sample_rate,
        "window_seconds": result.window_seconds,
        "step_seconds": result.step_seconds,
        "threshold": result.threshold,
        "model_name": result.model_name,
        "summary": {
            "window_count": len(result.windows),
            "usable_window_count": usable,
            "usable_fraction": usable / len(result.windows) if result.windows else 0.0,
        },
        "warnings": list(result.warnings),
        "windows": [_window_dict(window) for window in result.windows],
    }


def result_to_json(result: AnalysisResult, *, indent: int | None = 2) -> str:
    """Serialize an analysis as standards-compliant JSON (NaN is rejected)."""

    return json.dumps(result_to_dict(result), indent=indent, allow_nan=False)


def write_json_report(result: AnalysisResult, path: str | Path) -> Path:
    output = Path(path)
    output.write_text(result_to_json(result) + "\n", encoding="utf-8")
    return output


def _score_bar(score: float) -> str:
    width = max(0.0, min(100.0, score * 100.0))
    return (
        '<span class="bar" aria-label="'
        f'{score:.1%}"><span style="width:{width:.2f}%"></span></span>'
    )


def render_html_report(result: AnalysisResult) -> str:
    """Render a standalone HTML document with no network or script dependencies."""

    payload = result_to_dict(result)
    summary = payload["summary"]
    rows: list[str] = []
    for window in result.windows:
        artifacts = ", ".join(
            f"{html.escape(name)} {severity:.0%}"
            for name, severity in list(window.artifacts.items())[:3]
        )
        rows.append(
            "<tr>"
            f"<td>{window.index}</td>"
            f"<td>{window.start_seconds:.2f}-{window.stop_seconds:.2f}</td>"
            f"<td>{_score_bar(window.calibrated_score)} {window.calibrated_score:.3f}</td>"
            f'<td><span class="{"good" if window.usable else "bad"}">'
            f"{'usable' if window.usable else 'reject'}</span></td>"
            f"<td>{artifacts or '-'}</td>"
            "</tr>"
        )
    warnings = "".join(f"<li>{html.escape(item)}</li>" for item in result.warnings)
    warning_details = (
        f"<details><summary>Warnings ({len(result.warnings)})</summary>"
        f"<ul>{warnings}</ul></details>"
        if warnings
        else ""
    )
    embedded_json = json.dumps(payload, allow_nan=False).replace("</", "<\\/")
    title = html.escape(Path(result.source).name if result.source else "In-memory recording")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>biosqi report - {title}</title>
<style>
:root{{--ink:#15212b;--muted:#61717f;--line:#dbe3e8;--paper:#fff;--bg:#f4f7f8;
--good:#13795b;--bad:#b42318;--accent:#2b6cb0}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,sans-serif}} main{{max-width:1080px;margin:32px auto;
padding:0 20px}} .card{{background:var(--paper);border:1px solid var(--line);
border-radius:12px;padding:22px;margin:16px 0;box-shadow:0 2px 8px #172b3a0a}}
h1,h2{{margin:0 0 12px}} .muted{{color:var(--muted)}} .metrics{{display:grid;
grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}} .metric strong{{
font-size:24px;display:block}} table{{width:100%;border-collapse:collapse}} th,td{{
padding:10px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}} th{{
font-size:12px;text-transform:uppercase;color:var(--muted)}} .bar{{display:inline-block;
width:90px;height:9px;background:#e6edf1;border-radius:6px;overflow:hidden;vertical-align:middle}}
.bar span{{display:block;height:100%;background:var(--accent)}} .good{{color:var(--good);
font-weight:700}} .bad{{color:var(--bad);font-weight:700}} details{{margin-top:12px}}
code{{word-break:break-word}} @media(max-width:700px){{table{{font-size:12px}}th,td{{padding:7px}}}}
</style>
</head>
<body><main>
<section class="card"><h1>biosqi quality report</h1>
<p class="muted">{title} - {html.escape(result.signal_type.value.upper())} -
{result.sample_rate:g} Hz - {result.n_samples / result.sample_rate:.2f} s</p>
<div class="metrics">
<div class="metric"><strong>{summary["usable_fraction"]:.1%}</strong>usable windows</div>
<div class="metric"><strong>{summary["window_count"]}</strong>windows</div>
<div class="metric"><strong>{result.threshold:.2f}</strong>decision threshold</div>
<div class="metric"><strong>{len(result.channel_names)}</strong>channels</div>
</div></section>
<section class="card"><h2>Windows</h2>
<table><thead><tr><th>#</th><th>Time (s)</th><th>Quality</th><th>Decision</th>
<th>Leading artifact evidence</th></tr></thead><tbody>{"".join(rows)}</tbody></table></section>
<section class="card"><h2>Interpretation</h2>
<p>Scores are model outputs for signal-quality triage, not clinical diagnoses.
Artifact values are relative evidence derived from indicator deficits, not probabilities.
Thresholds and the bundled model require validation for the intended population and task.</p>
{warning_details}
<details><summary>Embedded machine-readable result</summary>
<p class="muted">The complete JSON result is embedded below for reproducibility.</p>
<code>schema 1.0 - model {html.escape(result.model_name)}</code></details></section>
<script id="biosqi-result" type="application/json">{embedded_json}</script>
</main></body></html>"""


def write_html_report(result: AnalysisResult, path: str | Path) -> Path:
    output = Path(path)
    output.write_text(render_html_report(result), encoding="utf-8")
    return output
