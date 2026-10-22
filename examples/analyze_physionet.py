"""Analyze one downloaded Challenge 2011 ECG and print transparent evidence."""

from pathlib import Path

from biosqi.analysis import analyze
from biosqi.io import read_wfdb
from biosqi.types import SignalType

cache = Path.home() / ".cache" / "biosqi" / "challenge-2011" / "set-a"
record_name = (cache / "RECORDS-acceptable").read_text().splitlines()[0]
recording = read_wfdb(cache / record_name, signal_type=SignalType.ECG)
result = analyze(recording, threshold=0.70)

for window in result.windows:
    attribution = ", ".join(
        f"{artifact}={severity:.0%}" for artifact, severity in window.artifacts.items()
    )
    print(
        f"{window.start_seconds:6.1f}-{window.stop_seconds:6.1f}s "
        f"score={window.calibrated_score:5.1%} usable={window.usable} {attribution}"
    )
