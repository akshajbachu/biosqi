"""Reproducible download helpers for the public validation datasets.

Downloaded data is intentionally kept outside the repository.  Each helper
records dataset provenance and never silently replaces an existing file.
"""

from __future__ import annotations

import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CHALLENGE_BASE = "https://physionet.org/files/challenge-2011/1.0.0/set-a"
NSTDB_BASE = "https://physionet.org/files/nstdb/1.0.0"
NSTDB_RECORDS = tuple(
    f"{subject}e{suffix}"
    for subject in ("118", "119")
    for suffix in ("24", "18", "12", "06", "00", "_6")
)


def cache_root() -> Path:
    """Return the user-overridable benchmark cache directory."""
    return Path(os.environ.get("BIOSQI_CACHE", Path.home() / ".cache" / "biosqi"))


def _download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size:
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "biosqi/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
    partial.replace(destination)
    return destination


def fetch_challenge_2011(*, limit: int | None = None, workers: int = 8) -> Path:
    """Fetch labeled Set A from the PhysioNet/CinC Challenge 2011.

    Set A contains 1,000 ten-second, 12-lead ECG records sampled at 500 Hz.
    Labels were assigned by 3-18 blinded annotators.  ``limit`` exists for
    smoke tests; publication benchmarks must leave it as ``None``.
    """
    root = cache_root() / "challenge-2011" / "set-a"
    acceptable_path = _download(f"{CHALLENGE_BASE}/RECORDS-acceptable", root / "RECORDS-acceptable")
    unacceptable_path = _download(
        f"{CHALLENGE_BASE}/RECORDS-unacceptable", root / "RECORDS-unacceptable"
    )
    acceptable = acceptable_path.read_text().split()
    unacceptable = unacceptable_path.read_text().split()
    records = (
        sorted((*acceptable, *unacceptable))
        if limit is None
        else sorted((*acceptable[:limit], *unacceptable[:limit]))
    )

    def fetch_record(record: str) -> None:
        for suffix in (".hea", ".dat"):
            _download(f"{CHALLENGE_BASE}/{record}{suffix}", root / f"{record}{suffix}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch_record, records))

    provenance = {
        "dataset": "PhysioNet/Computing in Cardiology Challenge 2011, Set A",
        "version": "1.0.0",
        "source": CHALLENGE_BASE,
        "license": "Open Data Commons Attribution 1.0",
        "records_downloaded": len(records),
        "acceptable_total": len(acceptable),
        "unacceptable_total": len(unacceptable),
    }
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return root


def fetch_nstdb(*, workers: int = 6) -> Path:
    """Fetch all 12 MIT-BIH Noise Stress Test ECG records.

    The records comprise subjects 118 and 119 with electrode-motion noise at
    24, 18, 12, 6, 0, and -6 dB SNR.  Noise alternates with clean segments.
    """
    root = cache_root() / "nstdb"

    def fetch_record(record: str) -> None:
        for suffix in (".hea", ".dat"):
            _download(f"{NSTDB_BASE}/{record}{suffix}", root / f"{record}{suffix}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch_record, NSTDB_RECORDS))
    provenance = {
        "dataset": "MIT-BIH Noise Stress Test Database",
        "version": "1.0.0",
        "doi": "10.13026/C2HS3T",
        "source": NSTDB_BASE,
        "license": "Open Data Commons Attribution 1.0",
        "records_downloaded": len(NSTDB_RECORDS),
        "snr_db": [24, 18, 12, 6, 0, -6],
    }
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return root


if __name__ == "__main__":
    fetch_challenge_2011()
    fetch_nstdb()
