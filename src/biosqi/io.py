"""Readers for common biomedical and array file formats."""

from __future__ import annotations

import csv
import importlib
from pathlib import Path
from typing import Any

import numpy as np

from biosqi.types import Recording, SignalType


class OptionalDependencyError(ImportError):
    """Raised when a file reader's explicitly optional package is unavailable."""


def _optional_import(module: str, extra: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise OptionalDependencyError(
            f"reading this format requires {module!r}; install biosqi[{extra}]"
        ) from exc


def _orient(data: np.ndarray[Any, Any], channel_axis: int) -> np.ndarray[Any, Any]:
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        return data[np.newaxis, :]
    if data.ndim != 2:
        raise ValueError("input must contain a one- or two-dimensional numeric array")
    if channel_axis not in (0, 1, -1, -2):
        raise ValueError("channel_axis must identify one axis of a two-dimensional array")
    normalized_axis = channel_axis % 2
    return data if normalized_axis == 0 else data.T


def read_numpy(
    path: str | Path,
    *,
    sample_rate: float | None = None,
    signal_type: SignalType = SignalType.GENERIC,
    channel_axis: int = 0,
) -> Recording:
    """Read ``.npy`` or ``.npz`` arrays.

    An NPZ may store ``data``, ``sample_rate`` (or ``fs``), and ``channel_names``.
    Plain NPY inputs require ``sample_rate``.
    """

    source = Path(path)
    loaded = np.load(source, allow_pickle=False)
    names: tuple[str, ...] = ()
    metadata: dict[str, Any] = {}
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            keys = loaded.files
            data_key = "data" if "data" in keys else keys[0] if keys else None
            if data_key is None:
                raise ValueError("NPZ archive contains no arrays")
            data = np.asarray(loaded[data_key])
            if sample_rate is None:
                for key in ("sample_rate", "fs"):
                    if key in keys:
                        sample_rate = float(np.asarray(loaded[key]).reshape(-1)[0])
                        break
            if "channel_names" in keys:
                names = tuple(str(name) for name in np.asarray(loaded["channel_names"]).tolist())
            metadata["npz_key"] = data_key
        finally:
            loaded.close()
    else:
        data = np.asarray(loaded)
    if sample_rate is None:
        raise ValueError("sample_rate is required unless stored in an NPZ archive")
    oriented = _orient(data, channel_axis)
    return Recording(oriented, sample_rate, names, signal_type, str(source), metadata)


def read_csv(
    path: str | Path,
    *,
    sample_rate: float,
    signal_type: SignalType = SignalType.GENERIC,
) -> Recording:
    """Read a numeric CSV/TSV where rows are samples and columns are channels."""

    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        rows = list(csv.reader(handle, dialect))
    rows = [row for row in rows if row and any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError("CSV file is empty")

    def numeric(row: list[str]) -> bool:
        try:
            [float(cell.strip()) for cell in row]
        except ValueError:
            return False
        return True

    has_header = not numeric(rows[0])
    names = tuple(cell.strip() or f"ch{i}" for i, cell in enumerate(rows[0])) if has_header else ()
    body = rows[1:] if has_header else rows
    width = len(body[0]) if body else 0
    if width == 0 or any(len(row) != width for row in body):
        raise ValueError("CSV rows must have a consistent, non-zero column count")
    if names and len(names) != width:
        raise ValueError("CSV header and data column counts differ")
    try:
        data = np.asarray([[float(cell.strip()) for cell in row] for row in body], dtype=np.float64)
    except ValueError as exc:
        raise ValueError("CSV contains non-numeric data") from exc
    return Recording(data.T, sample_rate, names, signal_type, str(source))


def read_wfdb(
    path: str | Path,
    *,
    signal_type: SignalType = SignalType.ECG,
) -> Recording:
    """Read a WFDB record using the optional ``wfdb`` package."""

    wfdb = _optional_import("wfdb", "data")
    source = Path(path)
    record_name = str(source.with_suffix("")) if source.suffix in {".hea", ".dat"} else str(source)
    record = wfdb.rdrecord(record_name)
    data = np.asarray(record.p_signal, dtype=np.float64).T
    names = tuple(str(name) for name in getattr(record, "sig_name", ()))
    metadata = {"units": list(getattr(record, "units", ()))}
    return Recording(data, float(record.fs), names, signal_type, str(source), metadata)


def read_edf(
    path: str | Path,
    *,
    signal_type: SignalType = SignalType.GENERIC,
) -> Recording:
    """Read EDF/BDF via ``pyedflib``; channels must share a sample rate."""

    pyedflib = _optional_import("pyedflib", "data")
    source = Path(path)
    reader = pyedflib.EdfReader(str(source))
    try:
        rates = np.asarray(reader.getSampleFrequencies(), dtype=np.float64)
        if rates.size == 0:
            raise ValueError("EDF file contains no channels")
        if not np.allclose(rates, rates[0]):
            raise ValueError(
                "EDF channels have different sample rates; select or resample channels first"
            )
        data = np.vstack([reader.readSignal(index) for index in range(reader.signals_in_file)])
        names = tuple(str(name).strip() for name in reader.getSignalLabels())
        metadata = {"physical_dimensions": list(reader.getPhysicalDimensions())}
    finally:
        reader.close()
    return Recording(data, float(rates[0]), names, signal_type, str(source), metadata)


def read_recording(
    path: str | Path,
    *,
    sample_rate: float | None = None,
    signal_type: SignalType = SignalType.GENERIC,
    channel_axis: int = 0,
) -> Recording:
    """Dispatch to a reader based on the source suffix."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in {".npy", ".npz"}:
        return read_numpy(
            source,
            sample_rate=sample_rate,
            signal_type=signal_type,
            channel_axis=channel_axis,
        )
    if suffix in {".csv", ".tsv", ".txt"}:
        if sample_rate is None:
            raise ValueError("sample_rate is required for delimited text files")
        return read_csv(source, sample_rate=sample_rate, signal_type=signal_type)
    if suffix in {".edf", ".bdf"}:
        return read_edf(source, signal_type=signal_type)
    if suffix in {".hea", ".dat"} or not suffix:
        return read_wfdb(source, signal_type=signal_type)
    raise ValueError(f"unsupported input format: {suffix or '<none>'}")
