from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import biosqi.io as bioio
from biosqi.io import OptionalDependencyError, read_csv, read_numpy, read_recording
from biosqi.types import SignalType


def test_read_numpy_npy_and_channel_axis(tmp_path: Path) -> None:
    path = tmp_path / "signal.npy"
    np.save(path, np.arange(12).reshape(6, 2))
    recording = read_numpy(path, sample_rate=100.0, channel_axis=1)
    assert recording.data.shape == (2, 6)
    assert recording.source == str(path)


def test_read_numpy_npz_metadata(tmp_path: Path) -> None:
    path = tmp_path / "signal.npz"
    np.savez(path, data=np.ones((2, 20)), fs=50.0, channel_names=np.array(["I", "II"]))
    recording = read_numpy(path, signal_type=SignalType.ECG)
    assert recording.sample_rate == 50.0
    assert recording.channel_names == ("I", "II")
    assert recording.signal_type is SignalType.ECG


def test_read_numpy_requires_rate_and_valid_shape(tmp_path: Path) -> None:
    path = tmp_path / "signal.npy"
    np.save(path, np.zeros((2, 3, 4)))
    with pytest.raises(ValueError, match="sample_rate"):
        read_numpy(path)
    with pytest.raises(ValueError, match="one- or two-dimensional"):
        read_numpy(path, sample_rate=10.0)
    two_dimensional = tmp_path / "two.npy"
    np.save(two_dimensional, np.zeros((2, 3)))
    with pytest.raises(ValueError, match="channel_axis"):
        read_numpy(two_dimensional, sample_rate=10.0, channel_axis=3)


def test_empty_npz_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.npz"
    np.savez(path)
    with pytest.raises(ValueError, match="no arrays"):
        read_numpy(path, sample_rate=10.0)


def test_read_csv_header_delimiter_and_nan(tmp_path: Path) -> None:
    path = tmp_path / "signal.tsv"
    path.write_text("red\tinfrared\n1\t2\nNaN\t3\n", encoding="utf-8")
    recording = read_csv(path, sample_rate=25.0, signal_type=SignalType.PPG)
    assert recording.channel_names == ("red", "infrared")
    assert recording.data.shape == (2, 2)
    assert np.isnan(recording.data[0, 1])


def test_read_csv_rejects_ragged_rows(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n1,2\n3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="consistent"):
        read_csv(path, sample_rate=1.0)
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        read_csv(path, sample_rate=1.0)


def test_dispatch_and_unsupported_format(tmp_path: Path) -> None:
    csv_path = tmp_path / "x.csv"
    csv_path.write_text("1,2\n3,4\n", encoding="utf-8")
    assert read_recording(csv_path, sample_rate=2.0).data.shape == (2, 2)
    with pytest.raises(ValueError, match="sample_rate"):
        read_recording(csv_path)
    unknown = tmp_path / "x.xyz"
    unknown.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        read_recording(unknown)


def test_optional_dependency_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(module: str) -> None:
        raise ImportError(module)

    monkeypatch.setattr(bioio.importlib, "import_module", missing)
    with pytest.raises(OptionalDependencyError, match=r"biosqi\[data\]"):
        bioio.read_wfdb("record")


def test_wfdb_reader_with_fake_optional_module(monkeypatch: pytest.MonkeyPatch) -> None:
    record = SimpleNamespace(
        p_signal=np.arange(12).reshape(6, 2),
        fs=125.0,
        sig_name=["I", "II"],
        units=["mV", "mV"],
    )
    module = SimpleNamespace(rdrecord=lambda _: record)
    monkeypatch.setattr(bioio, "_optional_import", lambda _module, _extra: module)
    loaded = bioio.read_wfdb("record.hea")
    assert loaded.data.shape == (2, 6)
    assert loaded.channel_names == ("I", "II")
    assert loaded.metadata["units"] == ["mV", "mV"]


def test_edf_reader_success_and_mismatched_rates(monkeypatch: pytest.MonkeyPatch) -> None:
    class Reader:
        signals_in_file = 2

        def __init__(self, _path: str, rates: tuple[float, float] = (100.0, 100.0)) -> None:
            self.rates = rates
            self.closed = False

        def getSampleFrequencies(self) -> tuple[float, float]:
            return self.rates

        def readSignal(self, index: int) -> np.ndarray:
            return np.arange(10) + index

        def getSignalLabels(self) -> list[str]:
            return ["A ", "B "]

        def getPhysicalDimensions(self) -> list[str]:
            return ["uV", "uV"]

        def close(self) -> None:
            self.closed = True

    good = SimpleNamespace(EdfReader=Reader)
    monkeypatch.setattr(bioio, "_optional_import", lambda _module, _extra: good)
    loaded = bioio.read_edf("record.edf")
    assert loaded.data.shape == (2, 10)
    assert loaded.channel_names == ("A", "B")

    bad = SimpleNamespace(EdfReader=lambda path: Reader(path, (100.0, 200.0)))
    monkeypatch.setattr(bioio, "_optional_import", lambda _module, _extra: bad)
    with pytest.raises(ValueError, match="different sample rates"):
        bioio.read_edf("record.edf")
