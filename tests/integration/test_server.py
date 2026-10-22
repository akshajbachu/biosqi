from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient

from biosqi.server.analysis import analyze_signal
from biosqi.server.app import (
    ANALYSES,
    _csv_signal,
    _numpy_signal,
    _remember,
    _wfdb_signal,
    create_app,
)


@pytest.fixture
def client() -> TestClient:
    ANALYSES.clear()
    return TestClient(create_app(static_dir=Path("/not/a/dashboard")))


def test_health_and_profiles(client: TestClient) -> None:
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["local_only"] is True

    response = client.get("/api/profiles")
    assert response.status_code == 200
    profiles = response.json()["profiles"]
    assert {"ecg", "ppg", "eeg"} <= {profile["id"] for profile in profiles}


def test_analyzes_csv_and_exports_results(client: TestClient) -> None:
    time = np.arange(0, 12, 1 / 100)
    values = np.sin(2 * np.pi * 1.2 * time)
    csv_data = "time,signal\n" + "\n".join(
        f"{timestamp:.3f},{value:.8f}" for timestamp, value in zip(time, values, strict=True)
    )
    response = client.post(
        "/api/analyze",
        files={"file": ("clean.csv", csv_data.encode(), "text/csv")},
        data={
            "profile": "ecg",
            "window_seconds": "4",
            "overlap": "0.5",
            "threshold": "0.65",
            "sample_rate": "100",
        },
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["filename"] == "clean.csv"
    assert result["summary"]["total_windows"] >= 3
    assert len(result["waveform"]) > 100
    assert result["windows"][0]["indicators"]

    csv_export = client.get(
        f"/api/analyses/{result['id']}/export",
        params={"format": "csv"},
    )
    assert csv_export.status_code == 200
    assert "indicator_contribution" in csv_export.text
    assert "attachment;" in csv_export.headers["content-disposition"]

    json_export = client.get(f"/api/export/{result['id']}", params={"format": "json"})
    assert json_export.status_code == 200
    assert json.loads(json_export.text)["id"] == result["id"]


def test_analyzes_npy_upload(client: TestClient) -> None:
    stream = io.BytesIO()
    np.save(stream, np.column_stack([np.sin(np.arange(500) / 20), np.ones(500)]))
    response = client.post(
        "/api/analyze",
        files={"files": ("record.npy", stream.getvalue(), "application/octet-stream")},
        data={"sample_rate": "100", "window_seconds": "2"},
    )
    assert response.status_code == 200
    assert response.json()["sample_rate"] == 100


def test_analyzes_npz_and_uses_embedded_metadata(client: TestClient) -> None:
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        data=np.sin(np.arange(1000) / 20),
        sample_rate=200.0,
        channel_names=np.asarray(["lead_ii"]),
    )
    response = client.post(
        "/api/analyze",
        files={"file": ("record.npz", stream.getvalue(), "application/octet-stream")},
        data={"sample_rate": "1", "window_seconds": "2"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["sample_rate"] == 200.0
    assert response.json()["channel"] == "lead_ii"


@pytest.mark.parametrize(
    ("data", "detail"),
    [
        ({"overlap": "1"}, "overlap"),
        ({"window_seconds": "0"}, "window_seconds"),
        ({"threshold": "2"}, "threshold"),
    ],
)
def test_rejects_invalid_analysis_settings(
    client: TestClient,
    data: dict[str, str],
    detail: str,
) -> None:
    response = client.post(
        "/api/analyze",
        files={"file": ("signal.csv", b"0\n1\n0", "text/csv")},
        data=data,
    )
    assert response.status_code == 422
    assert detail in response.json()["detail"]


def test_serves_built_dashboard(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<h1>BioSQI dashboard</h1>", encoding="utf-8")
    client = TestClient(create_app(static_dir=tmp_path))

    response = client.get("/")
    assert response.status_code == 200
    assert "BioSQI dashboard" in response.text


def test_missing_export_returns_not_found(client: TestClient) -> None:
    response = client.get("/api/analyses/missing/export")
    assert response.status_code == 404


def test_numpy_reader_rejects_corrupt_and_high_dimensional_data() -> None:
    with pytest.raises(ValueError, match="could not be read"):
        _numpy_signal(b"not a numpy file")

    stream = io.BytesIO()
    np.save(stream, np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="one- or two-dimensional"):
        _numpy_signal(stream.getvalue())


def test_csv_reader_handles_tsv_and_rejects_bad_inputs() -> None:
    values = _csv_signal(b"time\tsignal\n0\t1\n1\t10\n2\t2\n")
    np.testing.assert_array_equal(values, [1, 10, 2])

    with pytest.raises(ValueError, match="UTF-8"):
        _csv_signal(b"\xff\xfe")
    with pytest.raises(ValueError, match="at least two"):
        _csv_signal(b"1")


def test_wfdb_reader_requires_header() -> None:
    with pytest.raises(ValueError, match="header"):
        _wfdb_signal([("record.dat", b"")])


def test_analysis_rejects_invalid_signal_and_interpolates_for_display() -> None:
    with pytest.raises(ValueError, match="two samples"):
        analyze_signal(
            np.asarray([1.0]),
            filename="short.npy",
            sample_rate=100,
            profile="ecg",
            window_seconds=2,
            overlap=0.5,
            threshold=0.5,
        )
    with pytest.raises(ValueError, match="Unknown profile"):
        analyze_signal(
            np.arange(100),
            filename="bad.npy",
            sample_rate=100,
            profile="not-a-profile",
            window_seconds=2,
            overlap=0.5,
            threshold=0.5,
        )
    with pytest.raises(ValueError, match="finite samples"):
        analyze_signal(
            np.full(100, np.nan),
            filename="nan.npy",
            sample_rate=100,
            profile="ecg",
            window_seconds=2,
            overlap=0.5,
            threshold=0.5,
        )

    samples = np.sin(np.arange(500) / 20)
    samples[20] = np.nan
    result = analyze_signal(
        samples,
        filename="missing.npy",
        sample_rate=100,
        profile="ecg",
        window_seconds=2,
        overlap=0.5,
        threshold=0.5,
    )
    assert "visualization only" in result["warnings"][0]


def test_analysis_store_evicts_oldest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(_remember.__globals__, "MAX_STORED_ANALYSES", 2)
    ANALYSES.clear()
    for identifier in ("one", "two", "three"):
        _remember({"id": identifier})
    assert list(ANALYSES) == ["two", "three"]
