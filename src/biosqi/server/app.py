"""FastAPI application for the private, local BioSQI dashboard."""

from __future__ import annotations

import csv
import io
import json
import tempfile
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import numpy.typing as npt
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .analysis import analyze_signal, core_available
from .profiles import available_profiles

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_STORED_ANALYSES = 20

AnalysisStore = OrderedDict[str, dict[str, Any]]
ANALYSES: AnalysisStore = OrderedDict()


def _safe_filename(upload: UploadFile) -> str:
    return Path(upload.filename or "recording").name


def _numpy_signal(content: bytes) -> tuple[npt.NDArray[np.float64], float | None, str]:
    try:
        loaded = np.load(io.BytesIO(content), allow_pickle=False)
    except (ValueError, OSError) as error:
        raise ValueError("The NumPy file could not be read.") from error
    detected_rate: float | None = None
    channel = "Channel 1"
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if not loaded.files:
                raise ValueError("The NPZ archive contains no arrays.")
            key = "data" if "data" in loaded.files else loaded.files[0]
            array = np.asarray(loaded[key])
            for rate_key in ("sample_rate", "fs"):
                if rate_key in loaded.files:
                    detected_rate = float(np.asarray(loaded[rate_key]).reshape(-1)[0])
                    break
            if "channel_names" in loaded.files:
                names = np.asarray(loaded["channel_names"]).reshape(-1)
                if names.size:
                    channel = str(names[0])
        finally:
            loaded.close()
    else:
        array = np.asarray(loaded)
    array = np.asarray(array)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("The NPY array must use a numeric data type.")
    if array.ndim == 1:
        return array.astype(float), detected_rate, channel
    if array.ndim == 2:
        values = array[0, :] if array.shape[0] < array.shape[1] else array[:, 0]
        return np.asarray(values, dtype=float), detected_rate, channel
    raise ValueError("NumPy recordings must be one- or two-dimensional.")


def _csv_signal(content: bytes) -> npt.NDArray[np.float64]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("CSV files must be UTF-8 encoded.") from error
    first_line = next((line for line in text.splitlines() if line.strip()), "")
    delimiter = "\t" if "\t" in first_line and "," not in first_line else ","
    try:
        matrix = np.genfromtxt(
            io.StringIO(text),
            delimiter=delimiter,
            dtype=float,
            invalid_raise=False,
        )
    except ValueError as error:
        raise ValueError("The CSV file does not contain a readable numeric signal.") from error
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 0:
        matrix = matrix.reshape(1)
    if matrix.ndim == 1:
        values = matrix
    elif matrix.ndim == 2:
        matrix = matrix[~np.all(~np.isfinite(matrix), axis=1)]
        matrix = matrix[:, ~np.all(~np.isfinite(matrix), axis=0)]
        if not matrix.size:
            raise ValueError("The CSV file does not contain a numeric column.")
        variances = np.nanvar(matrix, axis=0)
        values = matrix[:, int(np.nanargmax(variances))]
    else:
        raise ValueError("The CSV file has an unsupported shape.")
    values = values[np.isfinite(values) | np.isnan(values)]
    if values.size < 2:
        raise ValueError("The CSV file must contain at least two numeric samples.")
    return np.asarray(values, dtype=np.float64)


def _wfdb_signal(
    files: list[tuple[str, bytes]],
) -> tuple[npt.NDArray[np.float64], float, str]:
    try:
        import wfdb
    except ImportError as error:
        raise ValueError(
            "WFDB support is optional. Install biosqi[data] to read .hea/.dat recordings."
        ) from error
    header = next((name for name, _ in files if name.lower().endswith(".hea")), None)
    if header is None:
        raise ValueError("Select both the WFDB .hea header and its companion data file.")
    with tempfile.TemporaryDirectory(prefix="biosqi-wfdb-") as directory:
        root = Path(directory)
        for name, content in files:
            (root / Path(name).name).write_bytes(content)
        record_path = root / Path(header).stem
        try:
            record = wfdb.rdrecord(str(record_path), physical=True)
        except Exception as error:
            raise ValueError(
                "The WFDB record could not be read; verify all companion files are selected."
            ) from error
        matrix = record.p_signal if record.p_signal is not None else record.d_signal
        if matrix is None:
            raise ValueError("The WFDB record does not contain signal samples.")
        signal = np.asarray(matrix, dtype=float)
        if signal.ndim == 2:
            signal = signal[:, 0]
        channel = record.sig_name[0] if getattr(record, "sig_name", None) else "Channel 1"
        return signal.reshape(-1), float(record.fs), str(channel)


async def _load_signal(
    uploads: list[UploadFile],
    requested_sample_rate: float,
) -> tuple[npt.NDArray[np.float64], float, str, str]:
    if not uploads:
        raise ValueError("Choose a CSV, NPY/NPZ, or WFDB recording.")
    loaded: list[tuple[str, bytes]] = []
    total = 0
    for upload in uploads:
        content = await upload.read()
        total += len(content)
        if total > MAX_UPLOAD_BYTES:
            raise ValueError("The selected files exceed the 100 MB local upload limit.")
        loaded.append((_safe_filename(upload), content))

    names = [name for name, _ in loaded]
    numpy_file = next(
        ((name, content) for name, content in loaded if name.lower().endswith((".npy", ".npz"))),
        None,
    )
    csv_file = next(
        ((name, content) for name, content in loaded if name.lower().endswith(".csv")),
        None,
    )
    if numpy_file is not None:
        signal, detected_rate, channel = _numpy_signal(numpy_file[1])
        return signal, detected_rate or requested_sample_rate, channel, numpy_file[0]
    if csv_file is not None:
        return _csv_signal(csv_file[1]), requested_sample_rate, "Channel 1", csv_file[0]
    if any(name.lower().endswith((".hea", ".dat")) for name in names):
        signal, detected_rate, channel = _wfdb_signal(loaded)
        return (
            signal,
            detected_rate,
            channel,
            next(name for name in names if name.lower().endswith(".hea")),
        )
    raise ValueError("Unsupported file type. Use CSV, NPY/NPZ, or WFDB .hea/.dat files.")


def _remember(result: dict[str, Any]) -> None:
    ANALYSES[result["id"]] = result
    ANALYSES.move_to_end(result["id"])
    while len(ANALYSES) > MAX_STORED_ANALYSES:
        ANALYSES.popitem(last=False)


def _csv_export(result: dict[str, Any]) -> str:
    stream = io.StringIO()
    fieldnames = [
        "window",
        "start_seconds",
        "end_seconds",
        "score",
        "quality",
        "indicator",
        "indicator_value",
        "indicator_normalized",
        "indicator_weight",
        "indicator_contribution",
        "artifacts",
    ]
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for window in result["windows"]:
        artifacts = "; ".join(
            f"{item['label']} ({item['confidence']:.3f})" for item in window["artifacts"]
        )
        for indicator in window["indicators"]:
            writer.writerow(
                {
                    "window": window["index"] + 1,
                    "start_seconds": window["start"],
                    "end_seconds": window["end"],
                    "score": window["score"],
                    "quality": window["quality"],
                    "indicator": indicator["id"],
                    "indicator_value": indicator["value"],
                    "indicator_normalized": indicator["normalized"],
                    "indicator_weight": indicator["weight"],
                    "indicator_contribution": indicator["contribution"],
                    "artifacts": artifacts,
                }
            )
    return stream.getvalue()


def create_app(static_dir: Path | None = None) -> FastAPI:
    """Create the API and optionally mount the built dashboard."""
    application = FastAPI(
        title="BioSQI local dashboard",
        version="0.1.0",
        description="Local-only biomedical signal-quality analysis API.",
    )

    @application.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "biosqi",
            "local_only": True,
            "core_available": core_available(),
            "version": application.version,
        }

    @application.get("/api/profiles")
    def profiles() -> dict[str, Any]:
        return {"profiles": available_profiles()}

    @application.post("/api/analyze")
    async def analyze(
        files: Annotated[list[UploadFile] | None, File()] = None,
        file: Annotated[UploadFile | None, File()] = None,
        profile: Annotated[str, Form()] = "ecg",
        window_seconds: Annotated[float, Form()] = 10.0,
        overlap: Annotated[float, Form()] = 0.5,
        threshold: Annotated[float, Form()] = 0.65,
        sample_rate: Annotated[float, Form()] = 250.0,
    ) -> JSONResponse:
        if not 1.0 <= window_seconds <= 120.0:
            raise HTTPException(422, "window_seconds must be between 1 and 120.")
        if not 0.0 <= overlap < 1.0:
            raise HTTPException(422, "overlap must be at least 0 and below 1.")
        if not 0.0 <= threshold <= 1.0:
            raise HTTPException(422, "threshold must be between 0 and 1.")
        if not 0.1 <= sample_rate <= 100_000:
            raise HTTPException(422, "sample_rate must be between 0.1 and 100000 Hz.")
        uploads = list(files or [])
        if file is not None:
            uploads.append(file)
        try:
            signal, actual_rate, channel, filename = await _load_signal(uploads, sample_rate)
            result = analyze_signal(
                signal,
                filename=filename,
                sample_rate=actual_rate,
                profile=profile,
                window_seconds=window_seconds,
                overlap=overlap,
                threshold=threshold,
                channel=channel,
            )
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        result["id"] = uuid.uuid4().hex
        _remember(result)
        return JSONResponse(result)

    @application.get("/api/analyses/{analysis_id}/export")
    @application.get("/api/export/{analysis_id}")
    def export_analysis(
        analysis_id: str,
        format: Literal["csv", "json"] = "json",
    ) -> Response:
        result = ANALYSES.get(analysis_id)
        if result is None:
            raise HTTPException(404, "Analysis not found in this local session.")
        base = Path(str(result["filename"])).stem
        if format == "csv":
            return Response(
                _csv_export(result),
                media_type="text/csv; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="{base}-biosqi.csv"'},
            )
        return Response(
            json.dumps(result, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{base}-biosqi.json"'},
        )

    assets = static_dir if static_dir is not None else STATIC_DIR
    if (assets / "index.html").is_file():
        application.mount("/", StaticFiles(directory=assets, html=True), name="dashboard")
    else:

        @application.get("/")
        def root() -> dict[str, str]:
            return {
                "service": "biosqi",
                "message": "Dashboard assets are not built. Run npm run build in frontend/.",
            }

    return application


app = create_app()
