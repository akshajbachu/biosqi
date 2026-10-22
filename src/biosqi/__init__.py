"""biosqi: transparent biomedical signal-quality assessment."""

from biosqi.analysis import analyze
from biosqi.fusion import FusionModel, fit_fusion_model, load_default_model
from biosqi.io import read_recording
from biosqi.types import AnalysisResult, Recording, SignalType

__all__ = [
    "AnalysisResult",
    "FusionModel",
    "Recording",
    "SignalType",
    "analyze",
    "fit_fusion_model",
    "load_default_model",
    "read_recording",
]

__version__ = "0.1.0"
