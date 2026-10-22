# Quickstart

## Install

```bash
pip install "biosqi[data,ui]"
```

The base package supports NumPy and delimited-text inputs. The `data` extra
adds WFDB and EDF/BDF readers; `ui` adds the local FastAPI dashboard.

## Open the dashboard

```bash
biosqi view
```

The browser communicates only with a server bound to localhost. Uploaded
recordings are processed in memory and are not sent to an external service.

## Score from the command line

```bash
biosqi score recording.csv \
  --sample-rate 250 \
  --signal-type ecg \
  --window 10 \
  --threshold 0.70 \
  --output result.json
```

Generate a self-contained report or export the usability mask:

```bash
biosqi report recording.npz --signal-type ppg --output report.html
biosqi mask recording.npz --signal-type ppg --output mask.csv
```

## Python

```python
from biosqi.analysis import analyze
from biosqi.io import read_recording
from biosqi.types import SignalType

recording = read_recording(
    "recording.npz",
    signal_type=SignalType.ECG,
)
result = analyze(recording, threshold=0.70)

print(result.scores)
print(result.mask)
```

For plain `.npy`, `.csv`, `.tsv`, or `.txt` inputs, provide the sample rate.
NPZ archives may include `data`, `sample_rate`, and `channel_names` arrays.
