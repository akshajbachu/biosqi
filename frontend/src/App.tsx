import { FormEvent, useEffect, useMemo, useState } from "react";

import { PsdChart } from "./components/PsdChart";
import { ScoreTimeline } from "./components/ScoreTimeline";
import { UploadZone } from "./components/UploadZone";
import { WaveformChart } from "./components/WaveformChart";
import { WindowEvidence } from "./components/WindowEvidence";
import { analyzeFiles, downloadExport, getProfiles } from "./lib/api";
import type { AnalysisResult, Profile } from "./types";

const FALLBACK_PROFILES: Profile[] = [
  {
    id: "ecg",
    name: "ECG · General",
    modality: "ECG",
    description: "Balanced morphology, baseline, amplitude, and spectral checks.",
    default_window_seconds: 10,
    default_overlap: 0.5,
    default_threshold: 0.65,
  },
  {
    id: "ppg",
    name: "PPG · General",
    modality: "PPG",
    description: "Pulse-shape and motion-aware assessment for optical recordings.",
    default_window_seconds: 8,
    default_overlap: 0.5,
    default_threshold: 0.62,
  },
  {
    id: "eeg",
    name: "EEG · General",
    modality: "EEG",
    description: "Amplitude, flatline, line-noise, and high-frequency artifact checks.",
    default_window_seconds: 4,
    default_overlap: 0.5,
    default_threshold: 0.6,
  },
];

type RequestState = "idle" | "analyzing" | "error";

function App() {
  const [profiles, setProfiles] = useState<Profile[]>(FALLBACK_PROFILES);
  const [profileId, setProfileId] = useState("ecg");
  const [files, setFiles] = useState<File[]>([]);
  const [windowSeconds, setWindowSeconds] = useState(10);
  const [overlap, setOverlap] = useState(0.5);
  const [threshold, setThreshold] = useState(0.65);
  const [sampleRate, setSampleRate] = useState(250);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [requestState, setRequestState] = useState<RequestState>("idle");
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    getProfiles(controller.signal)
      .then((items) => {
        if (items.length) setProfiles(items);
      })
      .catch(() => {
        // Built-in profiles keep the dashboard usable while the local API starts.
      });
    return () => controller.abort();
  }, []);

  const selectedProfile = useMemo(
    () => profiles.find((profile) => profile.id === profileId) ?? profiles[0],
    [profileId, profiles],
  );
  const selectedWindow = result?.windows[selectedIndex] ?? result?.windows[0];

  function applyProfile(id: string) {
    setProfileId(id);
    const profile = profiles.find((candidate) => candidate.id === id);
    if (profile) {
      setWindowSeconds(profile.default_window_seconds);
      setOverlap(profile.default_overlap);
      setThreshold(profile.default_threshold);
    }
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!files.length) {
      setError("Choose a CSV, NPY/NPZ, or WFDB recording before analyzing.");
      setRequestState("error");
      return;
    }
    setError("");
    setRequestState("analyzing");
    try {
      const analysis = await analyzeFiles(files, {
        profile: profileId,
        windowSeconds,
        overlap,
        threshold,
        sampleRate,
      });
      setResult(analysis);
      setSelectedIndex(0);
      setRequestState("idle");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The analysis could not be completed.");
      setRequestState("error");
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="BioSQI home">
          <span className="brand-mark" aria-hidden="true">
            <i />
            <i />
            <i />
          </span>
          <span>
            <strong>biosqi</strong>
            <small>signal quality intelligence</small>
          </span>
        </a>
        <div className="local-badge">
          <span className="pulse-dot" />
          Local session
          <span className="privacy-copy">· Your data never leaves this device</span>
        </div>
      </header>

      <main>
        <section className="hero">
          <div>
            <p className="eyebrow">Transparent signal quality assessment</p>
            <h1>Know which moments<br />you can trust.</h1>
            <p>
              Inspect biomedical recordings window by window, trace every score back to
              its evidence, and export reproducible quality decisions.
            </p>
          </div>
          <div className="privacy-card">
          <span aria-hidden="true">Local</span>
            <div>
              <strong>Private by design</strong>
              <p>Analysis runs on your local BioSQI server. No account, cloud upload, or telemetry.</p>
            </div>
          </div>
        </section>

        <section className="workspace-grid">
          <form className="control-panel panel" onSubmit={submit}>
            <div className="panel-heading">
              <div>
                <p className="step-label">01</p>
                <h2>Load recording</h2>
              </div>
              <span className="format-note">CSV · NPY/NPZ · WFDB</span>
            </div>
            <UploadZone files={files} onFiles={setFiles} disabled={requestState === "analyzing"} />

            <div className="section-divider" />
            <div className="panel-heading">
              <div>
                <p className="step-label">02</p>
                <h2>Analysis setup</h2>
              </div>
            </div>

            <label className="field">
              <span>Quality profile</span>
              <select value={profileId} onChange={(event) => applyProfile(event.target.value)}>
                {profiles.map((profile) => (
                  <option key={profile.id} value={profile.id}>{profile.name}</option>
                ))}
              </select>
              <small>{selectedProfile?.description}</small>
            </label>

            <div className="field-pair">
              <label className="field">
                <span>Window</span>
                <div className="input-unit">
                  <input
                    type="number"
                    min="1"
                    max="120"
                    step="0.5"
                    value={windowSeconds}
                    onChange={(event) => setWindowSeconds(Number(event.target.value))}
                  />
                  <i>sec</i>
                </div>
              </label>
              <label className="field">
                <span>Sample rate</span>
                <div className="input-unit">
                  <input
                    type="number"
                    min="1"
                    max="100000"
                    value={sampleRate}
                    onChange={(event) => setSampleRate(Number(event.target.value))}
                  />
                  <i>Hz</i>
                </div>
              </label>
            </div>

            <label className="range-field">
              <span><b>Overlap</b><output>{Math.round(overlap * 100)}%</output></span>
              <input
                type="range"
                min="0"
                max="0.9"
                step="0.05"
                value={overlap}
                onChange={(event) => setOverlap(Number(event.target.value))}
              />
            </label>
            <label className="range-field">
              <span><b>Quality threshold</b><output>{Math.round(threshold * 100)}%</output></span>
              <input
                type="range"
                min="0.3"
                max="0.95"
                step="0.01"
                value={threshold}
                onChange={(event) => setThreshold(Number(event.target.value))}
              />
              <small>Windows below this score are flagged for review.</small>
            </label>

            {error && (
              <div className="error-banner" role="alert">
                <span aria-hidden="true">!</span>
                <div><strong>Analysis unavailable</strong><p>{error}</p></div>
              </div>
            )}
            <button className="primary-button" type="submit" disabled={requestState === "analyzing"}>
              {requestState === "analyzing" ? <><i className="spinner" />Analyzing locally...</> : <>Analyze recording <span aria-hidden="true">&rarr;</span></>}
            </button>
          </form>

          <div className="results-column" aria-live="polite" aria-busy={requestState === "analyzing"}>
            {requestState === "analyzing" ? (
              <LoadingState />
            ) : result ? (
              <>
                <section className="results-header">
                  <div>
                    <p className="eyebrow">Analysis complete</p>
                    <h2>{result.filename}</h2>
                    <p>{result.profile.toUpperCase()} · {result.sample_rate} Hz · {result.summary.duration_seconds.toFixed(1)} seconds</p>
                  </div>
                  <div className="export-actions">
                    <button type="button" onClick={() => downloadExport(result.id, "csv")}>Export CSV</button>
                    <button type="button" onClick={() => downloadExport(result.id, "json")}>Export JSON</button>
                  </div>
                </section>
                <Summary result={result} />
                {result.warnings.map((warning) => (
                  <div key={warning} className="warning-banner">{warning}</div>
                ))}
                <WaveformChart
                  points={result.waveform}
                  windows={result.windows}
                  selectedIndex={selectedIndex}
                  onSelect={setSelectedIndex}
                />
                <ScoreTimeline
                  windows={result.windows}
                  threshold={result.threshold}
                  selectedIndex={selectedIndex}
                  onSelect={setSelectedIndex}
                />
                <div className="detail-grid">
                  {selectedWindow && <WindowEvidence window={selectedWindow} />}
                  <PsdChart points={selectedWindow?.psd ?? result.psd} />
                </div>
              </>
            ) : (
              <EmptyState />
            )}
          </div>
        </section>
      </main>
      <footer>
        <span>BioSQI is a research tool, not a medical device.</span>
        <span>Local API · <code>/api/health</code></span>
      </footer>
    </div>
  );
}

function Summary({ result }: { result: AnalysisResult }) {
  const { summary } = result;
  const kpis = [
    { label: "Overall quality", value: `${Math.round(summary.overall_score * 100)}`, suffix: "/100", tone: summary.overall_score >= result.threshold ? "good" : "poor" },
    { label: "Usable signal", value: `${Math.round(summary.usable_percent)}`, suffix: "%", tone: "neutral" },
    { label: "Good windows", value: `${summary.good_windows}`, suffix: `/${summary.total_windows}`, tone: "good" },
    { label: "Needs review", value: `${summary.poor_windows + summary.borderline_windows}`, suffix: `/${summary.total_windows}`, tone: summary.poor_windows ? "poor" : "neutral" },
  ];
  return (
    <section className="kpi-grid" aria-label="Analysis summary">
      {kpis.map((kpi) => (
        <article key={kpi.label} className={`kpi-card ${kpi.tone}`}>
          <span>{kpi.label}</span>
          <strong>{kpi.value}<small>{kpi.suffix}</small></strong>
        </article>
      ))}
    </section>
  );
}

function EmptyState() {
  return (
    <section className="empty-state panel">
      <div className="empty-visual" aria-hidden="true">
        <svg viewBox="0 0 420 130">
          <path d="M0 69 C28 68 34 65 53 68 S78 73 96 67 L111 66 L119 22 L130 111 L142 52 L152 69 C190 71 204 65 229 68 S261 72 282 67 L298 66 L307 35 L317 97 L328 57 L338 69 C363 68 385 67 420 68" />
        </svg>
      </div>
      <span className="empty-number">03</span>
      <h2>Your quality map will appear here</h2>
      <p>Load a recording and choose a profile to see signal quality, evidence, artifacts, and spectral context.</p>
      <div className="empty-features">
        <span>&check; Window-level scoring</span>
        <span>&check; Explainable indicators</span>
        <span>&check; Reproducible exports</span>
      </div>
    </section>
  );
}

function LoadingState() {
  return (
    <section className="loading-state panel" role="status">
      <div className="analysis-loader">
        <i /><i /><i /><i /><i />
      </div>
      <h2>Reading your signal</h2>
      <p>Calculating window indicators, quality scores, and spectral evidence locally.</p>
      <div className="skeleton-row"><span /><span /><span /><span /></div>
    </section>
  );
}

export default App;
