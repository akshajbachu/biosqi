import type { AnalysisWindow } from "../types";

export function WindowEvidence({ window }: { window: AnalysisWindow }) {
  return (
    <section className="panel evidence-panel" aria-labelledby="evidence-heading">
      <div className="panel-heading compact">
        <div>
          <p className="eyebrow">Selected window · {window.start.toFixed(1)}–{window.end.toFixed(1)}s</p>
          <h2 id="evidence-heading">Indicator breakdown</h2>
        </div>
        <div className={`score-orb ${window.quality}`}>
          <strong>{Math.round(window.score * 100)}</strong>
          <span>/ 100</span>
        </div>
      </div>
      <div className="indicator-list">
        {window.indicators.map((indicator) => (
          <article key={indicator.id} className="indicator">
            <div className="indicator-title">
              <span className={`status-pip ${indicator.status}`} aria-hidden="true" />
              <div>
                <strong>{indicator.label}</strong>
                <small>{indicator.description}</small>
              </div>
              <span>{Math.round(indicator.normalized * 100)}%</span>
            </div>
            <div className="indicator-track" aria-label={`${indicator.label} ${Math.round(indicator.normalized * 100)} percent`}>
              <span
                className={indicator.status}
                style={{ width: `${Math.max(indicator.normalized * 100, 2)}%` }}
              />
            </div>
            <div className="indicator-meta">
              <span>Raw {indicator.value === null ? "unavailable" : Number(indicator.value.toPrecision(3))}</span>
              <span>Model coefficient {indicator.weight.toFixed(2)}</span>
              <span>Contribution {indicator.contribution >= 0 ? "+" : ""}{(indicator.contribution * 100).toFixed(1)}</span>
            </div>
          </article>
        ))}
      </div>
      <div className="artifact-section">
        <h3>Likely artifact attribution</h3>
        {window.artifacts.length ? (
          window.artifacts.map((artifact) => (
            <div key={artifact.label} className="artifact-row">
              <span className="artifact-swatch" style={{ background: artifact.color }} />
              <span>{artifact.label}</span>
              <div className="artifact-track">
                <span style={{ width: `${artifact.contribution * 100}%`, background: artifact.color }} />
              </div>
              <strong>{Math.round(artifact.confidence * 100)}%</strong>
            </div>
          ))
        ) : (
          <p className="quiet">No dominant artifact signature in this window.</p>
        )}
        <p className="explanation-note">
          Attribution is diagnostic evidence, not a clinical determination.
        </p>
      </div>
    </section>
  );
}
