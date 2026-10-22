import type { AnalysisWindow } from "../types";

interface ScoreTimelineProps {
  windows: AnalysisWindow[];
  threshold: number;
  selectedIndex: number;
  onSelect: (index: number) => void;
}

export function ScoreTimeline({
  windows,
  threshold,
  selectedIndex,
  onSelect,
}: ScoreTimelineProps) {
  return (
    <section className="panel timeline-panel" aria-labelledby="timeline-heading">
      <div className="panel-heading compact">
        <div>
          <p className="eyebrow">Window confidence</p>
          <h2 id="timeline-heading">Quality score timeline</h2>
        </div>
        <span className="threshold-label">Threshold {Math.round(threshold * 100)}%</span>
      </div>
      <div className="timeline" role="group" aria-label="Analysis windows">
        <div
          className="threshold-line"
          style={{ bottom: `${threshold * 100}%` }}
          aria-hidden="true"
        />
        {windows.map((window) => (
          <button
            key={window.index}
            type="button"
            className={`score-bar ${window.quality} ${selectedIndex === window.index ? "selected" : ""}`}
            style={{ height: `${Math.max(window.score * 100, 4)}%` }}
            onClick={() => onSelect(window.index)}
            title={`${window.start.toFixed(1)}·${window.end.toFixed(1)}s · ${Math.round(window.score * 100)}%`}
            aria-label={`Window ${window.index + 1}: ${Math.round(window.score * 100)} percent`}
            aria-current={selectedIndex === window.index ? "true" : undefined}
          >
            <span className="visually-hidden">{Math.round(window.score * 100)}%</span>
          </button>
        ))}
      </div>
      <div className="timeline-axis">
        <span>0s</span>
        <span>{windows.at(-1)?.end.toFixed(1) ?? 0}s</span>
      </div>
    </section>
  );
}
