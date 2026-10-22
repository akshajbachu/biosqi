import { useMemo, useState } from "react";

import type { AnalysisWindow, SeriesPoint } from "../types";

interface WaveformChartProps {
  points: SeriesPoint[];
  windows: AnalysisWindow[];
  selectedIndex: number;
  onSelect: (index: number) => void;
}

const WIDTH = 1000;
const HEIGHT = 250;
const TOP = 18;
const BOTTOM = 34;

export function WaveformChart({
  points,
  windows,
  selectedIndex,
  onSelect,
}: WaveformChartProps) {
  const [zoom, setZoom] = useState(1);
  const duration = Math.max(points.at(-1)?.x ?? 1, 1);
  const selected = windows[selectedIndex];
  const viewportDuration = duration / zoom;
  const center = selected ? (selected.start + selected.end) / 2 : duration / 2;
  const viewStart = Math.max(0, Math.min(duration - viewportDuration, center - viewportDuration / 2));
  const viewEnd = viewStart + viewportDuration;

  const { path, min, max } = useMemo(() => {
    const visible = points.filter((point) => point.x >= viewStart && point.x <= viewEnd);
    const source = visible.length > 1 ? visible : points;
    const values = source.map((point) => point.y);
    const low = Math.min(...values);
    const high = Math.max(...values);
    const range = high - low || 1;
    const line = source
      .map((point, index) => {
        const x = ((point.x - viewStart) / (viewEnd - viewStart || 1)) * WIDTH;
        const y = TOP + ((high - point.y) / range) * (HEIGHT - TOP - BOTTOM);
        return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(" ");
    return { path: line, min: low, max: high };
  }, [points, viewEnd, viewStart]);

  const mapX = (seconds: number) => ((seconds - viewStart) / viewportDuration) * WIDTH;

  return (
    <section className="panel chart-panel" aria-labelledby="waveform-heading">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Signal trace</p>
          <h2 id="waveform-heading">Waveform quality map</h2>
        </div>
        <div className="zoom-controls" aria-label="Waveform zoom controls">
          <button type="button" onClick={() => setZoom((value) => Math.max(1, value / 2))} aria-label="Zoom out">-</button>
          <span>{zoom}×</span>
          <button type="button" onClick={() => setZoom((value) => Math.min(8, value * 2))} aria-label="Zoom in">+</button>
          <button type="button" onClick={() => setZoom(1)}>Reset</button>
        </div>
      </div>
      <div className="chart-wrap">
        <svg
          className="waveform"
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          role="img"
          aria-label={`Waveform from ${viewStart.toFixed(1)} to ${viewEnd.toFixed(1)} seconds with quality overlays`}
          preserveAspectRatio="none"
        >
          <defs>
            <pattern id="minorGrid" width="20" height="20" patternUnits="userSpaceOnUse">
              <path d="M 20 0 L 0 0 0 20" fill="none" stroke="currentColor" strokeOpacity=".08" />
            </pattern>
          </defs>
          <rect width={WIDTH} height={HEIGHT - BOTTOM} fill="url(#minorGrid)" />
          {windows.map((window) => {
            const start = Math.max(window.start, viewStart);
            const end = Math.min(window.end, viewEnd);
            if (end <= start) return null;
            const x = mapX(start);
            const width = mapX(end) - x;
            return (
              <rect
                key={window.index}
                x={x}
                y={0}
                width={width}
                height={HEIGHT - BOTTOM}
                className={`quality-overlay ${window.quality} ${window.index === selectedIndex ? "selected" : ""}`}
                onClick={() => onSelect(window.index)}
                role="button"
                tabIndex={0}
                aria-label={`Select window ${window.index + 1}, ${window.quality}, score ${Math.round(window.score * 100)} percent`}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") onSelect(window.index);
                }}
              />
            );
          })}
          <path d={path} className="waveform-line halo" />
          <path d={path} className="waveform-line" />
          <line x1="0" x2={WIDTH} y1={HEIGHT - BOTTOM} y2={HEIGHT - BOTTOM} className="chart-axis" />
          <text x="8" y={HEIGHT - 9} className="chart-label">{viewStart.toFixed(1)}s</text>
          <text x={WIDTH - 8} y={HEIGHT - 9} textAnchor="end" className="chart-label">{viewEnd.toFixed(1)}s</text>
          <text x="8" y="15" className="chart-label">{max.toFixed(2)}</text>
          <text x="8" y={HEIGHT - BOTTOM - 6} className="chart-label">{min.toFixed(2)}</text>
        </svg>
      </div>
      <div className="legend" aria-label="Quality legend">
        <span><i className="legend-dot good" />Good</span>
        <span><i className="legend-dot borderline" />Borderline</span>
        <span><i className="legend-dot poor" />Poor</span>
        <span className="legend-hint">Click a region to inspect its evidence</span>
      </div>
    </section>
  );
}
