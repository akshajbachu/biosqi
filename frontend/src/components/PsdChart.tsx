import { useMemo } from "react";

import type { SeriesPoint } from "../types";

const WIDTH = 600;
const HEIGHT = 220;
const PAD_X = 42;
const PAD_Y = 22;

export function PsdChart({ points }: { points: SeriesPoint[] }) {
  const { path, maxFrequency } = useMemo(() => {
    if (!points.length) return { path: "", maxFrequency: 0 };
    const powers = points.map((point) => Math.log10(Math.max(point.y, 1e-12)));
    const min = Math.min(...powers);
    const max = Math.max(...powers);
    const maxFreq = points.at(-1)?.x ?? 1;
    const line = points
      .map((point, index) => {
        const x = PAD_X + (point.x / maxFreq) * (WIDTH - PAD_X * 1.35);
        const power = Math.log10(Math.max(point.y, 1e-12));
        const y = PAD_Y + ((max - power) / (max - min || 1)) * (HEIGHT - PAD_Y * 2);
        return `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
    return { path: line, maxFrequency: maxFreq };
  }, [points]);

  return (
    <section className="panel psd-panel" aria-labelledby="psd-heading">
      <div className="panel-heading compact">
        <div>
          <p className="eyebrow">Frequency domain</p>
          <h2 id="psd-heading">Power spectral density</h2>
        </div>
        <span className="unit-chip">log power / Hz</span>
      </div>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="psd-chart" role="img" aria-label="Power spectral density plot">
        <defs>
          <linearGradient id="psdFill" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0" stopColor="#2e6f5d" stopOpacity=".28" />
            <stop offset="1" stopColor="#2e6f5d" stopOpacity="0" />
          </linearGradient>
        </defs>
        {[0, 1, 2, 3, 4].map((index) => {
          const y = PAD_Y + (index / 4) * (HEIGHT - PAD_Y * 2);
          return <line key={index} x1={PAD_X} x2={WIDTH - 12} y1={y} y2={y} className="psd-grid" />;
        })}
        <path d={`${path} L ${WIDTH - 12},${HEIGHT - PAD_Y} L ${PAD_X},${HEIGHT - PAD_Y} Z`} fill="url(#psdFill)" />
        <path d={path} className="psd-line" />
        <text x={PAD_X} y={HEIGHT - 3} className="chart-label">0 Hz</text>
        <text x={WIDTH - 12} y={HEIGHT - 3} textAnchor="end" className="chart-label">{maxFrequency.toFixed(0)} Hz</text>
      </svg>
    </section>
  );
}
