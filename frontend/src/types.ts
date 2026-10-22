export type QualityClass = "good" | "poor" | "borderline" | "unavailable";

export interface Profile {
  id: string;
  name: string;
  modality: string;
  description: string;
  default_window_seconds: number;
  default_overlap: number;
  default_threshold: number;
}

export interface Indicator {
  id: string;
  label: string;
  value: number | null;
  normalized: number;
  weight: number;
  contribution: number;
  status: QualityClass;
  description: string;
}

export interface Artifact {
  id: string;
  label: string;
  contribution: number;
  confidence: number;
  color: string;
}

export interface AnalysisWindow {
  index: number;
  start: number;
  end: number;
  score: number;
  raw_score: number;
  quality: QualityClass;
  indicators: Indicator[];
  artifacts: Artifact[];
  psd: SeriesPoint[];
}

export interface SeriesPoint {
  x: number;
  y: number;
}

export interface AnalysisSummary {
  overall_score: number;
  usable_percent: number;
  good_windows: number;
  poor_windows: number;
  borderline_windows: number;
  total_windows: number;
  duration_seconds: number;
}

export interface AnalysisResult {
  id: string;
  filename: string;
  profile: string;
  sample_rate: number;
  channel: string;
  model: string;
  generated_at: string;
  threshold: number;
  waveform: SeriesPoint[];
  psd: SeriesPoint[];
  windows: AnalysisWindow[];
  summary: AnalysisSummary;
  warnings: string[];
}

export interface AnalyzeOptions {
  profile: string;
  windowSeconds: number;
  overlap: number;
  threshold: number;
  sampleRate: number;
}
