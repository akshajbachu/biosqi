import type { AnalysisResult, AnalyzeOptions, Profile } from "../types";

async function parseResponse<T>(response: Response): Promise<T> {
  if (response.ok) return response.json() as Promise<T>;
  let message = `Request failed (${response.status})`;
  try {
    const payload = (await response.json()) as { detail?: string };
    if (payload.detail) message = payload.detail;
  } catch {
    // Keep the status-based message when the server did not return JSON.
  }
  throw new Error(message);
}

export async function getProfiles(signal?: AbortSignal): Promise<Profile[]> {
  const response = await fetch("/api/profiles", { signal });
  const payload = await parseResponse<{ profiles: Profile[] } | Profile[]>(response);
  return Array.isArray(payload) ? payload : payload.profiles;
}

export async function analyzeFiles(
  files: File[],
  options: AnalyzeOptions,
): Promise<AnalysisResult> {
  const data = new FormData();
  files.forEach((file) => data.append("files", file));
  data.set("profile", options.profile);
  data.set("window_seconds", String(options.windowSeconds));
  data.set("overlap", String(options.overlap));
  data.set("threshold", String(options.threshold));
  data.set("sample_rate", String(options.sampleRate));

  const response = await fetch("/api/analyze", { method: "POST", body: data });
  return parseResponse<AnalysisResult>(response);
}

export async function downloadExport(
  analysisId: string,
  format: "csv" | "json",
): Promise<void> {
  const response = await fetch(`/api/analyses/${analysisId}/export?format=${format}`);
  if (!response.ok) {
    await parseResponse(response);
    return;
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download =
    response.headers.get("content-disposition")?.match(/filename="?([^"]+)/)?.[1] ??
    `biosqi-analysis.${format}`;
  anchor.click();
  URL.revokeObjectURL(url);
}
