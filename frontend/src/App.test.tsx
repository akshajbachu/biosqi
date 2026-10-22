import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import { analysisFixture } from "./test/fixtures";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("BioSQI dashboard", () => {
  it("explains the empty state and local privacy boundary", () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("API offline"));
    render(<App />);

    expect(screen.getByRole("heading", { name: /know which moments/i })).toBeInTheDocument();
    expect(screen.getByText(/your quality map will appear here/i)).toBeInTheDocument();
    expect(screen.getByText(/your data never leaves this device/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /choose or drop/i })).toBeInTheDocument();
  });

  it("uploads a signal and renders an explainable analysis", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch");
    fetchMock.mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.endsWith("/api/profiles")) {
        return new Response(JSON.stringify({ profiles: [] }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      if (url.endsWith("/api/analyze") && init?.method === "POST") {
        expect(init.body).toBeInstanceOf(FormData);
        return new Response(JSON.stringify(analysisFixture), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      return new Response(null, { status: 404 });
    });
    const user = userEvent.setup();
    const { container } = render(<App />);
    const input = container.querySelector<HTMLInputElement>('input[type="file"]');
    expect(input).not.toBeNull();

    await user.upload(input!, new File(["value\n0\n1\n0"], "demo.csv", { type: "text/csv" }));
    await user.click(screen.getByRole("button", { name: /analyze recording/i }));

    await waitFor(() => {
      expect(screen.getByRole("heading", { name: "demo.csv" })).toBeInTheDocument();
    });
    expect(screen.getByText("Overall quality")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /waveform quality map/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /indicator breakdown/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /power spectral density/i })).toBeInTheDocument();

    const timeline = screen.getByRole("group", { name: "Analysis windows" });
    await user.click(within(timeline).getByRole("button", { name: /window 2/i }));
    expect(screen.getByText("Motion or muscle noise")).toBeInTheDocument();
  });

  it("shows a useful validation error when no file is selected", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("API offline"));
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: /analyze recording/i }));
    expect(screen.getByRole("alert")).toHaveTextContent(/choose a csv/i);
  });
});
