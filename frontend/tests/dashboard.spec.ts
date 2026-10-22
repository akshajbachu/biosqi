import { expect, test } from "@playwright/test";

test("loads the private local dashboard", async ({ page }) => {
  await page.goto("/");

  await expect(page).toHaveTitle(/BioSQI/i);
  await expect(page.getByRole("heading", { name: /Know which moments/i })).toBeVisible();
  await expect(page.getByRole("button", { name: /Choose or drop/i })).toBeVisible();
  await expect(page.getByText(/Your data never leaves this device/i)).toBeVisible();
  await expect(page.getByText(/Your quality map will appear here/i)).toBeVisible();
});

test("adapts the setup panel to a phone viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");

  await expect(page.getByRole("heading", { name: /Load recording/i })).toBeVisible();
  await expect(page.getByLabel("Quality profile")).toBeVisible();
  await expect(page.getByRole("button", { name: /Analyze recording/i })).toBeVisible();
});

test("uploads NPZ and renders a selectable quality map", async ({ page }) => {
  await page.goto("/");
  await page.locator('input[type="file"]').setInputFiles("../examples/demo_ecg.npz");
  await page.getByRole("button", { name: /Analyze recording/i }).click();

  await expect(page.getByText("Analysis complete")).toBeVisible();
  await expect(page.getByText(/250 Hz/)).toBeVisible();
  await expect(page.getByRole("heading", { name: "Waveform quality map" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Quality score timeline" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Indicator breakdown" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Power spectral density" })).toBeVisible();

  const windows = page.getByRole("group", { name: "Analysis windows" }).getByRole("button");
  await expect(windows).toHaveCount(7);
  await windows.nth(1).click();
  await expect(windows.nth(1)).toHaveAttribute("aria-current", "true");
});
