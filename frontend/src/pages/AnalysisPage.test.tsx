import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes, useNavigate } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { getAnalysisReport, getUserTrends } from "../api/analysis";
import { I18nProvider, useTranslation } from "../i18n";
import { AnalysisPage } from "./AnalysisPage";
import type { AnalysisReportResponse } from "../types/analysis";

vi.mock("../api/analysis", () => ({
  getAnalysisReport: vi.fn(),
  getUserTrends: vi.fn(),
}));

vi.mock("../components/AnalysisStatusPolling", () => ({
  AnalysisStatusPolling: ({
    analysisId,
    onCompleted,
  }: {
    analysisId: string;
    onCompleted: (status: unknown) => void;
  }) => {
    setTimeout(() => onCompleted({ analysis_id: analysisId, status: "completed" }), 0);
    return <div data-testid="mock-analysis-polling" />;
  },
}));

vi.mock("../components/AnalysisReport", () => ({
  AnalysisReport: ({
    report,
    trendData,
  }: {
    report: AnalysisReportResponse;
    trendData?: unknown;
  }) => (
    <div
      data-testid="analysis-report"
      data-analysis-id={report.analysis_id}
      data-has-trends={trendData ? "true" : "false"}
    />
  ),
}));

const report: AnalysisReportResponse = {
  analysis_id: "abc",
  user_id: "user-1",
  created_at: "2025-01-01T00:00:00Z",
  status: "completed",
  video_metadata: {},
  quality_check: {},
  analysis_metadata: {},
  swing_phases: [],
  biomechanics: null,
  metric_evaluations: [],
  improvements: [],
  drill_recommendations: [],
  overlay_video_url: null,
  trend_data: null,
};

function AnalysisRoutes() {
  const navigate = useNavigate();
  return (
    <>
      <button type="button" onClick={() => navigate("/analyses/def")}>
        open def
      </button>
      <Routes>
        <Route path="/analyses/:analysisId" element={<AnalysisPage />} />
      </Routes>
    </>
  );
}

function LanguageControl() {
  const { setLanguage } = useTranslation();
  return (
    <button type="button" onClick={() => setLanguage("en")}>
      switch to English
    </button>
  );
}

describe("AnalysisPage", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders the report when trend fetch fails", async () => {
    vi.mocked(getAnalysisReport).mockResolvedValue(report);
    vi.mocked(getUserTrends).mockRejectedValue(new Error("trend unavailable"));

    render(
      <MemoryRouter initialEntries={["/analyses/abc"]}>
        <Routes>
          <Route path="/analyses/:analysisId" element={<AnalysisPage />} />
        </Routes>
      </MemoryRouter>,
    );

    await waitFor(() => {
      expect(screen.getByTestId("analysis-report")).toHaveAttribute(
        "data-analysis-id",
        "abc",
      );
    });

    expect(screen.getByTestId("analysis-report")).toHaveAttribute(
      "data-has-trends",
      "false",
    );
    expect(screen.queryByTestId("analysis-page-error")).not.toBeInTheDocument();
  });

  it("ignores a report response from the previous route", async () => {
    let resolveOldReport: (value: AnalysisReportResponse) => void = () => undefined;
    vi.mocked(getAnalysisReport).mockImplementation((analysisId) => {
      if (analysisId === "abc") {
        return new Promise((resolve) => {
          resolveOldReport = resolve;
        });
      }
      return Promise.resolve({ ...report, analysis_id: analysisId });
    });
    vi.mocked(getUserTrends).mockResolvedValue({
      metrics_history: {},
      total_recordings: 0,
    });
    const user = userEvent.setup();

    render(
      <MemoryRouter initialEntries={["/analyses/abc"]}>
        <AnalysisRoutes />
      </MemoryRouter>,
    );

    await waitFor(() => expect(getAnalysisReport).toHaveBeenCalledWith("abc", "ko"));
    await user.click(screen.getByRole("button", { name: "open def" }));
    await waitFor(() => {
      expect(screen.getByTestId("analysis-report")).toHaveAttribute(
        "data-analysis-id",
        "def",
      );
    });

    await act(async () => {
      resolveOldReport(report);
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(screen.getByTestId("analysis-report")).toHaveAttribute(
      "data-analysis-id",
      "def",
    );
  });

  it("reloads a completed report when the locale changes", async () => {
    const languageStore = new Map<string, string>([["myswing.language", "ko"]]);
    Object.defineProperty(window, "localStorage", {
      configurable: true,
      value: {
        getItem: (key: string) => languageStore.get(key) ?? null,
        setItem: (key: string, value: string) => languageStore.set(key, value),
      },
    });
    vi.mocked(getAnalysisReport).mockResolvedValue(report);
    vi.mocked(getUserTrends).mockResolvedValue({
      metrics_history: {},
      total_recordings: 0,
    });
    const user = userEvent.setup();

    render(
      <I18nProvider>
        <MemoryRouter initialEntries={["/analyses/abc"]}>
          <LanguageControl />
          <Routes>
            <Route path="/analyses/:analysisId" element={<AnalysisPage />} />
          </Routes>
        </MemoryRouter>
      </I18nProvider>,
    );

    await waitFor(() => expect(getAnalysisReport).toHaveBeenCalledWith("abc", "ko"));
    await user.click(screen.getByRole("button", { name: "switch to English" }));
    await waitFor(() => expect(getAnalysisReport).toHaveBeenCalledWith("abc", "en"));
  });
});
