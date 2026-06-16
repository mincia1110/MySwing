import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { getAnalysisReport, getUserTrends } from "../api/analysis";
import { AnalysisPage } from "./AnalysisPage";
import type { AnalysisReportResponse } from "../types/analysis";

vi.mock("../api/analysis", () => ({
  getAnalysisReport: vi.fn(),
  getUserTrends: vi.fn(),
}));

vi.mock("../components/AnalysisStatusPolling", () => ({
  AnalysisStatusPolling: ({ onCompleted }: { onCompleted: (status: unknown) => void }) => {
    setTimeout(() => onCompleted({ analysis_id: "abc", status: "completed" }), 0);
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
});
