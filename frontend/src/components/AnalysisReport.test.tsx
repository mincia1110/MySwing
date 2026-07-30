import { render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it } from "vitest";
import { I18nProvider } from "../i18n";
import { AnalysisReport } from "./AnalysisReport";
import type {
  AnalysisReportResponse,
  TrendDataResponse,
} from "../types/analysis";

const originalLocalStorage = Object.getOwnPropertyDescriptor(
  window,
  "localStorage",
);

function renderInLanguage(component: ReactNode, language: "ko" | "en") {
  let storedLanguage = language;
  Object.defineProperty(window, "localStorage", {
    configurable: true,
    value: {
      getItem: (key: string) =>
        key === "myswing.language" ? storedLanguage : null,
      setItem: (key: string, value: string) => {
        if (key === "myswing.language") storedLanguage = value as "ko" | "en";
      },
    },
  });
  return render(<I18nProvider>{component}</I18nProvider>);
}

afterEach(() => {
  if (originalLocalStorage) {
    Object.defineProperty(window, "localStorage", originalLocalStorage);
  } else {
    Reflect.deleteProperty(window, "localStorage");
  }
});

const baseReport: AnalysisReportResponse = {
  analysis_id: "abc-123",
  user_id: "user-1",
  created_at: "2025-01-15T10:30:00Z",
  status: "completed",
  video_metadata: {},
  quality_check: {
    brightness_status: "pass",
    framing_status: "warning",
    resolution_status: "pass",
    frame_rate_stability_status: "pass",
    brightness_value: 55,
    swing_arc_visibility_percent: 72.5,
    frame_rate_variation_percent: 2.1,
    warnings: ["스윙 아크가 부분적으로 가려졌습니다"],
  },
  analysis_metadata: {},
  swing_phases: [
    { phase: "stance", start_frame: 0, end_frame: 6, duration_ms: 200 },
    { phase: "load", start_frame: 6, end_frame: 16, duration_ms: 333 },
  ],
  biomechanics: null,
  metric_evaluations: [
    {
      metric_name: "bat_speed",
      measured_value: 110,
      unit: "km/h",
      reference_min: 100,
      reference_max: 120,
      deviation_percent: 0,
      rating: "within_range",
      color_code: "green",
    },
  ],
  improvements: [
    {
      metric_name: "attack_angle",
      deviation_percent: 50,
      current_value: 22,
      target_range_min: 5,
      target_range_max: 15,
      rank: 1,
    },
  ],
  drill_recommendations: [
    {
      drill_name: "티 타격",
      target_metric: "attack_angle",
      description: "티 위에서 발사각 연습",
      direction: "above",
    },
    {
      drill_name: "허리 회전",
      target_metric: "hip_rotation",
      description: "허리 회전 속도 향상 드릴",
      direction: "below",
    },
  ],
  overlay_video_url: "https://example.com/overlay.mp4",
  trend_data: null,
};

describe("AnalysisReport", () => {
  it("renders all sub-components with report data", () => {
    render(<AnalysisReport report={baseReport} />);

    expect(screen.getByTestId("analysis-report")).toHaveAttribute(
      "data-analysis-id",
      "abc-123",
    );
    expect(screen.getByTestId("overlay-video")).toBeInTheDocument();
    expect(screen.getByTestId("quality-check")).toBeInTheDocument();
    expect(screen.getByTestId("analysis-report-hero")).toBeInTheDocument();
    expect(screen.getByTestId("analysis-report-insights")).toBeInTheDocument();
    expect(screen.getByTestId("biomechanics-measurements")).toBeInTheDocument();
    expect(screen.getByTestId("biomechanics-measurements-empty")).toBeInTheDocument();
    expect(screen.getByTestId("quality-check-framing")).toHaveAttribute(
      "data-status",
      "warning",
    );
    expect(screen.getByTestId("metrics-table")).toBeInTheDocument();
    expect(
      screen.getByTestId("metrics-table-row-bat_speed"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("improvements")).toBeInTheDocument();
    expect(
      screen.getByTestId("improvements-item-attack_angle"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("drill-card-티 타격"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("drill-card-허리 회전"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("drill-card-티 타격"),
    ).toHaveTextContent("기준 초과");
    expect(
      screen.getByTestId("drill-card-허리 회전"),
    ).toHaveTextContent("기준 미달");
    expect(screen.getByTestId("comparison-view")).toBeInTheDocument();
  });

  it("renders empty drill state when no drills are provided", () => {
    render(
      <AnalysisReport report={{ ...baseReport, drill_recommendations: [] }} />,
    );
    expect(
      screen.getByTestId("analysis-report-drills-empty"),
    ).toBeInTheDocument();
  });

  it("renders TrendChart when trendData is provided", () => {
    const trendData: TrendDataResponse = {
      metrics_history: {
        bat_speed: [
          {
            analysis_id: "a1",
            recorded_at: "2025-01-01T00:00:00Z",
            value: 100,
            rating: "within_range",
          },
          {
            analysis_id: "a2",
            recorded_at: "2025-01-02T00:00:00Z",
            value: 110,
            rating: "within_range",
          },
        ],
      },
      total_recordings: 2,
    };
    render(<AnalysisReport report={baseReport} trendData={trendData} />);
    expect(screen.getByTestId("trend-chart")).toHaveAttribute(
      "data-state",
      "ready",
    );
  });

  it("does not render TrendChart when trend data is missing", () => {
    render(<AnalysisReport report={baseReport} />);
    expect(screen.queryByTestId("trend-chart")).not.toBeInTheDocument();
  });

  it("passes raw biomechanics and phase provenance to the diagnostics panel", () => {
    render(
      <AnalysisReport
        report={{
          ...baseReport,
          phase_source: "observed_bat_contact_only",
          phase_evidence: { observed_non_predicted_bat_lines: 4 },
          biomechanics: {
            bat_speed: { speed_kmh: 98.4, precision: 1 },
            attack_angle: { angle_degrees: -6.2, precision: 0.5 },
            hand_path_efficiency: null,
            stride_length_cm: null,
            cog_sway_cm: null,
            cog_drop_cm: null,
            head_stability_cm: null,
            front_knee_extension_degrees: null,
            front_knee_flexion_degrees: null,
            spine_angle_degrees: null,
            unmeasurable_metrics: [
              { metric_name: "stride_length_cm", reason: "Pose unavailable" },
            ],
            processing_time_seconds: 4.1,
          },
        }}
      />,
    );

    expect(screen.getByTestId("biomechanics-measurement-bat_speed")).toHaveTextContent(
      "98.4 km/h",
    );
    expect(screen.getByTestId("biomechanics-measurement-attack_angle")).toHaveTextContent(
      "-6.2°",
    );
    expect(screen.getByTestId("biomechanics-phase-source")).toHaveTextContent(
      "검출기가 관측한 배트 접촉만 사용",
    );
    expect(screen.getByTestId("biomechanics-unmeasurable")).toHaveTextContent(
      "Pose unavailable",
    );
  });

  it("uses report.trend_data when no explicit trendData prop given", () => {
    const trendData: TrendDataResponse = {
      metrics_history: {},
      total_recordings: 1,
      message: "분석 기록이 부족합니다.",
    };
    render(
      <AnalysisReport report={{ ...baseReport, trend_data: trendData }} />,
    );
    expect(screen.getByTestId("trend-chart")).toHaveAttribute(
      "data-state",
      "insufficient",
    );
  });

  it("shows a localized notice when multiple swings were isolated", () => {
    render(
      <AnalysisReport
        report={{
          ...baseReport,
          phase_source: "mixed_pose_classifier_and_pose_motion_contact",
          phase_evidence: {
            swing_window: {
              start_frame: 0,
              end_frame: 44,
              selected_impact_frame: 15,
              candidate_count: 2,
              multiple_swing_detected: true,
              isolation_applied: true,
              selection_policy: "earliest_substantial_swing",
            },
          },
        }}
      />,
    );

    const notice = screen.getByTestId("analysis-report-swing-window-notice");
    expect(notice).toHaveTextContent(
      "영상에서 스윙 동작 후보 2개가 감지되어",
    );
    expect(notice).toHaveTextContent(
      "가장 먼저 시작된 유의미한 스윙만 분석했습니다",
    );
  });

  it("shows the multiple-swing notice in English with the candidate count", () => {
    renderInLanguage(
      <AnalysisReport
        report={{
          ...baseReport,
          phase_evidence: {
            swing_window: {
              candidate_count: 3,
              multiple_swing_detected: true,
              isolation_applied: true,
            },
          },
        }}
      />,
      "en",
    );

    expect(
      screen.getByTestId("analysis-report-swing-window-notice"),
    ).toHaveTextContent(
      "3 swing-motion candidates were detected in this video; only the earliest substantial swing was analyzed.",
    );
  });

  it("omits the count when candidate_count is missing or invalid", () => {
    render(
      <AnalysisReport
        report={{
          ...baseReport,
          phase_evidence: {
            swing_window: {
              multiple_swing_detected: true,
              isolation_applied: true,
            },
          },
        }}
      />,
    );

    const notice = screen.getByTestId("analysis-report-swing-window-notice");
    expect(notice).toHaveTextContent("여러 번의 스윙 동작이 감지되어");
    expect(notice).not.toHaveTextContent("후보");
  });

  it("renders no notice without multiple-swing isolation evidence", () => {
    render(
      <AnalysisReport
        report={{
          ...baseReport,
          phase_evidence: {
            swing_window: {
              candidate_count: 1,
              multiple_swing_detected: false,
              isolation_applied: false,
            },
          },
        }}
      />,
    );
    expect(
      screen.queryByTestId("analysis-report-swing-window-notice"),
    ).not.toBeInTheDocument();

    render(<AnalysisReport report={baseReport} />);
    expect(
      screen.queryByTestId("analysis-report-swing-window-notice"),
    ).not.toBeInTheDocument();
  });

  it("formats the creation date with the selected UI locale", () => {
    renderInLanguage(<AnalysisReport report={baseReport} />, "en");
    expect(screen.getByTestId("analysis-report")).toHaveTextContent(
      "1/15/2025",
    );

    renderInLanguage(<AnalysisReport report={baseReport} />, "ko");
    const meta = document.querySelectorAll(".analysis-report__meta")[1];
    expect(meta).toHaveTextContent("2025.");
    expect(meta.textContent).toMatch(/오전|오후/);
  });
});
