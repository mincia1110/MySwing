import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { AnalysisReport } from "./AnalysisReport";
import type {
  AnalysisReportResponse,
  TrendDataResponse,
} from "../types/analysis";

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
});
