import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { TrendChart } from "./TrendChart";
import type { TrendDataResponse } from "../types/analysis";

function makePoint(
  recordedAt: string,
  value: number,
  rating: "below_range" | "within_range" | "above_range" = "within_range",
) {
  return {
    analysis_id: `a-${recordedAt}`,
    recorded_at: recordedAt,
    value,
    rating,
  };
}

describe("TrendChart", () => {
  it("shows the minimum-recordings message when total_recordings < 2", () => {
    const data: TrendDataResponse = {
      metrics_history: {},
      total_recordings: 1,
      message: "최소 2회 분석이 필요합니다.",
    };
    render(<TrendChart trendData={data} />);

    expect(screen.getByTestId("trend-chart")).toHaveAttribute(
      "data-state",
      "insufficient",
    );
    expect(screen.getByTestId("trend-chart-message")).toHaveTextContent(
      "최소 2회 분석이 필요합니다.",
    );
  });

  it("falls back to a default message when no message is provided", () => {
    const data: TrendDataResponse = {
      metrics_history: {},
      total_recordings: 0,
    };
    render(<TrendChart trendData={data} />);

    expect(screen.getByTestId("trend-chart-message")).toHaveTextContent(
      /최소 2회 이상의 분석 기록이 필요/,
    );
  });

  it("renders a chart per metric when total_recordings >= 2", () => {
    const data: TrendDataResponse = {
      metrics_history: {
        bat_speed: [
          makePoint("2025-01-01T00:00:00Z", 100, "below_range"),
          makePoint("2025-01-02T00:00:00Z", 110, "within_range"),
          makePoint("2025-01-03T00:00:00Z", 115, "within_range"),
        ],
        launch_angle: [
          makePoint("2025-01-01T00:00:00Z", 22, "above_range"),
          makePoint("2025-01-02T00:00:00Z", 18, "above_range"),
        ],
      },
      total_recordings: 3,
    };
    render(<TrendChart trendData={data} />);

    expect(screen.getByTestId("trend-chart")).toHaveAttribute(
      "data-state",
      "ready",
    );
    expect(
      screen.getByTestId("trend-chart-metric-bat_speed"),
    ).toHaveAttribute("data-points", "3");
    expect(
      screen.getByTestId("trend-chart-metric-launch_angle"),
    ).toHaveAttribute("data-points", "2");
  });

  it("limits to the most recent 30 points", () => {
    const points = Array.from({ length: 40 }, (_, i) =>
      makePoint(
        new Date(2025, 0, i + 1).toISOString(),
        90 + i,
        "within_range",
      ),
    );
    const data: TrendDataResponse = {
      metrics_history: { bat_speed: points },
      total_recordings: 40,
    };
    render(<TrendChart trendData={data} />);
    expect(
      screen.getByTestId("trend-chart-metric-bat_speed"),
    ).toHaveAttribute("data-points", "30");
  });
});
