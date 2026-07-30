import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MetricsTable } from "./MetricsTable";
import type { MetricEvaluationResponse } from "../types/analysis";

const metrics: MetricEvaluationResponse[] = [
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
  {
    metric_name: "attack_angle",
    measured_value: 22,
    unit: "deg",
    reference_min: 5,
    reference_max: 15,
    deviation_percent: 46.7,
    rating: "above_range",
    color_code: "red",
  },
  {
    metric_name: "hand_path_efficiency",
    measured_value: 0.85,
    unit: "",
    reference_min: 0.9,
    reference_max: 1.0,
    deviation_percent: -5.6,
    rating: "below_range",
    color_code: "yellow",
  },
];

describe("MetricsTable", () => {
  it("renders one row per metric with measured value, range, and deviation", () => {
    render(<MetricsTable metrics={metrics} />);

    const expectedLabels: Record<string, string> = {
      bat_speed: "배트 속도",
      attack_angle: "어택 앵글",
      hand_path_efficiency: "핸드 패스 효율",
    };
    for (const m of metrics) {
      const row = screen.getByTestId(`metrics-table-row-${m.metric_name}`);
      expect(row).toBeInTheDocument();
      expect(row).toHaveAttribute("data-color", m.color_code);
      expect(row).toHaveAttribute("data-rating", m.rating);
      expect(within(row).getByText(expectedLabels[m.metric_name])).toBeInTheDocument();
    }
  });

  it("localizes known raw metric identifiers and keeps raw test ids", () => {
    render(
      <MetricsTable
        metrics={[
          {
            metric_name: "kinematic_chain",
            measured_value: 0.8,
            unit: "",
            reference_min: 0.7,
            reference_max: 1.0,
            deviation_percent: 0,
            rating: "within_range",
            color_code: "green",
          },
          {
            metric_name: "hip_shoulder_separation",
            measured_value: 42,
            unit: "deg",
            reference_min: 30,
            reference_max: 45,
            deviation_percent: 0,
            rating: "within_range",
            color_code: "green",
          },
        ]}
      />,
    );

    expect(
      screen.getByTestId("metrics-table-row-kinematic_chain"),
    ).toHaveTextContent("키네마틱 체인");
    expect(
      screen.getByTestId("metrics-table-row-hip_shoulder_separation"),
    ).toHaveTextContent("힙-숄더 분리");
  });

  it("humanizes unknown metric identifiers instead of showing snake_case", () => {
    render(
      <MetricsTable
        metrics={[
          {
            metric_name: "new_future_metric",
            measured_value: 1,
            unit: "",
            reference_min: 0,
            reference_max: 2,
            deviation_percent: 0,
            rating: "within_range",
            color_code: "green",
          },
        ]}
      />,
    );

    const row = screen.getByTestId("metrics-table-row-new_future_metric");
    expect(within(row).getByText("new future metric")).toBeInTheDocument();
    expect(row).not.toHaveTextContent("new_future_metric");
  });

  it("applies the color-coded class to the rating badge", () => {
    render(<MetricsTable metrics={metrics} />);

    const greenBadge = screen.getByTestId("metrics-table-rating-bat_speed");
    expect(greenBadge.className).toContain("metrics-table__rating--green");

    const redBadge = screen.getByTestId("metrics-table-rating-attack_angle");
    expect(redBadge.className).toContain("metrics-table__rating--red");

    const yellowBadge = screen.getByTestId(
      "metrics-table-rating-hand_path_efficiency",
    );
    expect(yellowBadge.className).toContain("metrics-table__rating--yellow");
  });

  it("formats deviation with a sign and one decimal", () => {
    render(<MetricsTable metrics={metrics} />);
    const attackRow = screen.getByTestId("metrics-table-row-attack_angle");
    expect(attackRow).toHaveTextContent("+46.7%");

    const handRow = screen.getByTestId(
      "metrics-table-row-hand_path_efficiency",
    );
    expect(handRow).toHaveTextContent("-5.6%");
  });

  it("renders an empty state when no metrics are provided", () => {
    render(<MetricsTable metrics={[]} />);
    expect(screen.getByTestId("metrics-table-empty")).toBeInTheDocument();
  });
});
