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

    for (const m of metrics) {
      const row = screen.getByTestId(`metrics-table-row-${m.metric_name}`);
      expect(row).toBeInTheDocument();
      expect(row).toHaveAttribute("data-color", m.color_code);
      expect(row).toHaveAttribute("data-rating", m.rating);
      expect(within(row).getByText(m.metric_name)).toBeInTheDocument();
    }
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
