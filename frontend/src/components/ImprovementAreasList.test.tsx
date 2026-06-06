import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ImprovementAreasList } from "./ImprovementAreasList";
import type { ImprovementAreaResponse } from "../types/analysis";

const improvements: ImprovementAreaResponse[] = [
  {
    metric_name: "attack_angle",
    deviation_percent: 50,
    current_value: 22,
    target_range_min: 5,
    target_range_max: 15,
    rank: 2,
  },
  {
    metric_name: "bat_speed",
    deviation_percent: 80,
    current_value: 70,
    target_range_min: 100,
    target_range_max: 120,
    rank: 1,
  },
  {
    metric_name: "hip_rotation",
    deviation_percent: 30,
    current_value: 60,
    target_range_min: 80,
    target_range_max: 100,
    rank: 3,
  },
];

describe("ImprovementAreasList", () => {
  it("renders items in rank order", () => {
    render(<ImprovementAreasList improvements={improvements} />);

    const items = screen.getAllByTestId(/^improvements-item-/);
    expect(items).toHaveLength(3);
    expect(items[0]).toHaveAttribute("data-rank", "1");
    expect(items[1]).toHaveAttribute("data-rank", "2");
    expect(items[2]).toHaveAttribute("data-rank", "3");
    expect(items[0]).toHaveTextContent("bat_speed");
  });

  it("shows deviation values per item", () => {
    render(<ImprovementAreasList improvements={improvements} />);
    expect(
      screen.getByTestId("improvements-deviation-bat_speed"),
    ).toHaveTextContent("80.0%");
    expect(
      screen.getByTestId("improvements-deviation-attack_angle"),
    ).toHaveTextContent("50.0%");
  });

  it("renders empty state when no improvements provided", () => {
    render(<ImprovementAreasList improvements={[]} />);
    expect(screen.getByTestId("improvements-empty")).toBeInTheDocument();
  });
});
