import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ComparisonView } from "./ComparisonView";
import type { SwingPhaseResponse } from "../types/analysis";

const userPhases: SwingPhaseResponse[] = [
  { phase: "stance", start_frame: 0, end_frame: 6, duration_ms: 200 },
  { phase: "load", start_frame: 6, end_frame: 16, duration_ms: 333 },
  { phase: "rotation", start_frame: 16, end_frame: 21, duration_ms: 167 },
];

describe("ComparisonView", () => {
  it("renders one row per phase merging user + reference phases", () => {
    render(
      <ComparisonView
        userPhases={userPhases}
        referencePhases={[
          { phase: "stance", duration_ms: 200 },
          { phase: "load", duration_ms: 300 },
          { phase: "rotation", duration_ms: 150 },
        ]}
      />,
    );

    expect(screen.getByTestId("comparison-row-stance")).toBeInTheDocument();
    expect(screen.getByTestId("comparison-row-load")).toBeInTheDocument();
    expect(screen.getByTestId("comparison-row-rotation")).toBeInTheDocument();
  });

  it("displays user and reference durations", () => {
    render(
      <ComparisonView
        userPhases={userPhases}
        referencePhases={[
          { phase: "stance", duration_ms: 200 },
          { phase: "load", duration_ms: 300 },
        ]}
      />,
    );

    expect(screen.getByTestId("comparison-user-stance")).toHaveTextContent(
      "200",
    );
    expect(screen.getByTestId("comparison-reference-stance")).toHaveTextContent(
      "200",
    );
    expect(screen.getByTestId("comparison-user-load")).toHaveTextContent("333");
    expect(screen.getByTestId("comparison-reference-load")).toHaveTextContent(
      "300",
    );
    // Phase only in user data should still render with reference dash.
    expect(
      screen.getByTestId("comparison-reference-rotation"),
    ).toHaveTextContent("-");
  });

  it("uses default reference set when none is supplied", () => {
    render(<ComparisonView userPhases={userPhases} />);
    expect(screen.getByTestId("comparison-row-impact")).toBeInTheDocument();
  });

  it("renders an empty state when both arrays are empty", () => {
    render(<ComparisonView userPhases={[]} referencePhases={[]} />);
    expect(screen.getByTestId("comparison-view-empty")).toBeInTheDocument();
  });
});
