import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { QualityCheckResult } from "./QualityCheckResult";
import type { QualityCheckResponse } from "../types/video";

const passResult: QualityCheckResponse = {
  brightness_status: "pass",
  framing_status: "pass",
  resolution_status: "pass",
  frame_rate_stability_status: "pass",
  brightness_value: 60,
  swing_arc_visibility_percent: 92,
  frame_rate_variation_percent: 2.1,
  warnings: [],
};

const warningResult: QualityCheckResponse = {
  brightness_status: "warning",
  framing_status: "pass",
  resolution_status: "warning",
  frame_rate_stability_status: "warning",
  brightness_value: 30,
  swing_arc_visibility_percent: 88,
  frame_rate_variation_percent: 15.4,
  warnings: ["조명이 부족합니다", "해상도가 720p 미만입니다"],
};

describe("QualityCheckResult", () => {
  it("renders pass status for all checks", () => {
    render(<QualityCheckResult result={passResult} />);

    for (const key of ["brightness", "framing", "resolution", "fps"]) {
      const item = screen.getByTestId(`quality-check-${key}`);
      expect(item).toHaveAttribute("data-status", "pass");
    }
    expect(screen.queryByTestId("quality-check-warnings")).not.toBeInTheDocument();
  });

  it("renders warning status and warning messages", () => {
    render(<QualityCheckResult result={warningResult} />);

    expect(
      screen.getByTestId("quality-check-brightness"),
    ).toHaveAttribute("data-status", "warning");
    expect(
      screen.getByTestId("quality-check-resolution"),
    ).toHaveAttribute("data-status", "warning");
    expect(
      screen.getByTestId("quality-check-fps"),
    ).toHaveAttribute("data-status", "warning");
    expect(
      screen.getByTestId("quality-check-framing"),
    ).toHaveAttribute("data-status", "pass");

    const warnings = screen.getByTestId("quality-check-warnings");
    expect(warnings).toHaveTextContent("조명이 부족합니다");
    expect(warnings).toHaveTextContent("해상도가 720p 미만입니다");
  });

  it("displays measured values for each check", () => {
    render(<QualityCheckResult result={warningResult} />);
    expect(screen.getByTestId("quality-check-brightness")).toHaveTextContent(
      "30 lux",
    );
    expect(screen.getByTestId("quality-check-framing")).toHaveTextContent("88%");
    expect(screen.getByTestId("quality-check-fps")).toHaveTextContent(
      "편차 15.4%",
    );
  });
});
