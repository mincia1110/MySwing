import { render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it } from "vitest";
import { I18nProvider } from "../i18n";
import type { BiomechanicsResponse } from "../types/analysis";
import { BiomechanicsMeasurements } from "./BiomechanicsMeasurements";

const originalLocalStorage = Object.getOwnPropertyDescriptor(window, "localStorage");

function renderInLanguage(component: ReactNode, language: "ko" | "en") {
  Object.defineProperty(window, "localStorage", {
    configurable: true,
    value: {
      getItem: (key: string) => (key === "myswing.language" ? language : null),
      setItem: () => undefined,
    },
  });
  return render(<I18nProvider>{component}</I18nProvider>);
}

function unavailableBiomechanics(reasons: [string, string]): BiomechanicsResponse {
  return {
    bat_speed: null,
    attack_angle: null,
    hand_path_efficiency: null,
    stride_length_cm: null,
    cog_sway_cm: null,
    cog_drop_cm: null,
    head_stability_cm: null,
    front_knee_extension_degrees: null,
    front_knee_flexion_degrees: null,
    spine_angle_degrees: null,
    unmeasurable_metrics: [
      { metric_name: "all", reason: reasons[0] },
      { metric_name: "impact_anchor", reason: reasons[1] },
    ],
    processing_time_seconds: null,
  };
}

afterEach(() => {
  if (originalLocalStorage) {
    Object.defineProperty(window, "localStorage", originalLocalStorage);
  } else {
    Reflect.deleteProperty(window, "localStorage");
  }
});

describe("BiomechanicsMeasurements abstention labels", () => {
  it("shows confirmed metric identifiers with Korean labels", () => {
    renderInLanguage(
      <BiomechanicsMeasurements
        biomechanics={unavailableBiomechanics([
          "사용 가능한 포즈 데이터가 없습니다.",
          "검출기로 관측한 배트 근거나 명시적인 임팩트 구간을 사용할 수 없습니다.",
        ])}
      />,
      "ko",
    );

    const panel = screen.getByTestId("biomechanics-unmeasurable");
    expect(panel).toHaveTextContent(
      "전체 메트릭: 사용 가능한 포즈 데이터가 없습니다.",
    );
    expect(panel).toHaveTextContent(
      "임팩트 기준점: 검출기로 관측한 배트 근거나 명시적인 임팩트 구간을 사용할 수 없습니다.",
    );
    expect(panel).not.toHaveTextContent("impact anchor");
  });

  it("keeps the confirmed metric labels in English for English reports", () => {
    renderInLanguage(
      <BiomechanicsMeasurements
        biomechanics={unavailableBiomechanics([
          "No pose data available",
          "No detector-observed bat evidence or explicit impact phase was available",
        ])}
      />,
      "en",
    );

    const panel = screen.getByTestId("biomechanics-unmeasurable");
    expect(panel).toHaveTextContent("All Metrics: No pose data available");
    expect(panel).toHaveTextContent(
      "Impact Anchor: No detector-observed bat evidence or explicit impact phase was available",
    );
  });
});
