import { render, screen, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it } from "vitest";
import { I18nProvider } from "../i18n";
import type { BiomechanicsResponse } from "../types/analysis";
import { BiomechanicsMeasurements } from "./BiomechanicsMeasurements";

const availableBiomechanics: BiomechanicsResponse = {
  bat_speed: { speed_kmh: 120.5, precision: 1 },
  attack_angle: { angle_degrees: 12.3, precision: 0.5 },
  hand_path_efficiency: 0.85,
  stride_length_cm: 42.1,
  cog_sway_cm: 8.2,
  cog_drop_cm: 4.3,
  head_stability_cm: 3.4,
  front_knee_extension_degrees: 164.2,
  front_knee_flexion_degrees: 28.6,
  spine_angle_degrees: -7.5,
  unmeasurable_metrics: [],
  processing_time_seconds: 10.2,
};

const originalLocalStorage = Object.getOwnPropertyDescriptor(window, "localStorage");

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

describe("BiomechanicsMeasurements", () => {
  it("renders every available projected value without presenting resolution as accuracy", () => {
    renderInLanguage(
      <BiomechanicsMeasurements
        biomechanics={availableBiomechanics}
        phaseSource="mixed_pose_and_observed_bat_contact"
      />,
      "en",
    );

    const panel = screen.getByTestId("biomechanics-measurements");
    expect(panel).toHaveTextContent("monocular 2D projected diagnostics");
    expect(panel).toHaveTextContent("not validated professional-reference grades");
    expect(screen.getAllByTestId(/^biomechanics-measurement-/)).toHaveLength(10);
    expect(screen.getByTestId("biomechanics-measurement-bat_speed")).toHaveTextContent(
      "120.5 km/h",
    );
    expect(screen.getByTestId("biomechanics-measurement-attack_angle")).toHaveTextContent(
      "+12.3°",
    );
    expect(
      screen.getByTestId("biomechanics-measurement-hand_path_efficiency"),
    ).toHaveTextContent("Hand-path straightness (proxy)");
    expect(
      screen.getByTestId("biomechanics-measurement-hand_path_efficiency"),
    ).toHaveTextContent("85.0%");
    expect(panel).toHaveTextContent("Reporting resolution: 1.0 km/h");
    expect(panel).not.toHaveTextContent(/measurement accuracy/i);
    expect(screen.getByTestId("biomechanics-phase-source")).toHaveTextContent(
      "Pose-derived phases + detector-observed bat contact",
    );
  });

  it("renders an explicit empty state for a legacy report with no measurements", () => {
    renderInLanguage(<BiomechanicsMeasurements biomechanics={null} />, "en");

    expect(screen.getByTestId("biomechanics-measurements-empty")).toHaveTextContent(
      "No projected biomechanics values could be reported from this video.",
    );
    expect(screen.getByTestId("biomechanics-phase-source")).toHaveTextContent(
      "Not recorded (legacy analysis)",
    );
    expect(screen.queryByTestId("biomechanics-unmeasurable")).not.toBeInTheDocument();
  });

  it("renders each unmeasurable metric and its stored reason", () => {
    const unavailableBiomechanics: BiomechanicsResponse = {
      ...availableBiomechanics,
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
        { metric_name: "bat_speed", reason: "Observed bat line unavailable" },
        { metric_name: "impact_anchor", reason: "Impact was not observed" },
      ],
    };

    renderInLanguage(
      <BiomechanicsMeasurements
        biomechanics={unavailableBiomechanics}
        phaseSource="unavailable"
      />,
      "en",
    );

    const reasons = screen.getByTestId("biomechanics-unmeasurable");
    expect(within(reasons).getByText(/Observed bat line unavailable/)).toBeInTheDocument();
    expect(within(reasons).getByText(/Impact was not observed/)).toBeInTheDocument();
    expect(screen.getByTestId("biomechanics-phase-source")).toHaveTextContent(
      "Unavailable",
    );
  });

  it("localizes unmeasurable metric identifiers outside the projected cards", () => {
    renderInLanguage(
      <BiomechanicsMeasurements
        biomechanics={{
          ...availableBiomechanics,
          unmeasurable_metrics: [
            { metric_name: "kinematic_chain", reason: "reason one" },
            {
              metric_name: "hip_shoulder_separation",
              reason: "reason two",
            },
          ],
        }}
      />,
      "ko",
    );

    const reasons = screen.getByTestId("biomechanics-unmeasurable");
    expect(reasons).toHaveTextContent("키네마틱 체인: reason one");
    expect(reasons).toHaveTextContent("힙-숄더 분리: reason two");
  });

  it("provides the same accuracy caveat in Korean", () => {
    renderInLanguage(
      <BiomechanicsMeasurements
        biomechanics={availableBiomechanics}
        phaseSource="pose_classifier"
      />,
      "ko",
    );

    expect(screen.getByText("투영 생체역학 진단값")).toBeInTheDocument();
    expect(screen.getByTestId("biomechanics-measurements")).toHaveTextContent(
      "검증된 프로 기준 등급이나 실제 3D 측정값이 아닙니다",
    );
    expect(screen.getByTestId("biomechanics-phase-source")).toHaveTextContent(
      "포즈 기반 분류기",
    );
  });

  it("labels the pose-motion contact provenance in Korean", () => {
    renderInLanguage(
      <BiomechanicsMeasurements
        biomechanics={null}
        phaseSource="mixed_pose_classifier_and_pose_motion_contact"
      />,
      "ko",
    );
    expect(screen.getByTestId("biomechanics-phase-source")).toHaveTextContent(
      "포즈 기반 구간 + 포즈 동작 기반 접촉 추정",
    );
    expect(
      screen.getByTestId("biomechanics-phase-source"),
    ).not.toHaveTextContent("mixed_pose_classifier_and_pose_motion_contact");
  });

  it("labels the pose-motion contact provenance in English", () => {
    renderInLanguage(
      <BiomechanicsMeasurements
        biomechanics={null}
        phaseSource="mixed_pose_classifier_and_pose_motion_contact"
      />,
      "en",
    );
    expect(screen.getByTestId("biomechanics-phase-source")).toHaveTextContent(
      "Pose-derived phases + pose-motion contact estimate",
    );
  });

  it("humanizes unknown phase sources instead of leaking snake_case", () => {
    renderInLanguage(
      <BiomechanicsMeasurements
        biomechanics={null}
        phaseSource="some_future_source"
      />,
      "en",
    );
    const source = screen.getByTestId("biomechanics-phase-source");
    expect(source).toHaveTextContent("some future source");
    expect(source).not.toHaveTextContent("some_future_source");
  });
});
