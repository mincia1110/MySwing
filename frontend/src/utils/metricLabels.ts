/**
 * Centralized display labels for raw API metric identifiers.
 *
 * Raw `metric_name` values (e.g. "kinematic_chain") stay the stable keys for
 * data-testid/React keys; this module maps the known identifiers to localized
 * labels and humanizes unknown identifiers instead of leaking raw snake_case.
 */
import { useCallback } from "react";
import { useTranslation } from "../i18n";

/** Known raw metric identifiers -> i18n label keys (metricLabels.*). */
export const METRIC_LABEL_KEYS: Record<string, string> = {
  bat_speed: "metricLabels.batSpeed",
  attack_angle: "metricLabels.attackAngle",
  hand_path_efficiency: "metricLabels.handPathEfficiency",
  kinematic_chain: "metricLabels.kinematicChain",
  hip_shoulder_separation: "metricLabels.hipShoulderSeparation",
};

const DIRECT_METRIC_LABELS: Record<"ko" | "en", Record<string, string>> = {
  ko: {
    all: "전체 메트릭",
    impact_anchor: "임팩트 기준점",
  },
  en: {
    all: "All Metrics",
    impact_anchor: "Impact Anchor",
  },
} as const;

/** Turn an unknown snake_case identifier into readable text. */
export function humanizeMetricName(metricName: string): string {
  return metricName.replace(/_/g, " ").trim();
}

/**
 * Returns a resolver that maps a raw metric identifier to a localized label,
 * falling back to humanized text for unknown identifiers.
 */
export function useMetricLabel(): (metricName: string) => string {
  const { language, t } = useTranslation();
  return useCallback(
    (metricName: string) => {
      const directLabel = DIRECT_METRIC_LABELS[language][metricName];
      if (directLabel) return directLabel;

      const key = METRIC_LABEL_KEYS[metricName];
      return key ? t(key) : humanizeMetricName(metricName);
    },
    [language, t],
  );
}
