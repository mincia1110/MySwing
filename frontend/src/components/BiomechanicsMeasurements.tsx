import type { BiomechanicsResponse } from "../types/analysis";
import { useTranslation } from "../i18n";
import { humanizeMetricName, useMetricLabel } from "../utils/metricLabels";
import "./BiomechanicsMeasurements.css";

export interface BiomechanicsMeasurementsProps {
  biomechanics?: BiomechanicsResponse | null;
  phaseSource?: string | null;
}

interface ProjectedMeasurement {
  key: string;
  labelKey: string;
  value: number;
  unit: string;
  signed?: boolean;
  reportingResolution?: number;
}

const PHASE_SOURCE_KEYS: Record<string, string> = {
  pose_classifier: "biomechanics.phaseSources.poseClassifier",
  partial_pose_classifier: "biomechanics.phaseSources.partialPoseClassifier",
  mixed_pose_and_observed_bat_contact:
    "biomechanics.phaseSources.mixedPoseAndObservedBatContact",
  mixed_pose_classifier_and_pose_motion_contact:
    "biomechanics.phaseSources.mixedPoseClassifierAndPoseMotionContact",
  pose_motion_contact_only: "biomechanics.phaseSources.poseMotionContactOnly",
  observed_bat_contact_only:
    "biomechanics.phaseSources.observedBatContactOnly",
  unavailable: "biomechanics.phaseSources.unavailable",
};

const METRIC_LABEL_KEYS: Record<string, string> = {
  bat_speed: "biomechanics.metrics.batSpeed",
  attack_angle: "biomechanics.metrics.attackAngle",
  hand_path_efficiency: "biomechanics.metrics.handPathEfficiency",
  stride_length_cm: "biomechanics.metrics.strideLength",
  cog_sway_cm: "biomechanics.metrics.cogSway",
  cog_drop_cm: "biomechanics.metrics.cogDrop",
  head_stability_cm: "biomechanics.metrics.headStability",
  front_knee_extension_degrees:
    "biomechanics.metrics.frontKneeExtension",
  front_knee_flexion_degrees: "biomechanics.metrics.frontKneeFlexion",
  spine_angle_degrees: "biomechanics.metrics.spineAngle",
};

function isFiniteNumber(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function formatNumber(value: number, signed = false): string {
  const normalized = Object.is(value, -0) ? 0 : value;
  const formatted = normalized.toFixed(1);
  return signed && normalized > 0 ? `+${formatted}` : formatted;
}

function buildMeasurements(
  biomechanics?: BiomechanicsResponse | null,
): ProjectedMeasurement[] {
  if (!biomechanics) return [];

  const measurements: ProjectedMeasurement[] = [];
  const batSpeed = biomechanics.bat_speed;
  const attackAngle = biomechanics.attack_angle;

  if (isFiniteNumber(batSpeed?.speed_kmh)) {
    measurements.push({
      key: "bat_speed",
      labelKey: METRIC_LABEL_KEYS.bat_speed,
      value: batSpeed.speed_kmh,
      unit: " km/h",
      reportingResolution: isFiniteNumber(batSpeed.precision)
        ? batSpeed.precision
        : undefined,
    });
  }
  if (isFiniteNumber(attackAngle?.angle_degrees)) {
    measurements.push({
      key: "attack_angle",
      labelKey: METRIC_LABEL_KEYS.attack_angle,
      value: attackAngle.angle_degrees,
      unit: "°",
      signed: true,
      reportingResolution: isFiniteNumber(attackAngle.precision)
        ? attackAngle.precision
        : undefined,
    });
  }
  if (isFiniteNumber(biomechanics.hand_path_efficiency)) {
    measurements.push({
      key: "hand_path_efficiency",
      labelKey: METRIC_LABEL_KEYS.hand_path_efficiency,
      value: biomechanics.hand_path_efficiency * 100,
      unit: "%",
    });
  }

  const scalarMeasurements: Array<
    [
      keyof Pick<
        BiomechanicsResponse,
        | "stride_length_cm"
        | "cog_sway_cm"
        | "cog_drop_cm"
        | "head_stability_cm"
        | "front_knee_extension_degrees"
        | "front_knee_flexion_degrees"
        | "spine_angle_degrees"
      >,
      string,
    ]
  > = [
    ["stride_length_cm", " cm"],
    ["cog_sway_cm", " cm"],
    ["cog_drop_cm", " cm"],
    ["head_stability_cm", " cm"],
    ["front_knee_extension_degrees", "°"],
    ["front_knee_flexion_degrees", "°"],
    ["spine_angle_degrees", "°"],
  ];

  for (const [key, unit] of scalarMeasurements) {
    const value = biomechanics[key];
    if (isFiniteNumber(value)) {
      measurements.push({
        key,
        labelKey: METRIC_LABEL_KEYS[key],
        value,
        unit,
      });
    }
  }

  return measurements;
}

export function BiomechanicsMeasurements({
  biomechanics,
  phaseSource,
}: BiomechanicsMeasurementsProps) {
  const { t } = useTranslation();
  const metricLabel = useMetricLabel();
  const measurements = buildMeasurements(biomechanics);
  const unmeasurableMetrics = biomechanics?.unmeasurable_metrics ?? [];
  const phaseSourceKey = phaseSource ? PHASE_SOURCE_KEYS[phaseSource] : null;
  const phaseSourceLabel = phaseSourceKey
    ? t(phaseSourceKey)
    : phaseSource
      ? humanizeMetricName(phaseSource)
      : t("biomechanics.phaseSources.notRecorded");

  return (
    <section
      className="biomechanics-measurements"
      aria-label={t("biomechanics.aria")}
      data-testid="biomechanics-measurements"
    >
      <div className="biomechanics-measurements__heading">
        <h3 className="biomechanics-measurements__title">
          {t("biomechanics.title")}
        </h3>
        <p className="biomechanics-measurements__disclaimer">
          {t("biomechanics.disclaimer")}
        </p>
      </div>

      <p
        className="biomechanics-measurements__phase-source"
        data-testid="biomechanics-phase-source"
      >
        <span>{t("biomechanics.phaseSource")}</span>
        <strong>{phaseSourceLabel}</strong>
      </p>

      {measurements.length === 0 ? (
        <p
          className="biomechanics-measurements__empty"
          data-testid="biomechanics-measurements-empty"
        >
          {t("biomechanics.empty")}
        </p>
      ) : (
        <dl className="biomechanics-measurements__grid">
          {measurements.map((measurement) => (
            <div
              className="biomechanics-measurements__item"
              data-testid={`biomechanics-measurement-${measurement.key}`}
              key={measurement.key}
            >
              <dt>{t(measurement.labelKey)}</dt>
              <dd>
                <strong>
                  {formatNumber(measurement.value, measurement.signed)}
                  {measurement.unit}
                </strong>
                {isFiniteNumber(measurement.reportingResolution) ? (
                  <span className="biomechanics-measurements__resolution">
                    {t("biomechanics.reportingResolution", {
                      value: formatNumber(measurement.reportingResolution),
                      unit: measurement.unit,
                    })}
                  </span>
                ) : null}
              </dd>
            </div>
          ))}
        </dl>
      )}

      {unmeasurableMetrics.length > 0 ? (
        <div
          className="biomechanics-measurements__unmeasurable"
          data-testid="biomechanics-unmeasurable"
        >
          <h4>{t("biomechanics.unmeasurableTitle")}</h4>
          <ul>
            {unmeasurableMetrics.map((metric, index) => {
              const labelKey = METRIC_LABEL_KEYS[metric.metric_name];
              const label = labelKey
                ? t(labelKey)
                : metricLabel(metric.metric_name);
              return (
                <li key={`${metric.metric_name}-${index}`}>
                  <strong>{label}</strong>: {metric.reason}
                </li>
              );
            })}
          </ul>
        </div>
      ) : null}
    </section>
  );
}

export default BiomechanicsMeasurements;
