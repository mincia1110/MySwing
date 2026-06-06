/**
 * Tabular display of measured metrics with color-coded ratings
 * (Requirements 8.3, 8.6).
 *
 * Each row contains:
 *  - metric name
 *  - measured value with unit
 *  - reference range
 *  - deviation %
 *  - color-coded rating badge (green / yellow / red)
 */
import type {
  ColorCode,
  MetricEvaluationResponse,
} from "../types/analysis";
import { useTranslation } from "../i18n";
import "./MetricsTable.css";

export interface MetricsTableProps {
  metrics: MetricEvaluationResponse[];
  /** Optional title; defaults to the localized "Metrics" label. */
  title?: string;
}

function formatNumber(value: number, fractionDigits = 1): string {
  if (!Number.isFinite(value)) return "-";
  // Avoid trailing -0
  const fixed = value.toFixed(fractionDigits);
  return fixed === "-0.0" ? "0.0" : fixed;
}

function formatDeviation(value: number): string {
  if (!Number.isFinite(value)) return "-";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(1)}%`;
}

function ratingClass(color: ColorCode): string {
  return `metrics-table__rating metrics-table__rating--${color}`;
}

export function MetricsTable({
  metrics,
  title,
}: MetricsTableProps) {
  const { t } = useTranslation();
  const resolvedTitle = title ?? t("metrics.title");

  if (metrics.length === 0) {
    return (
      <section
        className="metrics-table"
        aria-label={t("metrics.aria")}
        data-testid="metrics-table"
      >
        <h3 className="metrics-table__title">{resolvedTitle}</h3>
        <p className="metrics-table__empty" data-testid="metrics-table-empty">
          {t("metrics.empty")}
        </p>
      </section>
    );
  }

  return (
    <section
      className="metrics-table"
      aria-label={t("metrics.aria")}
      data-testid="metrics-table"
    >
      <h3 className="metrics-table__title">{resolvedTitle}</h3>
      <table className="metrics-table__table">
        <thead>
          <tr>
            <th scope="col">{t("metrics.metric")}</th>
            <th scope="col">{t("metrics.measured")}</th>
            <th scope="col">{t("metrics.referenceRange")}</th>
            <th scope="col">{t("metrics.deviation")}</th>
            <th scope="col">{t("metrics.rating")}</th>
          </tr>
        </thead>
        <tbody>
          {metrics.map((m) => (
            <tr
              key={m.metric_name}
              data-testid={`metrics-table-row-${m.metric_name}`}
              data-color={m.color_code}
              data-rating={m.rating}
            >
              <th scope="row">{m.metric_name}</th>
              <td>
                {formatNumber(m.measured_value)} {m.unit}
              </td>
              <td>
                {formatNumber(m.reference_min)} - {formatNumber(m.reference_max)}{" "}
                {m.unit}
              </td>
              <td className="metrics-table__deviation">
                {formatDeviation(m.deviation_percent)}
              </td>
              <td>
                <span
                  className={ratingClass(m.color_code)}
                  data-testid={`metrics-table-rating-${m.metric_name}`}
                >
                  {t(`metrics.ratings.${m.rating}`)}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

export default MetricsTable;
