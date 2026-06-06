/**
 * Display pass/warning per quality check (brightness, framing, resolution, fps).
 *
 * Requirement 9.8: surface quality issues with actionable warnings.
 */
import type { QualityCheckResponse, QualityStatus } from "../types/video";
import { useTranslation } from "../i18n";
import "./QualityCheckResult.css";

export interface QualityCheckResultProps {
  result: QualityCheckResponse;
}

interface CheckRow {
  key: string;
  label: string;
  status: QualityStatus;
  value: string;
}

export function QualityCheckResult({ result }: QualityCheckResultProps) {
  const { t } = useTranslation();
  const rows: CheckRow[] = [
    {
      key: "brightness",
      label: t("quality.brightness"),
      status: result.brightness_status,
      value: `${result.brightness_value.toFixed(0)} lux`,
    },
    {
      key: "framing",
      label: t("quality.framing"),
      status: result.framing_status,
      value: `${result.swing_arc_visibility_percent.toFixed(0)}%`,
    },
    {
      key: "resolution",
      label: t("quality.resolution"),
      status: result.resolution_status,
      value: result.resolution_status === "pass" ? t("quality.sufficient") : t("quality.low"),
    },
    {
      key: "fps",
      label: t("quality.fps"),
      status: result.frame_rate_stability_status,
      value: t("quality.variation", {
        value: result.frame_rate_variation_percent.toFixed(1),
      }),
    },
  ];

  return (
    <section
      className="quality-check"
      aria-label={t("quality.aria")}
      data-testid="quality-check"
    >
      <h3 className="quality-check__title">{t("quality.title")}</h3>
      <ul className="quality-check__list">
        {rows.map((row) => (
          <li
            key={row.key}
            className={`quality-check__item quality-check__item--${row.status}`}
            data-testid={`quality-check-${row.key}`}
            data-status={row.status}
          >
            <span className="quality-check__label">{row.label}</span>
            <span
              className={`quality-check__status quality-check__status--${row.status}`}
              aria-label={row.status === "pass" ? t("quality.pass") : t("quality.warning")}
            >
              {row.status === "pass" ? t("quality.pass") : t("quality.warning")}
            </span>
            <span className="quality-check__value">{row.value}</span>
          </li>
        ))}
      </ul>
      {result.warnings.length > 0 ? (
        <ul className="quality-check__warnings" data-testid="quality-check-warnings">
          {result.warnings.map((warning, idx) => (
            <li key={idx} role="alert">
              {warning}
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}

export default QualityCheckResult;
