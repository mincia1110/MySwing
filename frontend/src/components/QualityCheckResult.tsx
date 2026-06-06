/**
 * Display pass/warning per quality check (brightness, framing, resolution, fps).
 *
 * Requirement 9.8: surface quality issues with actionable warnings.
 */
import type { QualityCheckResponse, QualityStatus } from "../types/video";
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
  const rows: CheckRow[] = [
    {
      key: "brightness",
      label: "밝기",
      status: result.brightness_status,
      value: `${result.brightness_value.toFixed(0)} lux`,
    },
    {
      key: "framing",
      label: "스윙 아크 가시성",
      status: result.framing_status,
      value: `${result.swing_arc_visibility_percent.toFixed(0)}%`,
    },
    {
      key: "resolution",
      label: "해상도",
      status: result.resolution_status,
      value: result.resolution_status === "pass" ? "충분" : "낮음",
    },
    {
      key: "fps",
      label: "프레임레이트 안정성",
      status: result.frame_rate_stability_status,
      value: `편차 ${result.frame_rate_variation_percent.toFixed(1)}%`,
    },
  ];

  return (
    <section
      className="quality-check"
      aria-label="비디오 품질 검증 결과"
      data-testid="quality-check"
    >
      <h3 className="quality-check__title">품질 검증 결과</h3>
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
              aria-label={row.status === "pass" ? "통과" : "경고"}
            >
              {row.status === "pass" ? "통과" : "경고"}
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
