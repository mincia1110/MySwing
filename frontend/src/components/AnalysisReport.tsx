/**
 * Composite analysis report view (Requirement 8.1).
 *
 * Combines:
 *  - Overlay video player (8.2)
 *  - Metrics table with color-coded ratings (8.3 / 8.6)
 *  - Top-3 improvement areas (7.6 / 8.4)
 *  - Drill recommendation cards (8.4)
 *  - Comparison view: user vs professional reference (8.5)
 *  - Trend chart (8.7 / 8.8)
 *
 * Trend data is optional. When supplied to the parent (typically via
 * `getUserTrends`), this component delegates rendering to `TrendChart`.
 * When fewer than 2 recordings exist, `TrendChart` itself displays the
 * "minimum recordings" message (Requirement 8.8).
 */
import type {
  AnalysisReportResponse,
  SwingWindowEvidence,
  TrendDataResponse,
} from "../types/analysis";
import type { QualityCheckResponse } from "../types/video";
import { useTranslation } from "../i18n";
import { ComparisonView } from "./ComparisonView";
import { BiomechanicsMeasurements } from "./BiomechanicsMeasurements";
import { DrillRecommendationCard } from "./DrillRecommendationCard";
import { ImprovementAreasList } from "./ImprovementAreasList";
import { MetricsTable } from "./MetricsTable";
import { OverlayVideoPlayer } from "./OverlayVideoPlayer";
import { QualityCheckResult } from "./QualityCheckResult";
import { TrendChart } from "./TrendChart";
import "./AnalysisReport.css";

export interface AnalysisReportProps {
  report: AnalysisReportResponse;
  /** Optional trend data fetched separately via getUserTrends. */
  trendData?: TrendDataResponse | null;
}

function isQualityStatus(value: unknown): value is QualityCheckResponse["brightness_status"] {
  return value === "pass" || value === "warning";
}

function isQualityCheckResponse(value: unknown): value is QualityCheckResponse {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  return (
    isQualityStatus(candidate.brightness_status) &&
    isQualityStatus(candidate.framing_status) &&
    isQualityStatus(candidate.resolution_status) &&
    isQualityStatus(candidate.frame_rate_stability_status) &&
    typeof candidate.brightness_value === "number" &&
    typeof candidate.swing_arc_visibility_percent === "number" &&
    typeof candidate.frame_rate_variation_percent === "number" &&
    Array.isArray(candidate.warnings)
  );
}

function formatCreatedAt(iso: string, language: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString(language === "ko" ? "ko-KR" : "en-US");
}

/**
 * Extracts validated multiple-swing isolation evidence from the loosely typed
 * phase_evidence payload. Returns null when no isolation notice applies.
 */
function extractIsolatedSwingWindow(
  phaseEvidence: Record<string, unknown> | undefined,
): { candidateCount: number | null } | null {
  if (!phaseEvidence || typeof phaseEvidence !== "object") return null;
  const swingWindow = phaseEvidence.swing_window;
  if (!swingWindow || typeof swingWindow !== "object") return null;
  const evidence = swingWindow as SwingWindowEvidence;
  if (evidence.multiple_swing_detected !== true) return null;
  if (evidence.isolation_applied !== true) return null;
  const count = evidence.candidate_count;
  const candidateCount =
    typeof count === "number" && Number.isFinite(count) && count >= 2
      ? Math.trunc(count)
      : null;
  return { candidateCount };
}

export function AnalysisReport({ report, trendData }: AnalysisReportProps) {
  const { t, language } = useTranslation();
  const effectiveTrendData = trendData ?? report.trend_data ?? null;
  const isolatedSwingWindow = extractIsolatedSwingWindow(report.phase_evidence);
  const qualityCheck = isQualityCheckResponse(report.quality_check)
    ? report.quality_check
    : null;
  const overlayFps =
    report.analysis_metadata.video_normalization?.analysis_fps ??
    report.analysis_metadata.video_normalization?.original_fps ??
    30;

  return (
    <article
      className="analysis-report"
      aria-label={t("report.aria")}
      data-testid="analysis-report"
      data-analysis-id={report.analysis_id}
    >
      <header className="analysis-report__header">
        <h2 className="analysis-report__title">{t("report.title")}</h2>
        <span className="analysis-report__meta">
          {t("report.analysisId", { id: report.analysis_id })} -{" "}
          {t("report.createdAt", {
            date: formatCreatedAt(report.created_at, language),
          })}
        </span>
      </header>

      {isolatedSwingWindow ? (
        <p
          className="analysis-report__swing-window-notice"
          role="status"
          aria-label={t("report.multipleSwingsAria")}
          data-testid="analysis-report-swing-window-notice"
        >
          {isolatedSwingWindow.candidateCount !== null
            ? t("report.multipleSwingsWithCount", {
                count: isolatedSwingWindow.candidateCount,
              })
            : t("report.multipleSwings")}
        </p>
      ) : null}

      <div className="analysis-report__hero" data-testid="analysis-report-hero">
        <OverlayVideoPlayer
          videoUrl={report.overlay_video_url}
          phases={report.swing_phases}
          fps={overlayFps}
        />
        {qualityCheck ? <QualityCheckResult result={qualityCheck} /> : null}
      </div>

      <BiomechanicsMeasurements
        biomechanics={report.biomechanics}
        phaseSource={report.phase_source}
      />

      <div
        className="analysis-report__insights"
        data-testid="analysis-report-insights"
      >
        <MetricsTable metrics={report.metric_evaluations} />
        <ImprovementAreasList improvements={report.improvements} />
      </div>

      <section
        className="analysis-report__section"
        aria-label={t("report.drillsAria")}
        data-testid="analysis-report-drills"
      >
        <h3 className="analysis-report__section-title">{t("report.drillsTitle")}</h3>
        {report.drill_recommendations.length === 0 ? (
          <p data-testid="analysis-report-drills-empty">
            {t("report.noDrills")}
          </p>
        ) : (
          <div className="analysis-report__drills-grid">
            {report.drill_recommendations.map((drill, idx) => (
              <DrillRecommendationCard
                key={`${drill.drill_name}-${idx}`}
                drill={drill}
              />
            ))}
          </div>
        )}
      </section>

      <ComparisonView userPhases={report.swing_phases} />

      {effectiveTrendData ? (
        <TrendChart trendData={effectiveTrendData} />
      ) : null}
    </article>
  );
}

export default AnalysisReport;
