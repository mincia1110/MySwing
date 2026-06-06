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
import type { AnalysisReportResponse, TrendDataResponse } from "../types/analysis";
import { useTranslation } from "../i18n";
import { ComparisonView } from "./ComparisonView";
import { DrillRecommendationCard } from "./DrillRecommendationCard";
import { ImprovementAreasList } from "./ImprovementAreasList";
import { MetricsTable } from "./MetricsTable";
import { OverlayVideoPlayer } from "./OverlayVideoPlayer";
import { TrendChart } from "./TrendChart";
import "./AnalysisReport.css";

export interface AnalysisReportProps {
  report: AnalysisReportResponse;
  /** Optional trend data fetched separately via getUserTrends. */
  trendData?: TrendDataResponse | null;
}

function formatCreatedAt(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString();
}

export function AnalysisReport({ report, trendData }: AnalysisReportProps) {
  const { t } = useTranslation();
  const effectiveTrendData = trendData ?? report.trend_data ?? null;

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
          {t("report.createdAt", { date: formatCreatedAt(report.created_at) })}
        </span>
      </header>

      <OverlayVideoPlayer videoUrl={report.overlay_video_url} />

      <MetricsTable metrics={report.metric_evaluations} />

      <ImprovementAreasList improvements={report.improvements} />

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
