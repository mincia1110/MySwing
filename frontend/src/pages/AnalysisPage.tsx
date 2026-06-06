/**
 * Analysis result page.
 *
 * Drives the polling-then-report flow:
 *  1. Reads :analysisId from the route (also accepts ?userId= for trend data).
 *  2. Renders <AnalysisStatusPolling /> until the job reaches a terminal state.
 *  3. On completion, fetches the full report (and trend data if userId is
 *     present) and renders <AnalysisReport />.
 */
import { useCallback, useEffect, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { getAnalysisReport, getUserTrends } from "../api/analysis";
import { AnalysisReport } from "../components/AnalysisReport";
import { AnalysisStatusPolling } from "../components/AnalysisStatusPolling";
import { useTranslation } from "../i18n";
import type {
  AnalysisReportResponse,
  AnalysisStatusResponse,
  TrendDataResponse,
} from "../types/analysis";

export function AnalysisPage() {
  const { analysisId } = useParams<{ analysisId: string }>();
  const [searchParams] = useSearchParams();
  const { language, t } = useTranslation();
  const userId = searchParams.get("userId");

  const [phase, setPhase] = useState<"polling" | "loading" | "ready" | "failed">(
    "polling",
  );
  const [report, setReport] = useState<AnalysisReportResponse | null>(null);
  const [trendData, setTrendData] = useState<TrendDataResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleCompleted = useCallback(
    async (_status: AnalysisStatusResponse) => {
      if (!analysisId) return;
      setPhase("loading");
      try {
        const [reportData, trends] = await Promise.all([
          getAnalysisReport(analysisId, language),
          userId ? getUserTrends(userId) : Promise.resolve(null),
        ]);
        setReport(reportData);
        setTrendData(trends);
        setPhase("ready");
      } catch (err) {
        setError(
          err instanceof Error ? err.message : t("analysisPage.reportError"),
        );
        setPhase("failed");
      }
    },
    [analysisId, userId, language, t],
  );

  const handleFailed = useCallback(
    (status: AnalysisStatusResponse | null, e?: Error) => {
      setError(
        e?.message ??
          status?.error_message ??
          t("analysisPage.analysisError"),
      );
      setPhase("failed");
    },
    [t],
  );

  useEffect(() => {
    setPhase("polling");
    setReport(null);
    setTrendData(null);
    setError(null);
  }, [analysisId]);

  if (!analysisId) {
    return (
      <main style={{ maxWidth: 960, margin: "0 auto" }}>
        <h1>{t("analysisPage.title")}</h1>
        <p role="alert" data-testid="analysis-page-missing-id">
          {t("analysisPage.missingId")}
        </p>
      </main>
    );
  }

  return (
    <main style={{ maxWidth: 960, margin: "0 auto", padding: "2rem 1rem" }} data-testid="analysis-page">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
        <h1 style={{ margin: 0 }}>{t("analysisPage.title")}</h1>
        <Link
          to="/"
          style={{
            padding: "0.6rem 1.2rem",
            backgroundColor: "#2563eb",
            color: "white",
            textDecoration: "none",
            borderRadius: "6px",
            fontSize: "0.9rem",
            fontWeight: 500,
          }}
        >
          {t("analysisPage.newAnalysis")}
        </Link>
      </div>
      {phase !== "ready" ? (
        <AnalysisStatusPolling
          analysisId={analysisId}
          onCompleted={handleCompleted}
          onFailed={handleFailed}
        />
      ) : null}
      {phase === "loading" ? (
        <p data-testid="analysis-page-loading">{t("analysisPage.loading")}</p>
      ) : null}
      {phase === "failed" ? (
        <div>
          <p
            role="alert"
            data-testid="analysis-page-error"
            style={{ color: "#b91c1c" }}
          >
            {error}
          </p>
          <Link
            to="/"
            style={{
              display: "inline-block",
              marginTop: "1rem",
              padding: "0.6rem 1.2rem",
              backgroundColor: "#2563eb",
              color: "white",
              textDecoration: "none",
              borderRadius: "6px",
            }}
          >
            {t("analysisPage.retry")}
          </Link>
        </div>
      ) : null}
      {phase === "ready" && report ? (
        <AnalysisReport report={report} trendData={trendData} />
      ) : null}
    </main>
  );
}

export default AnalysisPage;
