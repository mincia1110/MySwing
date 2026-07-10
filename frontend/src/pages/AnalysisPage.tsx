/**
 * Analysis result page.
 *
 * Drives the polling-then-report flow:
 *  1. Reads :analysisId from the route.
 *  2. Renders <AnalysisStatusPolling /> until the job reaches a terminal state.
 *  3. On completion, fetches the full report and current user's trend data.
 */
import { useCallback, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
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
  const { language, t } = useTranslation();

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
        const [reportResult, trendResult] = await Promise.allSettled([
          getAnalysisReport(analysisId, language),
          getUserTrends(),
        ]);

        if (reportResult.status === "rejected") {
          throw reportResult.reason;
        }

        setReport(reportResult.value);
        setTrendData(trendResult.status === "fulfilled" ? trendResult.value : null);
        setPhase("ready");
      } catch (err) {
        setError(
          err instanceof Error ? err.message : t("analysisPage.reportError"),
        );
        setPhase("failed");
      }
    },
    [analysisId, language, t],
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
      <main className="page page--analysis">
        <h1>{t("analysisPage.title")}</h1>
        <p role="alert" data-testid="analysis-page-missing-id">
          {t("analysisPage.missingId")}
        </p>
      </main>
    );
  }

  return (
    <main className="page page--analysis" data-testid="analysis-page">
      <div className="page__header">
        <h1 className="page__title">{t("analysisPage.title")}</h1>
        <Link
          to="/"
          className="button button--primary"
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
            className="page__error"
          >
            {error}
          </p>
          <Link
            to="/"
            className="button button--primary"
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
