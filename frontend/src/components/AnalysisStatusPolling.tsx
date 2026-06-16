/**
 * Polls the backend for analysis job status until it reaches a terminal
 * state (completed or failed) (Requirement 8.1).
 *
 * - Polls every `pollIntervalMs` (default: 2000ms) while the job is in a
 *   non-terminal state.
 * - Stops polling on terminal status, errors, or unmount.
 * - Renders the current phase and a progress bar derived from the ordered
 *   pipeline phases.
 * - Notifies the parent via `onCompleted` / `onFailed` callbacks.
 */
import { useEffect, useRef, useState } from "react";
import { getAnalysisStatus } from "../api/analysis";
import { useTranslation } from "../i18n";
import {
  ANALYSIS_STATUS_SEQUENCE,
  isTerminalStatus,
  type AnalysisStatus,
  type AnalysisStatusResponse,
} from "../types/analysis";
import "./AnalysisStatusPolling.css";

export interface AnalysisStatusPollingProps {
  /** UUID of the analysis job to poll. */
  analysisId: string;
  /** Polling interval in milliseconds. Defaults to 2000. */
  pollIntervalMs?: number;
  /** Called once when status === "completed". */
  onCompleted?: (status: AnalysisStatusResponse) => void;
  /** Called once when status === "failed" or polling errors exceed retry budget. */
  onFailed?: (status: AnalysisStatusResponse | null, error?: Error) => void;
  /** Number of transient polling errors to retry before failing. Defaults to 3. */
  maxErrorRetries?: number;
  /**
   * Optional override of the API call. Used for testing without mocking
   * axios; defaults to `getAnalysisStatus`.
   */
  fetchStatus?: (analysisId: string) => Promise<AnalysisStatusResponse>;
}

function progressPercent(status: AnalysisStatus): number {
  if (status === "failed") return 0;
  const idx = ANALYSIS_STATUS_SEQUENCE.indexOf(status);
  if (idx < 0) return 0;
  // Map index to [0, 100]; pending=0, completed=100.
  const span = ANALYSIS_STATUS_SEQUENCE.length - 1;
  return Math.round((idx / span) * 100);
}

export function AnalysisStatusPolling({
  analysisId,
  pollIntervalMs = 2000,
  onCompleted,
  onFailed,
  maxErrorRetries = 3,
  fetchStatus = getAnalysisStatus,
}: AnalysisStatusPollingProps) {
  const { t } = useTranslation();
  const [status, setStatus] = useState<AnalysisStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Refs guard against firing terminal callbacks twice within the active job.
  const completedNotifiedRef = useRef(false);
  const failedNotifiedRef = useRef(false);

  useEffect(() => {
    completedNotifiedRef.current = false;
    failedNotifiedRef.current = false;

    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let consecutiveErrors = 0;

    const scheduleNextPoll = (): void => {
      timer = setTimeout(() => {
        void poll();
      }, pollIntervalMs);
    };

    const poll = async (): Promise<void> => {
      if (cancelled) return;

      try {
        const next = await fetchStatus(analysisId);
        if (cancelled) return;

        consecutiveErrors = 0;
        setStatus(next);
        setError(null);

        if (next.status === "completed") {
          if (!completedNotifiedRef.current) {
            completedNotifiedRef.current = true;
            onCompleted?.(next);
          }
          return;
        }
        if (next.status === "failed") {
          if (!failedNotifiedRef.current) {
            failedNotifiedRef.current = true;
            onFailed?.(next);
          }
          return;
        }

        // Schedule the next poll for non-terminal status.
        scheduleNextPoll();
      } catch (err) {
        if (cancelled) return;
        const e = err instanceof Error ? err : new Error("Status poll failed");
        consecutiveErrors += 1;
        setError(e.message);
        if (consecutiveErrors <= maxErrorRetries) {
          scheduleNextPoll();
          return;
        }
        if (!failedNotifiedRef.current) {
          failedNotifiedRef.current = true;
          onFailed?.(null, e);
        }
      }
    };

    void poll();

    return () => {
      cancelled = true;
      if (timer !== null) {
        clearTimeout(timer);
      }
    };
  }, [analysisId, pollIntervalMs, onCompleted, onFailed, maxErrorRetries, fetchStatus]);

  const currentStatus: AnalysisStatus = status?.status ?? "pending";
  const percent = progressPercent(currentStatus);
  const isTerminal = isTerminalStatus(currentStatus);
  const variant =
    currentStatus === "failed"
      ? "failed"
      : currentStatus === "completed"
        ? "completed"
        : "running";
  const phaseLabel = t(`status.phases.${currentStatus}`);

  return (
    <section
      className={`analysis-status analysis-status--${variant}`}
      aria-label={t("status.aria")}
      data-testid="analysis-status"
      data-status={currentStatus}
      data-terminal={isTerminal ? "true" : "false"}
    >
      <h3 className="analysis-status__title">{t("status.title")}</h3>
      <p className="analysis-status__phase" data-testid="analysis-status-phase">
        {t("status.currentPhase", { phase: phaseLabel }).replace(phaseLabel, "")}
        <strong>{phaseLabel}</strong>
      </p>
      <div
        className="analysis-status__bar"
        role="progressbar"
        aria-valuenow={percent}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={t("status.progress")}
      >
        <div
          className="analysis-status__fill"
          style={{ width: `${percent}%` }}
          data-testid="analysis-status-fill"
        />
      </div>
      <span data-testid="analysis-status-percent">{percent}%</span>
      {error || status?.error_message ? (
        <p
          className="analysis-status__error"
          role="alert"
          data-testid="analysis-status-error"
        >
          {error ?? status?.error_message}
        </p>
      ) : null}
    </section>
  );
}

export default AnalysisStatusPolling;
