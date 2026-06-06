/**
 * API client methods for analysis and user history endpoints.
 *
 * Backend endpoints (Requirements 1.1, 8.1-8.8):
 *  - POST /api/v1/analyses                       -> create analysis job
 *  - GET  /api/v1/analyses/{id}/status           -> poll status
 *  - GET  /api/v1/analyses/{id}/report           -> full report (after completion)
 *  - GET  /api/v1/analyses/{id}/overlay          -> overlay video URL
 *  - GET  /api/v1/users/{id}/analyses            -> paginated history
 *  - GET  /api/v1/users/{id}/trends              -> trend data
 */

import type { AxiosInstance } from "axios";
import { apiClient as defaultClient } from "./client";
import type {
  AnalysisCreateResponse,
  AnalysisHistoryResponse,
  AnalysisReportResponse,
  AnalysisStatusResponse,
  OverlayUrlResponse,
  TrendDataResponse,
} from "../types/analysis";

/**
 * Create a new analysis job for an uploaded video.
 *
 * Returns 202 Accepted with the analysis_id and initial status.
 */
export async function createAnalysis(
  fileKey: string,
  userId: string,
  client: AxiosInstance = defaultClient,
): Promise<AnalysisCreateResponse> {
  const response = await client.post<AnalysisCreateResponse>("/analyses", {
    file_key: fileKey,
    user_id: userId,
  });
  return response.data;
}

/**
 * Get the current status of an analysis job.
 *
 * Used for polling progress (status transitions: pending → preprocessing →
 * analyzing → evaluating → generating_report → completed/failed).
 */
export async function getAnalysisStatus(
  analysisId: string,
  client: AxiosInstance = defaultClient,
): Promise<AnalysisStatusResponse> {
  const response = await client.get<AnalysisStatusResponse>(
    `/analyses/${encodeURIComponent(analysisId)}/status`,
  );
  return response.data;
}

/**
 * Retrieve the complete analysis report.
 *
 * Only succeeds when status === "completed". Returns 409 otherwise.
 */
export async function getAnalysisReport(
  analysisId: string,
  localeOrClient: "ko" | "en" | AxiosInstance = defaultClient,
  clientArg?: AxiosInstance,
): Promise<AnalysisReportResponse> {
  const locale = typeof localeOrClient === "string" ? localeOrClient : null;
  const client = typeof localeOrClient === "string" ? (clientArg ?? defaultClient) : localeOrClient;
  const url = `/analyses/${encodeURIComponent(analysisId)}/report`;
  const response = locale
    ? await client.get<AnalysisReportResponse>(url, { params: { locale } })
    : await client.get<AnalysisReportResponse>(url);
  return response.data;
}

/**
 * Retrieve the presigned URL for the overlay video (Requirement 8.2).
 */
export async function getAnalysisOverlay(
  analysisId: string,
  client: AxiosInstance = defaultClient,
): Promise<OverlayUrlResponse> {
  const response = await client.get<OverlayUrlResponse>(
    `/analyses/${encodeURIComponent(analysisId)}/overlay`,
  );
  return response.data;
}

/** @deprecated use getAnalysisOverlay */
export const getOverlayUrl = getAnalysisOverlay;

/**
 * Response shape for the analysis metrics endpoint.
 *
 * The backend returns biomechanics measurements plus the array of metric
 * evaluations (color-coded) for the completed analysis.
 */
export interface AnalysisMetricsResponse {
  analysis_id: string;
  biomechanics: Record<string, unknown>;
  evaluations: unknown[];
  processing_time_seconds?: number | null;
}

/**
 * Retrieve metric data (biomechanics + evaluations) for a completed analysis.
 */
export async function getAnalysisMetrics(
  analysisId: string,
  client: AxiosInstance = defaultClient,
): Promise<AnalysisMetricsResponse> {
  const response = await client.get<AnalysisMetricsResponse>(
    `/analyses/${encodeURIComponent(analysisId)}/metrics`,
  );
  return response.data;
}

/**
 * Retrieve paginated analysis history for a user.
 */
export async function getUserAnalyses(
  userId: string,
  page = 1,
  pageSize = 20,
  client: AxiosInstance = defaultClient,
): Promise<AnalysisHistoryResponse> {
  const response = await client.get<AnalysisHistoryResponse>(
    `/users/${encodeURIComponent(userId)}/analyses`,
    {
      params: { page, page_size: pageSize },
    },
  );
  return response.data;
}

/**
 * Retrieve trend data for a user (Requirements 8.7, 8.8).
 *
 * Returns metrics_history populated when the user has >= 2 completed
 * analyses; otherwise returns an empty history with a `message` field.
 */
export async function getUserTrends(
  userId: string,
  client: AxiosInstance = defaultClient,
): Promise<TrendDataResponse> {
  const response = await client.get<TrendDataResponse>(
    `/users/${encodeURIComponent(userId)}/trends`,
  );
  return response.data;
}
