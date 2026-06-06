/**
 * Type definitions for analysis API.
 *
 * Matches backend Pydantic schemas in app/schemas/analysis.py and
 * app/schemas/history.py.
 */

export type AnalysisStatus =
  | "pending"
  | "preprocessing"
  | "analyzing"
  | "evaluating"
  | "generating_report"
  | "completed"
  | "failed";

export type MetricRating = "below_range" | "within_range" | "above_range";

export type ColorCode = "green" | "yellow" | "red";

export interface AnalysisCreateRequest {
  file_key: string;
  user_id: string;
}

export interface AnalysisCreateResponse {
  analysis_id: string;
  status: AnalysisStatus;
  message?: string;
}

export interface AnalysisStatusResponse {
  analysis_id: string;
  status: AnalysisStatus;
  error_message?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
  created_at: string;
}

export interface MetricEvaluationResponse {
  metric_name: string;
  measured_value: number;
  unit: string;
  reference_min: number;
  reference_max: number;
  deviation_percent: number;
  rating: MetricRating;
  color_code: ColorCode;
}

export interface ImprovementAreaResponse {
  metric_name: string;
  deviation_percent: number;
  current_value: number;
  target_range_min: number;
  target_range_max: number;
  rank: number;
}

export interface DrillRecommendationResponse {
  drill_name: string;
  target_metric: string;
  description: string;
  /** "below" = 기준 미달, "above" = 기준 초과, "generic" = 전용 매핑 부재 */
  direction?: "below" | "above" | "generic";
}

export interface MetricDataPointResponse {
  analysis_id: string;
  recorded_at: string;
  value: number;
  rating: MetricRating;
}

export interface TrendDataResponse {
  metrics_history: Record<string, MetricDataPointResponse[]>;
  total_recordings: number;
  date_range_start?: string | null;
  date_range_end?: string | null;
  message?: string | null;
}

export interface VideoNormalizationMetadataResponse {
  normalization_applied: boolean;
  normalization_target_fps?: number | null;
  normalization_crop_box?: [number, number, number, number] | null;
  normalization_sampled_frame_count?: number | null;
  original_fps?: number | null;
  original_video_width?: number | null;
  original_video_height?: number | null;
  original_frame_count?: number | null;
  analysis_fps?: number | null;
  analysis_video_width?: number | null;
  analysis_video_height?: number | null;
  analysis_frame_count?: number | null;
}

export interface AnalysisMetadataResponse {
  video_normalization?: VideoNormalizationMetadataResponse;
  analysis_coordinate_system?: string | null;
  canonical_batting_direction?: "right" | "left" | null;
}

export interface SwingPhaseResponse {
  phase: string;
  start_frame: number;
  end_frame: number;
  duration_ms: number;
}

export interface BatSpeedResponse {
  speed_kmh: number;
  precision: number;
}

export interface LaunchAngleResponse {
  angle_degrees: number;
  precision: number;
}

export interface BiomechanicsResponse {
  bat_speed: BatSpeedResponse | null;
  attack_angle: LaunchAngleResponse | null;
  hand_path_efficiency: number | null;
  stride_length_cm: number | null;
  cog_sway_cm: number | null;
  cog_drop_cm: number | null;
  head_stability_cm: number | null;
  front_knee_flexion_degrees: number | null;
  spine_angle_degrees: number | null;
  processing_time_seconds: number | null;
}

export interface AnalysisReportResponse {
  analysis_id: string;
  user_id: string;
  created_at: string;
  status: string;
  video_metadata: Record<string, unknown>;
  quality_check: Record<string, unknown>;
  analysis_metadata: AnalysisMetadataResponse;
  swing_phases: SwingPhaseResponse[];
  biomechanics: BiomechanicsResponse | null;
  metric_evaluations: MetricEvaluationResponse[];
  improvements: ImprovementAreaResponse[];
  drill_recommendations: DrillRecommendationResponse[];
  overlay_video_url?: string | null;
  trend_data?: TrendDataResponse | null;
}

export interface OverlayUrlResponse {
  analysis_id: string;
  overlay_video_url: string;
  expires_in: number;
}

export interface AnalysisHistoryItem {
  analysis_id: string;
  video_id: string;
  status: string;
  created_at: string;
  completed_at?: string | null;
  video_file_name?: string | null;
  processing_time_seconds?: number | null;
}

export interface AnalysisHistoryResponse {
  items: AnalysisHistoryItem[];
  total: number;
  page: number;
  page_size: number;
  has_next: boolean;
}

/** Minimum recordings required before trend data is shown (Requirement 8.7/8.8). */
export const MIN_RECORDINGS_FOR_TREND = 2;

/** Maximum number of trend recordings shown (Requirement 8.7). */
export const MAX_TREND_RECORDINGS = 30;

/**
 * Ordered list of analysis status values for progress display.
 *
 * Used to compute progress percentage and to render the status timeline.
 * Terminal states ("completed", "failed") are excluded from the
 * progress sequence and handled separately by the UI.
 */
export const ANALYSIS_STATUS_SEQUENCE: AnalysisStatus[] = [
  "pending",
  "preprocessing",
  "analyzing",
  "evaluating",
  "generating_report",
  "completed",
];

/** Whether a status indicates the analysis has stopped progressing. */
export function isTerminalStatus(status: AnalysisStatus): boolean {
  return status === "completed" || status === "failed";
}
