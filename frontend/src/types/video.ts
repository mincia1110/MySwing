/**
 * Type definitions matching backend Pydantic schemas in app/schemas/video.py.
 */

export interface PresignedUrlRequest {
  file_name: string;
  /** MIME type matching ^video/(mp4|quicktime|x-msvideo)$ */
  content_type: string;
}

export interface PresignedUrlResponse {
  upload_url: string;
  file_key: string;
  expires_in: number;
}

export interface ResolutionResponse {
  width: number;
  height: number;
}

export interface VideoMetadataResponse {
  file_key: string;
  file_name: string;
  file_size_bytes: number;
  duration_seconds: number;
  resolution: ResolutionResponse;
  frame_rate: number;
  codec: string;
  format: string;
  thumbnail_url?: string | null;
}


export type VideoInputPolicySeverity = "ok" | "warning" | "error";

export type VideoInputPolicyReason =
  | "video_too_short"
  | "video_longer_than_recommended"
  | "video_too_long"
  | "metadata_unavailable";

export interface VideoInputPolicyResponse {
  accepted: boolean;
  severity: VideoInputPolicySeverity;
  reason?: VideoInputPolicyReason | null;
  duration_sec?: number | null;
  ideal_duration_sec: number;
  recommended_min_duration_sec: number;
  recommended_max_duration_sec: number;
  max_duration_sec: number;
  message: string;
  recommendation: string;
}

export interface VideoMetadataWithThumbnailResponse {
  file_name: string;
  duration_seconds: number;
  resolution: ResolutionResponse;
  file_size_bytes: number;
  thumbnail_url?: string | null;
  input_validation?: VideoInputPolicyResponse | null;
}

export type QualityStatus = "pass" | "warning";

export interface QualityCheckResponse {
  brightness_status: QualityStatus;
  framing_status: QualityStatus;
  resolution_status: QualityStatus;
  frame_rate_stability_status: QualityStatus;
  brightness_value: number;
  swing_arc_visibility_percent: number;
  frame_rate_variation_percent: number;
  warnings: string[];
}

export interface VideoValidationResponse {
  is_valid: boolean;
  format_ok: boolean;
  size_ok: boolean;
  resolution_ok: boolean;
  frame_rate_ok: boolean;
  errors: string[];
}
