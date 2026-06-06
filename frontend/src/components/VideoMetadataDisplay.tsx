/**
 * Display thumbnail + filename, duration, resolution, and file size after
 * a successful upload (Requirement 1.7).
 */
import type { VideoMetadataWithThumbnailResponse } from "../types/video";
import "./VideoMetadataDisplay.css";

export interface VideoMetadataDisplayProps {
  metadata: VideoMetadataWithThumbnailResponse;
}

export function formatFileSize(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) {
    return "0 B";
  }
  const units = ["B", "KB", "MB", "GB"];
  let size = bytes;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  const decimals = unitIndex === 0 ? 0 : size >= 100 ? 0 : size >= 10 ? 1 : 2;
  return `${size.toFixed(decimals)} ${units[unitIndex]}`;
}

export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) {
    return "0:00";
  }
  const total = Math.round(seconds);
  const minutes = Math.floor(total / 60);
  const remaining = total % 60;
  return `${minutes}:${remaining.toString().padStart(2, "0")}`;
}

export function VideoMetadataDisplay({ metadata }: VideoMetadataDisplayProps) {
  const { file_name, duration_seconds, resolution, file_size_bytes, thumbnail_url } =
    metadata;

  return (
    <section
      className="video-metadata"
      aria-label="비디오 정보"
      data-testid="video-metadata"
    >
      <div className="video-metadata__thumbnail">
        {thumbnail_url ? (
          <img
            src={thumbnail_url}
            alt={`${file_name} 썸네일`}
            data-testid="video-metadata-thumbnail"
          />
        ) : (
          <div
            className="video-metadata__placeholder"
            aria-label="썸네일 사용 불가"
            data-testid="video-metadata-thumbnail-placeholder"
          >
            🎬
          </div>
        )}
      </div>
      <dl className="video-metadata__details">
        <dt>파일명</dt>
        <dd data-testid="video-metadata-filename">{file_name}</dd>

        <dt>길이</dt>
        <dd data-testid="video-metadata-duration">
          {formatDuration(duration_seconds)}
        </dd>

        <dt>해상도</dt>
        <dd data-testid="video-metadata-resolution">
          {resolution.width} × {resolution.height}
        </dd>

        <dt>크기</dt>
        <dd data-testid="video-metadata-filesize">
          {formatFileSize(file_size_bytes)}
        </dd>
      </dl>
    </section>
  );
}

export default VideoMetadataDisplay;
