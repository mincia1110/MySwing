/**
 * Visual progress indicator (0-100%).
 *
 * Renders an accessible progress bar with a percentage label.
 */
import "./UploadProgress.css";

export interface UploadProgressProps {
  /** Current upload progress as a percentage in [0, 100]. */
  percent: number;
  /** Optional descriptive label rendered above the bar. */
  label?: string;
  /** Marks the operation as failed and shows an error style. */
  error?: string | null;
}

function clampPercent(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  if (value < 0) return 0;
  if (value > 100) return 100;
  return Math.round(value);
}

export function UploadProgress({ percent, label, error }: UploadProgressProps) {
  const safePercent = clampPercent(percent);
  const status: "idle" | "uploading" | "complete" | "error" = error
    ? "error"
    : safePercent >= 100
      ? "complete"
      : safePercent > 0
        ? "uploading"
        : "idle";

  return (
    <div
      className={`upload-progress upload-progress--${status}`}
      data-testid="upload-progress"
    >
      {label ? (
        <div className="upload-progress__label">{label}</div>
      ) : null}
      <div
        className="upload-progress__bar"
        role="progressbar"
        aria-valuenow={safePercent}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={label ?? "Upload progress"}
      >
        <div
          className="upload-progress__fill"
          style={{ width: `${safePercent}%` }}
          data-testid="upload-progress-fill"
        />
      </div>
      <div className="upload-progress__meta">
        <span className="upload-progress__percent" data-testid="upload-progress-percent">
          {safePercent}%
        </span>
        {error ? (
          <span className="upload-progress__error" role="alert">
            {error}
          </span>
        ) : null}
      </div>
    </div>
  );
}

export default UploadProgress;
