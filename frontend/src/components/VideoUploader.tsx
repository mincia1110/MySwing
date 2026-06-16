/**
 * Video upload component.
 *
 * Flow:
 *  1. User selects a file via input or drag-and-drop.
 *  2. Component requests a presigned URL from the backend.
 *  3. File is uploaded directly to S3 via XMLHttpRequest with progress.
 *  4. After upload, backend metadata endpoint is invoked to get
 *     thumbnail + video metadata (Requirement 1.7).
 *  5. Caller is notified via onUploadComplete with the file_key + metadata.
 *
 * Quality check display (Requirement 9.8) is rendered when a `qualityResult`
 * prop is supplied by the parent (the quality endpoint is fed by a separate
 * task).
 */
import { useCallback, useRef, useState } from "react";
import {
  getMetadata,
  getPresignedUrl,
  uploadToS3,
  type UploadProgressEvent,
} from "../api/client";
import { useTranslation } from "../i18n";
import type {
  PresignedUrlResponse,
  QualityCheckResponse,
  VideoMetadataWithThumbnailResponse,
} from "../types/video";
import { QualityCheckResult } from "./QualityCheckResult";
import { UploadProgress } from "./UploadProgress";
import { VideoMetadataDisplay } from "./VideoMetadataDisplay";
import "./VideoUploader.css";

export interface VideoUploaderProps {
  /** Called when the full pipeline (upload + metadata) succeeds. */
  onUploadComplete?: (result: {
    fileKey: string;
    metadata: VideoMetadataWithThumbnailResponse;
  }) => void;
  /** Called whenever an error occurs during the upload pipeline. */
  onUploadError?: (error: Error) => void;
  /** Optional quality check result rendered after upload completes. */
  qualityResult?: QualityCheckResponse | null;
  /** Maximum allowed file size in bytes. Defaults to 500MB (Requirement 1.2). */
  maxFileSizeBytes?: number;
  /** Allowed MIME types. Defaults to backend-supported video formats. */
  acceptedMimeTypes?: string[];
}

const DEFAULT_MAX_SIZE = 500 * 1024 * 1024;
const RECOMMENDED_MIN_DURATION_SEC = 3;
const RECOMMENDED_MAX_DURATION_SEC = 7;
const MAX_DURATION_SEC = 10;
const DEFAULT_MIME_TYPES = [
  "video/mp4",
  "video/quicktime",
  "video/x-msvideo",
];

// Allowed file extensions (fallback when MIME type is empty or generic)
const ALLOWED_EXTENSIONS = [".mp4", ".mov", ".avi"];

type UploadStage =
  | "idle"
  | "preparing"
  | "uploading"
  | "fetching-metadata"
  | "complete"
  | "error";

interface UploadState {
  stage: UploadStage;
  percent: number;
  fileName: string | null;
  errorMessage: string | null;
  warningMessage: string | null;
  metadata: VideoMetadataWithThumbnailResponse | null;
  fileKey: string | null;
}

const INITIAL_STATE: UploadState = {
  stage: "idle",
  percent: 0,
  fileName: null,
  errorMessage: null,
  warningMessage: null,
  metadata: null,
  fileKey: null,
};

function readVideoDuration(file: File): Promise<number | null> {
  const directDuration = (file as File & { duration?: unknown }).duration;
  if (typeof directDuration === "number" && Number.isFinite(directDuration)) {
    return Promise.resolve(directDuration);
  }

  if (
    typeof document === "undefined" ||
    typeof URL === "undefined" ||
    typeof URL.createObjectURL !== "function" ||
    typeof URL.revokeObjectURL !== "function"
  ) {
    return Promise.resolve(null);
  }

  return new Promise((resolve) => {
    const video = document.createElement("video");
    const objectUrl = URL.createObjectURL(file);
    const cleanup = () => {
      URL.revokeObjectURL(objectUrl);
      video.removeAttribute("src");
    };
    video.preload = "metadata";
    video.onloadedmetadata = () => {
      const duration = Number.isFinite(video.duration) ? video.duration : null;
      cleanup();
      resolve(duration);
    };
    video.onerror = () => {
      cleanup();
      resolve(null);
    };
    video.src = objectUrl;
  });
}

function stageLabel(stage: UploadStage, t: (key: string) => string): string {
  switch (stage) {
    case "preparing":
      return t("uploader.preparing");
    case "uploading":
      return t("uploader.uploading");
    case "fetching-metadata":
      return t("uploader.fetchingMetadata");
    case "complete":
      return t("uploader.complete");
    case "error":
      return t("uploader.error");
    default:
      return "";
  }
}

export function VideoUploader({
  onUploadComplete,
  onUploadError,
  qualityResult,
  maxFileSizeBytes = DEFAULT_MAX_SIZE,
  acceptedMimeTypes = DEFAULT_MIME_TYPES,
}: VideoUploaderProps) {
  const { t } = useTranslation();
  const [state, setState] = useState<UploadState>(INITIAL_STATE);
  const [isDragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const validateFile = useCallback(
    (file: File): string | null => {
      // Check MIME type first; if empty/generic, fall back to extension check
      const mimeOk = acceptedMimeTypes.length === 0 || acceptedMimeTypes.includes(file.type);
      const ext = "." + file.name.split(".").pop()?.toLowerCase();
      const extOk = ALLOWED_EXTENSIONS.includes(ext);

      if (!mimeOk && !extOk) {
        return t("uploader.unsupportedType", { type: file.type || ext });
      }
      if (file.size > maxFileSizeBytes) {
        const limitMb = Math.round(maxFileSizeBytes / (1024 * 1024));
        return t("uploader.fileTooLarge", { limitMb });
      }
      return null;
    },
    [acceptedMimeTypes, maxFileSizeBytes, t],
  );

  const startUpload = useCallback(
    async (file: File) => {
      const validationError = validateFile(file);
      if (validationError) {
        setState({
          stage: "error",
          percent: 0,
          fileName: file.name,
          errorMessage: validationError,
          warningMessage: null,
          metadata: null,
          fileKey: null,
        });
        onUploadError?.(new Error(validationError));
        return;
      }

      const duration = await readVideoDuration(file);
      if (duration != null && duration > MAX_DURATION_SEC) {
        const errorMessage = t("uploader.durationTooLong");
        setState({
          stage: "error",
          percent: 0,
          fileName: file.name,
          errorMessage,
          warningMessage: null,
          metadata: null,
          fileKey: null,
        });
        onUploadError?.(new Error(errorMessage));
        return;
      }
      const warningMessage =
        duration != null &&
        (duration < RECOMMENDED_MIN_DURATION_SEC || duration > RECOMMENDED_MAX_DURATION_SEC)
          ? t("uploader.durationWarning")
          : null;

      setState({
        stage: "preparing",
        percent: 0,
        fileName: file.name,
        errorMessage: null,
        warningMessage,
        metadata: null,
        fileKey: null,
      });

      let presigned: PresignedUrlResponse;
      let resolvedContentType = "";
      try {
        // Determine content_type: use file.type if available, otherwise infer from extension
        resolvedContentType = file.type;
        if (!resolvedContentType) {
          const ext = file.name.split(".").pop()?.toLowerCase();
          if (ext === "mov") resolvedContentType = "video/quicktime";
          else if (ext === "avi") resolvedContentType = "video/x-msvideo";
          else resolvedContentType = "video/mp4";
        }
        presigned = await getPresignedUrl({
          file_name: file.name,
          content_type: resolvedContentType,
        });
      } catch (err) {
        const error =
          err instanceof Error ? err : new Error("Presigned URL request failed");
        setState((prev) => ({
          ...prev,
          stage: "error",
          errorMessage: error.message,
        }));
        onUploadError?.(error);
        return;
      }

      setState((prev) => ({ ...prev, stage: "uploading", fileKey: presigned.file_key }));

      try {
        await uploadToS3(presigned.upload_url, file, {
          contentType: resolvedContentType,
          onProgress: (event: UploadProgressEvent) => {
            setState((prev) =>
              prev.stage === "uploading"
                ? { ...prev, percent: event.percent }
                : prev,
            );
          },
        });
      } catch (err) {
        const error = err instanceof Error ? err : new Error("S3 upload failed");
        setState((prev) => ({
          ...prev,
          stage: "error",
          errorMessage: error.message,
        }));
        onUploadError?.(error);
        return;
      }

      setState((prev) => ({
        ...prev,
        stage: "fetching-metadata",
        percent: 100,
      }));

      try {
        const metadata = await getMetadata(presigned.file_key);
        const inputValidation = metadata.input_validation;
        if (inputValidation?.accepted === false || inputValidation?.severity === "error") {
          const error = new Error(inputValidation.message);
          setState((prev) => ({
            ...prev,
            stage: "error",
            percent: 100,
            errorMessage: inputValidation.message,
            metadata,
            fileKey: presigned.file_key,
          }));
          onUploadError?.(error);
          return;
        }

        setState((prev) => ({
          stage: "complete",
          percent: 100,
          fileName: file.name,
          errorMessage: null,
          warningMessage:
            inputValidation?.severity === "warning"
              ? inputValidation.message
              : prev.warningMessage,
          metadata,
          fileKey: presigned.file_key,
        }));
        onUploadComplete?.({ fileKey: presigned.file_key, metadata });
      } catch (err) {
        const error =
          err instanceof Error ? err : new Error("Metadata fetch failed");
        setState((prev) => ({
          ...prev,
          stage: "error",
          errorMessage: error.message,
        }));
        onUploadError?.(error);
      }
    },
    [onUploadComplete, onUploadError, t, validateFile],
  );

  const handleFileInput = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (file) {
        void startUpload(file);
      }
      // Allow re-uploading the same file by clearing the input value.
      event.target.value = "";
    },
    [startUpload],
  );

  const handleDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      setDragOver(false);
      const file = event.dataTransfer.files?.[0];
      if (file) {
        void startUpload(file);
      }
    },
    [startUpload],
  );

  const handleDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setDragOver(false);
  }, []);

  const handleSelectClick = useCallback(() => {
    inputRef.current?.click();
  }, []);

  const reset = useCallback(() => {
    setState(INITIAL_STATE);
  }, []);

  const isUploading =
    state.stage === "preparing" ||
    state.stage === "uploading" ||
    state.stage === "fetching-metadata";

  return (
    <div className="video-uploader" data-testid="video-uploader">
      <div
        className={[
          "video-uploader__dropzone",
          isDragOver ? "video-uploader__dropzone--dragover" : "",
          isUploading ? "video-uploader__dropzone--disabled" : "",
        ]
          .filter(Boolean)
          .join(" ")}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={isUploading ? undefined : handleSelectClick}
        role="button"
        tabIndex={0}
        aria-disabled={isUploading}
        aria-label={t("uploader.dropzoneLabel")}
        data-testid="video-uploader-dropzone"
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            if (!isUploading) handleSelectClick();
          }
        }}
      >
        <p className="video-uploader__hint">
          {t("uploader.hint")}
        </p>
        <p className="video-uploader__formats">{t("uploader.formats")}</p>
        <input
          ref={inputRef}
          type="file"
          accept=".mp4,.mov,.avi,video/mp4,video/quicktime,video/x-msvideo"
          onChange={handleFileInput}
          data-testid="video-uploader-input"
          aria-label={t("uploader.inputLabel")}
          style={{ display: "none" }}
        />
      </div>

      {state.warningMessage ? (
        <p className="video-uploader__warning" role="status">
          {state.warningMessage}
        </p>
      ) : null}

      {state.fileName && state.stage !== "idle" ? (
        <div className="video-uploader__status">
          <UploadProgress
            percent={state.percent}
            label={`${state.fileName} - ${stageLabel(state.stage, t)}`}
            error={state.errorMessage}
          />
        </div>
      ) : null}

      {state.metadata ? (
        <div className="video-uploader__metadata">
          <VideoMetadataDisplay metadata={state.metadata} />
        </div>
      ) : null}

      {qualityResult ? (
        <div className="video-uploader__quality">
          <QualityCheckResult result={qualityResult} />
        </div>
      ) : null}

      {state.stage === "error" || state.stage === "complete" ? (
        <button
          type="button"
          className="video-uploader__reset"
          onClick={reset}
          data-testid="video-uploader-reset"
        >
          {state.stage === "error" ? t("uploader.retry") : t("uploader.uploadAnother")}
        </button>
      ) : null}
    </div>
  );
}

export default VideoUploader;
