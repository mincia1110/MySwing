/**
 * API client for MySwing backend.
 *
 * Provides methods for:
 * - Generating presigned S3 upload URLs (POST /api/v1/upload/presigned-url)
 * - Uploading the file directly to S3 with progress reporting via XMLHttpRequest
 * - Fetching video metadata + thumbnail (POST /api/v1/videos/{file_key}/metadata)
 */

import axios, { AxiosInstance } from "axios";
import type {
  PresignedUrlRequest,
  PresignedUrlResponse,
  VideoMetadataWithThumbnailResponse,
} from "../types/video";

export const API_BASE_URL =
  (import.meta.env?.VITE_API_BASE_URL as string | undefined) ?? "/api/v1";

export interface UploadProgressEvent {
  loaded: number;
  total: number;
  percent: number;
}

export interface UploadOptions {
  onProgress?: (event: UploadProgressEvent) => void;
  /** Optional AbortSignal to cancel the upload. */
  signal?: AbortSignal;
  /** Content-Type to send with the upload (must match presigned URL). */
  contentType?: string;
}

/**
 * Build a configured axios instance for backend API calls.
 */
export function createApiClient(baseURL: string = API_BASE_URL): AxiosInstance {
  return axios.create({
    baseURL,
    headers: {
      "Content-Type": "application/json",
    },
    timeout: 30_000,
  });
}

const defaultClient = createApiClient();

/**
 * Request a presigned S3 upload URL from the backend.
 */
export async function getPresignedUrl(
  request: PresignedUrlRequest,
  client: AxiosInstance = defaultClient,
): Promise<PresignedUrlResponse> {
  const response = await client.post<PresignedUrlResponse>(
    "/upload/presigned-url",
    request,
  );
  return response.data;
}

/**
 * Upload a file directly to S3 using a presigned URL.
 *
 * Uses XMLHttpRequest to report progress events (axios browser progress
 * support is limited and inconsistent).
 */
export function uploadToS3(
  uploadUrl: string,
  file: File,
  options: UploadOptions = {},
): Promise<void> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();

    xhr.open("PUT", uploadUrl, true);
    // Always set Content-Type to match what was used to generate the presigned URL.
    // If contentType option is provided, use it; otherwise fall back to file.type.
    const ct = options.contentType || file.type;
    if (ct) {
      xhr.setRequestHeader("Content-Type", ct);
    }

    xhr.upload.onprogress = (event: ProgressEvent) => {
      if (!options.onProgress) {
        return;
      }
      const total = event.lengthComputable ? event.total : file.size;
      const percent =
        total > 0 ? Math.min(100, Math.round((event.loaded / total) * 100)) : 0;
      options.onProgress({ loaded: event.loaded, total, percent });
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        // Final 100% progress event so UIs always converge on 100.
        options.onProgress?.({
          loaded: file.size,
          total: file.size,
          percent: 100,
        });
        resolve();
        return;
      }
      reject(
        new Error(
          `Upload failed with status ${xhr.status}: ${xhr.responseText || xhr.statusText}`,
        ),
      );
    };

    xhr.onerror = () => {
      reject(new Error("Network error during S3 upload"));
    };

    xhr.onabort = () => {
      reject(new DOMException("Upload aborted", "AbortError"));
    };

    if (options.signal) {
      if (options.signal.aborted) {
        xhr.abort();
        return;
      }
      options.signal.addEventListener("abort", () => xhr.abort(), {
        once: true,
      });
    }

    xhr.send(file);
  });
}

/**
 * Trigger backend metadata extraction + thumbnail generation for an
 * uploaded file_key. Backend must respond within 5s per Requirement 1.7.
 */
export async function getMetadata(
  fileKey: string,
  client: AxiosInstance = defaultClient,
): Promise<VideoMetadataWithThumbnailResponse> {
  const encodedKey = fileKey
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
  const response = await client.post<VideoMetadataWithThumbnailResponse>(
    `/videos/${encodedKey}/metadata`,
  );
  return response.data;
}

export const apiClient = defaultClient;
