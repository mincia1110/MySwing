import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { VideoUploader } from "./VideoUploader";
import * as apiClient from "../api/client";
import type {
  PresignedUrlResponse,
  VideoMetadataWithThumbnailResponse,
} from "../types/video";

const samplePresigned: PresignedUrlResponse = {
  upload_url: "https://s3.example.com/upload",
  file_key: "videos/abc-123.mp4",
  expires_in: 3600,
};

const sampleMetadata: VideoMetadataWithThumbnailResponse = {
  file_name: "swing.mp4",
  duration_seconds: 8.5,
  resolution: { width: 1920, height: 1080 },
  file_size_bytes: 1024 * 1024,
  thumbnail_url: "https://example.com/thumb.jpg",
};

function makeFile(
  name = "swing.mp4",
  type = "video/mp4",
  size = 1024 * 1024,
): File {
  const file = new File(["x".repeat(16)], name, { type });
  // Override size for upload validation tests.
  Object.defineProperty(file, "size", { value: size });
  return file;
}

describe("VideoUploader", () => {
  beforeEach(() => {
    vi.spyOn(apiClient, "getPresignedUrl").mockResolvedValue(samplePresigned);
    vi.spyOn(apiClient, "uploadToS3").mockImplementation(
      (_url, file, options) => {
        options?.onProgress?.({
          loaded: file.size,
          total: file.size,
          percent: 100,
        });
        return Promise.resolve();
      },
    );
    vi.spyOn(apiClient, "getMetadata").mockResolvedValue(sampleMetadata);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders the dropzone with hint text", () => {
    render(<VideoUploader />);
    expect(screen.getByTestId("video-uploader-dropzone")).toBeInTheDocument();
    expect(
      screen.getByText(/한 번의 스윙만 담긴 짧은 영상/),
    ).toBeInTheDocument();
  });

  it("rejects files with unsupported MIME types", async () => {
    const onError = vi.fn();
    render(<VideoUploader onUploadError={onError} />);

    const input = screen.getByTestId("video-uploader-input") as HTMLInputElement;
    const badFile = makeFile("doc.pdf", "application/pdf", 100);
    fireEvent.change(input, { target: { files: [badFile] } });

    await waitFor(() => {
      expect(onError).toHaveBeenCalledWith(
        expect.objectContaining({ message: expect.stringContaining("지원하지 않는") }),
      );
    });

    expect(apiClient.getPresignedUrl).not.toHaveBeenCalled();
  });

  it("rejects files larger than the maximum size", async () => {
    const onError = vi.fn();
    render(
      <VideoUploader
        onUploadError={onError}
        maxFileSizeBytes={1024 * 1024}
      />,
    );

    const input = screen.getByTestId("video-uploader-input") as HTMLInputElement;
    const bigFile = makeFile("swing.mp4", "video/mp4", 10 * 1024 * 1024);
    fireEvent.change(input, { target: { files: [bigFile] } });

    await waitFor(() => {
      expect(onError).toHaveBeenCalledWith(
        expect.objectContaining({ message: expect.stringContaining("너무 큽니다") }),
      );
    });
  });


  it("blocks videos longer than 10 seconds before requesting an upload URL", async () => {
    const onError = vi.fn();
    render(<VideoUploader onUploadError={onError} />);

    const input = screen.getByTestId("video-uploader-input") as HTMLInputElement;
    const longFile = makeFile("long-session.mp4", "video/mp4", 1024 * 1024);
    Object.defineProperty(longFile, "duration", { value: 11 });
    fireEvent.change(input, { target: { files: [longFile] } });

    await waitFor(() => {
      expect(onError).toHaveBeenCalledWith(
        expect.objectContaining({ message: expect.stringContaining("10초") }),
      );
    });

    expect(apiClient.getPresignedUrl).not.toHaveBeenCalled();
  });


  it("shows a warning but uploads when client-side duration is shorter than recommended", async () => {
    const onComplete = vi.fn();
    render(<VideoUploader onUploadComplete={onComplete} />);

    const input = screen.getByTestId("video-uploader-input") as HTMLInputElement;
    const shortFile = makeFile("short-swing.mp4", "video/mp4", 1024 * 1024);
    Object.defineProperty(shortFile, "duration", { value: 2 });
    fireEvent.change(input, { target: { files: [shortFile] } });

    expect(
      await screen.findByText(/권장 길이는 3~7초입니다/),
    ).toBeInTheDocument();

    await waitFor(() => {
      expect(onComplete).toHaveBeenCalled();
    });
    expect(apiClient.getPresignedUrl).toHaveBeenCalled();
  });

  it("continues upload when browser duration metadata is unavailable", async () => {
    const onComplete = vi.fn();
    render(<VideoUploader onUploadComplete={onComplete} />);

    const input = screen.getByTestId("video-uploader-input") as HTMLInputElement;
    fireEvent.change(input, { target: { files: [makeFile("unknown-duration.mp4")] } });

    await waitFor(() => {
      expect(onComplete).toHaveBeenCalledWith({
        fileKey: samplePresigned.file_key,
        metadata: sampleMetadata,
      });
    });
    expect(apiClient.getPresignedUrl).toHaveBeenCalledWith({
      file_name: "unknown-duration.mp4",
      content_type: "video/mp4",
    });
  });

  it("runs the full upload pipeline and notifies onUploadComplete", async () => {
    const onComplete = vi.fn();
    render(<VideoUploader onUploadComplete={onComplete} />);

    const input = screen.getByTestId("video-uploader-input") as HTMLInputElement;
    const goodFile = makeFile();
    fireEvent.change(input, { target: { files: [goodFile] } });

    await waitFor(() => {
      expect(onComplete).toHaveBeenCalledWith({
        fileKey: samplePresigned.file_key,
        metadata: sampleMetadata,
      });
    });

    expect(apiClient.getPresignedUrl).toHaveBeenCalledWith({
      file_name: "swing.mp4",
      content_type: "video/mp4",
    });
    expect(apiClient.uploadToS3).toHaveBeenCalledWith(
      samplePresigned.upload_url,
      goodFile,
      expect.objectContaining({ onProgress: expect.any(Function) }),
    );
    expect(apiClient.getMetadata).toHaveBeenCalledWith(samplePresigned.file_key);

    // Metadata display should appear after completion.
    expect(screen.getByTestId("video-metadata")).toBeInTheDocument();
    expect(screen.getByTestId("video-metadata-filename")).toHaveTextContent(
      "swing.mp4",
    );
  });

  it("blocks completion when server metadata validation rejects the video", async () => {
    const rejectedMetadata: VideoMetadataWithThumbnailResponse = {
      ...sampleMetadata,
      input_validation: {
        accepted: false,
        severity: "error",
        reason: "video_too_long",
        duration_sec: 11,
        ideal_duration_sec: 5,
        recommended_min_duration_sec: 3,
        recommended_max_duration_sec: 7,
        max_duration_sec: 10,
        message: "10초 이하 영상만 분석할 수 있습니다.",
        recommendation: "영상을 잘라 다시 업로드해 주세요.",
      },
    };
    vi.spyOn(apiClient, "getMetadata").mockResolvedValue(rejectedMetadata);
    const onComplete = vi.fn();
    const onError = vi.fn();

    render(<VideoUploader onUploadComplete={onComplete} onUploadError={onError} />);

    const input = screen.getByTestId("video-uploader-input") as HTMLInputElement;
    fireEvent.change(input, { target: { files: [makeFile()] } });

    await waitFor(() => {
      expect(onError).toHaveBeenCalledWith(
        expect.objectContaining({ message: "10초 이하 영상만 분석할 수 있습니다." }),
      );
    });

    expect(onComplete).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent(
      "10초 이하 영상만 분석할 수 있습니다.",
    );
  });

  it("surfaces presigned URL errors via onUploadError", async () => {
    vi.spyOn(apiClient, "getPresignedUrl").mockRejectedValue(
      new Error("Server unreachable"),
    );
    const onError = vi.fn();
    render(<VideoUploader onUploadError={onError} />);

    const input = screen.getByTestId("video-uploader-input") as HTMLInputElement;
    fireEvent.change(input, { target: { files: [makeFile()] } });

    await waitFor(() => {
      expect(onError).toHaveBeenCalledWith(
        expect.objectContaining({ message: "Server unreachable" }),
      );
    });

    expect(apiClient.uploadToS3).not.toHaveBeenCalled();
  });

  it("renders quality check results when supplied", () => {
    render(
      <VideoUploader
        qualityResult={{
          brightness_status: "pass",
          framing_status: "warning",
          resolution_status: "pass",
          frame_rate_stability_status: "pass",
          brightness_value: 55,
          swing_arc_visibility_percent: 75,
          frame_rate_variation_percent: 3.2,
          warnings: ["스윙 아크가 부분적으로 가려졌습니다"],
        }}
      />,
    );

    expect(screen.getByTestId("quality-check")).toBeInTheDocument();
    expect(screen.getByTestId("quality-check-framing")).toHaveAttribute(
      "data-status",
      "warning",
    );
  });
});
