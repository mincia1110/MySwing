import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import {
  VideoMetadataDisplay,
  formatDuration,
  formatFileSize,
} from "./VideoMetadataDisplay";
import type { VideoMetadataWithThumbnailResponse } from "../types/video";

const baseMetadata: VideoMetadataWithThumbnailResponse = {
  file_name: "swing.mp4",
  duration_seconds: 12.4,
  resolution: { width: 1920, height: 1080 },
  file_size_bytes: 5_242_880,
  thumbnail_url: "https://example.com/thumb.jpg",
};

describe("formatFileSize", () => {
  it("formats small sizes in bytes", () => {
    expect(formatFileSize(512)).toBe("512 B");
  });

  it("formats kilobytes", () => {
    expect(formatFileSize(2048)).toBe("2.00 KB");
  });

  it("formats megabytes with decimals", () => {
    expect(formatFileSize(5_242_880)).toBe("5.00 MB");
  });

  it("formats gigabytes", () => {
    expect(formatFileSize(2.5 * 1024 ** 3)).toBe("2.50 GB");
  });

  it("handles invalid values gracefully", () => {
    expect(formatFileSize(-1)).toBe("0 B");
    expect(formatFileSize(NaN)).toBe("0 B");
  });
});

describe("formatDuration", () => {
  it("formats seconds as M:SS", () => {
    expect(formatDuration(12.4)).toBe("0:12");
    expect(formatDuration(125)).toBe("2:05");
  });

  it("handles invalid values", () => {
    expect(formatDuration(-1)).toBe("0:00");
    expect(formatDuration(NaN)).toBe("0:00");
  });
});

describe("VideoMetadataDisplay", () => {
  it("renders thumbnail and metadata fields", () => {
    render(<VideoMetadataDisplay metadata={baseMetadata} />);

    const thumbnail = screen.getByTestId("video-metadata-thumbnail");
    expect(thumbnail).toHaveAttribute("src", "https://example.com/thumb.jpg");
    expect(screen.getByTestId("video-metadata-filename")).toHaveTextContent(
      "swing.mp4",
    );
    expect(screen.getByTestId("video-metadata-duration")).toHaveTextContent("0:12");
    expect(screen.getByTestId("video-metadata-resolution")).toHaveTextContent(
      "1920 × 1080",
    );
    expect(screen.getByTestId("video-metadata-filesize")).toHaveTextContent(
      "5.00 MB",
    );
  });

  it("renders a placeholder when no thumbnail URL is provided", () => {
    render(
      <VideoMetadataDisplay
        metadata={{ ...baseMetadata, thumbnail_url: null }}
      />,
    );

    expect(
      screen.getByTestId("video-metadata-thumbnail-placeholder"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("video-metadata-thumbnail"),
    ).not.toBeInTheDocument();
  });
});
