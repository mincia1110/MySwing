import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import type { VideoMetadataWithThumbnailResponse } from "../types/video";
import { UploadPage } from "./UploadPage";

vi.mock("../api/analysis", () => ({
  createAnalysis: vi.fn(),
}));

vi.mock("../components/UserProfileForm", () => ({
  UserProfileForm: () => <div data-testid="mock-profile-form" />,
}));

vi.mock("../components/VideoUploader", () => ({
  VideoUploader: ({
    onUploadComplete,
  }: {
    onUploadComplete?: (result: {
      fileKey: string;
      metadata: VideoMetadataWithThumbnailResponse;
    }) => void;
  }) => (
    <button
      type="button"
      onClick={() =>
        onUploadComplete?.({
          fileKey: "uploads/user-1/swing.mp4",
          metadata: {
            file_name: "swing.mp4",
            duration_seconds: 5,
            resolution: { width: 1280, height: 720 },
            file_size_bytes: 1_048_576,
            thumbnail_url: null,
          },
        })
      }
    >
      finish upload
    </button>
  ),
}));

describe("UploadPage", () => {
  it("keeps the uploaded video details visible on the profile step", async () => {
    const user = userEvent.setup();

    render(
      <MemoryRouter>
        <UploadPage />
      </MemoryRouter>,
    );

    await user.click(screen.getByRole("button", { name: "finish upload" }));

    expect(
      screen.getByRole("heading", { name: "업로드한 영상" }),
    ).toBeInTheDocument();
    expect(screen.getByTestId("video-metadata-filename")).toHaveTextContent(
      "swing.mp4",
    );
    expect(screen.getByTestId("mock-profile-form")).toBeInTheDocument();
    expect(
      screen.getByText("분석을 시작하려면 프로필 정보를 입력하세요."),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "건너뛰고 분석 시작" }),
    ).not.toBeInTheDocument();
  });
});
