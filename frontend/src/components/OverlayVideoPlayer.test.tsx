import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { OverlayVideoPlayer } from "./OverlayVideoPlayer";

describe("OverlayVideoPlayer", () => {
  it("renders a video element when a URL is provided", () => {
    render(
      <OverlayVideoPlayer
        videoUrl="https://example.com/overlay.mp4"
        posterUrl="https://example.com/poster.jpg"
      />,
    );
    const video = screen.getByTestId("overlay-video-element") as HTMLVideoElement;
    expect(video).toBeInTheDocument();
    expect(video).toHaveAttribute("src", "https://example.com/overlay.mp4");
    expect(video).toHaveAttribute("poster", "https://example.com/poster.jpg");
    expect(video).toHaveAttribute("controls");
  });

  it("renders a placeholder when no URL is provided", () => {
    render(<OverlayVideoPlayer videoUrl={null} />);
    expect(
      screen.getByTestId("overlay-video-placeholder"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("overlay-video-element"),
    ).not.toBeInTheDocument();
  });

  it("jumps to a swing phase start time", async () => {
    const user = userEvent.setup();
    render(
      <OverlayVideoPlayer
        videoUrl="https://example.com/overlay.mp4"
        fps={60}
        phases={[
          { phase: "stance", start_frame: 0, end_frame: 5, duration_ms: 83 },
          { phase: "load", start_frame: 12, end_frame: 30, duration_ms: 300 },
        ]}
      />,
    );

    const video = screen.getByTestId("overlay-video-element") as HTMLVideoElement;
    await user.click(screen.getByRole("button", { name: "로드 구간으로 이동" }));

    expect(video.currentTime).toBeCloseTo(0.2);
  });
});
