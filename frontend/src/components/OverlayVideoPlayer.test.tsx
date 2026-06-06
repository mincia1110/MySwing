import { render, screen } from "@testing-library/react";
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
});
