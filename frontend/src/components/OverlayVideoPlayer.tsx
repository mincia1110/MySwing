/**
 * Plays the analysis overlay video that shows pose skeleton + bat trajectory
 * (Requirement 8.2).
 */
import "./OverlayVideoPlayer.css";

export interface OverlayVideoPlayerProps {
  /** Presigned S3 URL for the overlay video. May be null when unavailable. */
  videoUrl?: string | null;
  /** Optional poster image. */
  posterUrl?: string | null;
  /** Optional title shown above the player. */
  title?: string;
}

export function OverlayVideoPlayer({
  videoUrl,
  posterUrl,
  title = "오버레이 비디오",
}: OverlayVideoPlayerProps) {
  return (
    <section
      className="overlay-video"
      aria-label="오버레이 비디오 플레이어"
      data-testid="overlay-video"
    >
      <h3 className="overlay-video__title">{title}</h3>
      {videoUrl ? (
        <video
          className="overlay-video__player"
          src={videoUrl}
          poster={posterUrl ?? undefined}
          controls
          preload="metadata"
          data-testid="overlay-video-element"
        >
          이 브라우저는 비디오 재생을 지원하지 않습니다.
        </video>
      ) : (
        <div
          className="overlay-video__placeholder"
          data-testid="overlay-video-placeholder"
          role="status"
        >
          오버레이 비디오를 사용할 수 없습니다.
        </div>
      )}
    </section>
  );
}

export default OverlayVideoPlayer;
