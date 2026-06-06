/**
 * Plays the analysis overlay video that shows pose skeleton + bat trajectory
 * (Requirement 8.2).
 */
import { useTranslation } from "../i18n";
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
  title,
}: OverlayVideoPlayerProps) {
  const { t } = useTranslation();
  const resolvedTitle = title ?? t("overlay.title");

  return (
    <section
      className="overlay-video"
      aria-label={t("overlay.aria")}
      data-testid="overlay-video"
    >
      <h3 className="overlay-video__title">{resolvedTitle}</h3>
      {videoUrl ? (
        <video
          className="overlay-video__player"
          src={videoUrl}
          poster={posterUrl ?? undefined}
          controls
          preload="metadata"
          data-testid="overlay-video-element"
        >
          {t("overlay.unsupported")}
        </video>
      ) : (
        <div
          className="overlay-video__placeholder"
          data-testid="overlay-video-placeholder"
          role="status"
        >
          {t("overlay.unavailable")}
        </div>
      )}
    </section>
  );
}

export default OverlayVideoPlayer;
