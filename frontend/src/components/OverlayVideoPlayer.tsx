/**
 * Plays the analysis overlay video that shows pose skeleton + bat trajectory
 * (Requirement 8.2).
 */
import { useRef } from "react";
import type { SwingPhaseResponse } from "../types/analysis";
import { useTranslation } from "../i18n";
import "./OverlayVideoPlayer.css";

export interface OverlayVideoPlayerProps {
  /** Presigned S3 URL for the overlay video. May be null when unavailable. */
  videoUrl?: string | null;
  /** Optional poster image. */
  posterUrl?: string | null;
  /** Optional title shown above the player. */
  title?: string;
  /** Swing phases used for quick navigation. */
  phases?: SwingPhaseResponse[];
  /** Frames per second used to map phase frames to video time. */
  fps?: number;
}

export function OverlayVideoPlayer({
  videoUrl,
  posterUrl,
  title,
  phases = [],
  fps = 30,
}: OverlayVideoPlayerProps) {
  const { t } = useTranslation();
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const resolvedTitle = title ?? t("overlay.title");
  const safeFps = Number.isFinite(fps) && fps > 0 ? fps : 30;
  const phaseLabel = (phase: string) => {
    const key = `phases.${phase}`;
    const label = t(key);
    return label === key ? phase : label;
  };

  const jumpToPhase = (phase: SwingPhaseResponse) => {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = Math.max(0, phase.start_frame / safeFps);
  };

  return (
    <section
      className="overlay-video"
      aria-label={t("overlay.aria")}
      data-testid="overlay-video"
    >
      <h3 className="overlay-video__title">{resolvedTitle}</h3>
      {videoUrl ? (
        <>
          <video
            ref={videoRef}
            className="overlay-video__player"
            src={videoUrl}
            poster={posterUrl ?? undefined}
            controls
            preload="metadata"
            data-testid="overlay-video-element"
          >
            {t("overlay.unsupported")}
          </video>
          {phases.length > 0 ? (
            <div className="overlay-video__phases">
              <span className="overlay-video__phases-label">
                {t("overlay.phases")}
              </span>
              <div className="overlay-video__phase-list">
                {phases.map((phase) => (
                  <button
                    key={`${phase.phase}-${phase.start_frame}`}
                    type="button"
                    className="overlay-video__phase-button"
                    onClick={() => jumpToPhase(phase)}
                    aria-label={t("overlay.jumpToPhase", {
                      phase: phaseLabel(phase.phase),
                    })}
                  >
                    {phaseLabel(phase.phase)}
                  </button>
                ))}
              </div>
            </div>
          ) : null}
        </>
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
