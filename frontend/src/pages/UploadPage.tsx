import { useCallback, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createAnalysis } from "../api/analysis";
import { UserProfileForm } from "../components/UserProfileForm";
import { VideoMetadataDisplay } from "../components/VideoMetadataDisplay";
import { VideoUploader } from "../components/VideoUploader";
import { useTranslation } from "../i18n";
import type { VideoMetadataWithThumbnailResponse } from "../types/video";

type Step = "upload" | "profile" | "starting";

export function UploadPage() {
  const navigate = useNavigate();
  const { t } = useTranslation();
  const [step, setStep] = useState<Step>("upload");
  const [fileKey, setFileKey] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<VideoMetadataWithThumbnailResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isStarting, setIsStarting] = useState(false);

  const handleUploadComplete = useCallback(
    (result: { fileKey: string; metadata: VideoMetadataWithThumbnailResponse }) => {
      setFileKey(result.fileKey);
      setMetadata(result.metadata);
      setStep("profile");
    },
    [],
  );

  const handleStartAnalysis = useCallback(async () => {
    if (!fileKey) return;

    setIsStarting(true);
    setStep("starting");
    setError(null);

    try {
      const result = await createAnalysis(fileKey);
      navigate(`/analyses/${result.analysis_id}`);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : t("uploadPage.startError");
      setError(message);
      setStep("profile");
      setIsStarting(false);
    }
  }, [fileKey, navigate, t]);

  const handleSkipProfile = useCallback(() => {
    void handleStartAnalysis();
  }, [handleStartAnalysis]);

  const handleProfileSaved = useCallback(() => {
    void handleStartAnalysis();
  }, [handleStartAnalysis]);

  return (
    <main className="page page--upload">
      <h1>{t("uploadPage.title")}</h1>

      {step === "upload" && (
        <>
          <p>{t("uploadPage.intro")}</p>
          <VideoUploader
            onUploadComplete={handleUploadComplete}
            onUploadError={(err) => console.error("upload failed", err)}
          />
        </>
      )}

      {step === "profile" && (
        <>
          {metadata ? (
            <section
              className="upload-page__uploaded-video"
              aria-label={t("uploadPage.uploadedVideo")}
            >
              <h2>{t("uploadPage.uploadedVideo")}</h2>
              <VideoMetadataDisplay metadata={metadata} />
            </section>
          ) : null}
          <h2>{t("uploadPage.profileTitle")}</h2>
          <p>{t("uploadPage.profileIntro")}</p>
          <UserProfileForm onSaved={handleProfileSaved} />
          <div className="page__actions">
            <button
              type="button"
              onClick={handleSkipProfile}
              disabled={isStarting}
              className="button button--secondary"
            >
              {t("uploadPage.skipAndStart")}
            </button>
          </div>
          {error && (
            <p className="page__error" role="alert">
              {error}
            </p>
          )}
        </>
      )}

      {step === "starting" && (
        <div className="page__loading">
          <p>{t("uploadPage.starting")}</p>
        </div>
      )}
    </main>
  );
}

export default UploadPage;
