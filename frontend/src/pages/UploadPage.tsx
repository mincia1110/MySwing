import { useCallback, useState } from "react";
import { useNavigate } from "react-router-dom";
import { createAnalysis } from "../api/analysis";
import { UserProfileForm } from "../components/UserProfileForm";
import { VideoUploader } from "../components/VideoUploader";
import { useTranslation } from "../i18n";
import type { VideoMetadataWithThumbnailResponse } from "../types/video";

type Step = "upload" | "profile" | "starting";

// Default user ID for MVP (single-user mode)
const DEFAULT_USER_ID = "00000000-0000-0000-0000-000000000001";

export function UploadPage() {
  const navigate = useNavigate();
  const { t } = useTranslation();
  const [step, setStep] = useState<Step>("upload");
  const [fileKey, setFileKey] = useState<string | null>(null);
  const [, setMetadata] = useState<VideoMetadataWithThumbnailResponse | null>(null);
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
    setError(null);

    try {
      const result = await createAnalysis(fileKey, DEFAULT_USER_ID);
      navigate(`/analyses/${result.analysis_id}?userId=${DEFAULT_USER_ID}`);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : t("uploadPage.startError");
      setError(message);
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
    <main style={{ maxWidth: 800, margin: "0 auto", padding: "2rem 1rem" }}>
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
          <h2>{t("uploadPage.profileTitle")}</h2>
          <p>{t("uploadPage.profileIntro")}</p>
          <UserProfileForm
            userId={DEFAULT_USER_ID}
            onSaved={handleProfileSaved}
          />
          <div style={{ marginTop: "1rem", display: "flex", gap: "1rem" }}>
            <button
              type="button"
              onClick={handleSkipProfile}
              disabled={isStarting}
              style={{
                padding: "0.75rem 1.5rem",
                backgroundColor: "#6c757d",
                color: "white",
                border: "none",
                borderRadius: "4px",
                cursor: isStarting ? "not-allowed" : "pointer",
              }}
            >
              {t("uploadPage.skipAndStart")}
            </button>
          </div>
          {error && (
            <p style={{ color: "red", marginTop: "0.5rem" }}>{error}</p>
          )}
        </>
      )}

      {step === "starting" && (
        <div style={{ textAlign: "center", padding: "2rem" }}>
          <p>{t("uploadPage.starting")}</p>
        </div>
      )}
    </main>
  );
}

export default UploadPage;
