import { Route, Routes } from "react-router-dom";
import { useTranslation } from "./i18n";
import { AnalysisPage } from "./pages/AnalysisPage";
import { UploadPage } from "./pages/UploadPage";

function LanguageSwitcher() {
  const { language, setLanguage, t } = useTranslation();

  return (
    <div
      aria-label={t("app.language")}
      style={{
        display: "flex",
        justifyContent: "flex-end",
        gap: "0.5rem",
        maxWidth: 960,
        margin: "0 auto",
        padding: "1rem 1rem 0",
      }}
    >
      <button
        type="button"
        onClick={() => setLanguage("ko")}
        aria-pressed={language === "ko"}
        style={{
          padding: "0.4rem 0.75rem",
          border: "1px solid #d1d5db",
          borderRadius: 4,
          background: language === "ko" ? "#111827" : "#fff",
          color: language === "ko" ? "#fff" : "#111827",
        }}
      >
        {t("app.korean")}
      </button>
      <button
        type="button"
        onClick={() => setLanguage("en")}
        aria-pressed={language === "en"}
        style={{
          padding: "0.4rem 0.75rem",
          border: "1px solid #d1d5db",
          borderRadius: 4,
          background: language === "en" ? "#111827" : "#fff",
          color: language === "en" ? "#fff" : "#111827",
        }}
      >
        {t("app.english")}
      </button>
    </div>
  );
}

export function App() {
  return (
    <>
      <LanguageSwitcher />
      <Routes>
        <Route path="/" element={<UploadPage />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/analyses/:analysisId" element={<AnalysisPage />} />
      </Routes>
    </>
  );
}

export default App;
