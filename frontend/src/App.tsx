import { Route, Routes } from "react-router-dom";
import { useTranslation } from "./i18n";
import { AnalysisPage } from "./pages/AnalysisPage";
import { UploadPage } from "./pages/UploadPage";

function LanguageSwitcher() {
  const { language, setLanguage, t } = useTranslation();

  return (
    <div aria-label={t("app.language")} className="language-switcher">
      <button
        type="button"
        onClick={() => setLanguage("ko")}
        aria-pressed={language === "ko"}
        className={[
          "language-switcher__button",
          language === "ko" ? "language-switcher__button--active" : "",
        ]
          .filter(Boolean)
          .join(" ")}
      >
        {t("app.korean")}
      </button>
      <button
        type="button"
        onClick={() => setLanguage("en")}
        aria-pressed={language === "en"}
        className={[
          "language-switcher__button",
          language === "en" ? "language-switcher__button--active" : "",
        ]
          .filter(Boolean)
          .join(" ")}
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
