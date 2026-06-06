import { Route, Routes } from "react-router-dom";
import { AnalysisPage } from "./pages/AnalysisPage";
import { UploadPage } from "./pages/UploadPage";

export function App() {
  return (
    <Routes>
      <Route path="/" element={<UploadPage />} />
      <Route path="/upload" element={<UploadPage />} />
      <Route path="/analyses/:analysisId" element={<AnalysisPage />} />
    </Routes>
  );
}

export default App;
