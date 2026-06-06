/**
 * Tests for analysis API client wrappers.
 */
import { describe, expect, it, vi } from "vitest";
import {
  createAnalysis,
  getAnalysisMetrics,
  getAnalysisOverlay,
  getAnalysisReport,
  getAnalysisStatus,
  getOverlayUrl,
  getUserAnalyses,
  getUserTrends,
} from "./analysis";

function makeFakeAxios<T>(data: T) {
  return {
    get: vi.fn().mockResolvedValue({ data }),
    post: vi.fn().mockResolvedValue({ data }),
  };
}

describe("createAnalysis", () => {
  it("posts to /analyses with file_key + user_id", async () => {
    const fake = makeFakeAxios({
      analysis_id: "id-1",
      status: "pending",
    });
    const result = await createAnalysis(
      "videos/abc.mp4",
      "user-1",
      fake as unknown as Parameters<typeof createAnalysis>[2],
    );
    expect(fake.post).toHaveBeenCalledWith("/analyses", {
      file_key: "videos/abc.mp4",
      user_id: "user-1",
    });
    expect(result.analysis_id).toBe("id-1");
  });
});

describe("getAnalysisStatus", () => {
  it("encodes the id in the URL and gets /status", async () => {
    const fake = makeFakeAxios({
      analysis_id: "x",
      status: "analyzing",
      created_at: "2025-01-01T00:00:00Z",
    });
    await getAnalysisStatus(
      "id 1",
      fake as unknown as Parameters<typeof getAnalysisStatus>[1],
    );
    expect(fake.get).toHaveBeenCalledWith("/analyses/id%201/status");
  });
});

describe("getAnalysisReport", () => {
  it("issues GET /report", async () => {
    const fake = makeFakeAxios({
      analysis_id: "x",
      user_id: "u",
      created_at: "2025-01-01T00:00:00Z",
      status: "completed",
      video_metadata: {},
      quality_check: {},
      swing_phases: [],
      biomechanics: null,
      metric_evaluations: [],
      improvements: [],
      drill_recommendations: [],
    });
    await getAnalysisReport(
      "id-1",
      fake as unknown as Parameters<typeof getAnalysisReport>[1],
    );
    expect(fake.get).toHaveBeenCalledWith("/analyses/id-1/report");
  });
});

describe("getAnalysisOverlay", () => {
  it("issues GET /overlay", async () => {
    const fake = makeFakeAxios({
      analysis_id: "id-1",
      overlay_video_url: "https://example.com/overlay.mp4",
      expires_in: 3600,
    });
    await getAnalysisOverlay(
      "id-1",
      fake as unknown as Parameters<typeof getAnalysisOverlay>[1],
    );
    expect(fake.get).toHaveBeenCalledWith("/analyses/id-1/overlay");
  });

  it("getOverlayUrl is an alias for getAnalysisOverlay", () => {
    expect(getOverlayUrl).toBe(getAnalysisOverlay);
  });
});

describe("getAnalysisMetrics", () => {
  it("issues GET /metrics", async () => {
    const fake = makeFakeAxios({
      analysis_id: "id-1",
      biomechanics: {},
      evaluations: [],
    });
    await getAnalysisMetrics(
      "id-1",
      fake as unknown as Parameters<typeof getAnalysisMetrics>[1],
    );
    expect(fake.get).toHaveBeenCalledWith("/analyses/id-1/metrics");
  });
});

describe("getUserAnalyses", () => {
  it("issues GET /users/{id}/analyses with pagination params", async () => {
    const fake = makeFakeAxios({
      items: [],
      total: 0,
      page: 1,
      page_size: 10,
      has_next: false,
    });
    await getUserAnalyses(
      "u 1",
      2,
      10,
      fake as unknown as Parameters<typeof getUserAnalyses>[3],
    );
    expect(fake.get).toHaveBeenCalledWith("/users/u%201/analyses", {
      params: { page: 2, page_size: 10 },
    });
  });
});

describe("getUserTrends", () => {
  it("issues GET /users/{id}/trends", async () => {
    const fake = makeFakeAxios({
      metrics_history: {},
      total_recordings: 0,
    });
    await getUserTrends(
      "u-1",
      fake as unknown as Parameters<typeof getUserTrends>[1],
    );
    expect(fake.get).toHaveBeenCalledWith("/users/u-1/trends");
  });
});
