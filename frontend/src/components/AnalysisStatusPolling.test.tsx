import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AnalysisStatusPolling } from "./AnalysisStatusPolling";
import type { AnalysisStatusResponse } from "../types/analysis";

function makeStatus(
  overrides: Partial<AnalysisStatusResponse> = {},
): AnalysisStatusResponse {
  return {
    analysis_id: "abc",
    status: "preprocessing",
    error_message: null,
    started_at: null,
    completed_at: null,
    created_at: "2025-01-01T00:00:00Z",
    ...overrides,
  };
}

/**
 * Poll using a 5ms interval so tests stay fast under real timers.
 * Real timers avoid the microtask/timer interleaving issues with vitest's
 * fake timers when the polled function returns a Promise.
 */
const POLL_INTERVAL_MS = 5;

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

describe("AnalysisStatusPolling", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("polls repeatedly while status is non-terminal", async () => {
    const fetchStatus = vi
      .fn()
      .mockResolvedValueOnce(makeStatus({ status: "preprocessing" }))
      .mockResolvedValueOnce(makeStatus({ status: "analyzing" }))
      .mockResolvedValueOnce(makeStatus({ status: "evaluating" }))
      // After we observe 3 calls we still need to keep returning non-terminal
      // statuses so any pending poll resolves cleanly when the component
      // unmounts.
      .mockResolvedValue(makeStatus({ status: "evaluating" }));

    const { unmount } = render(
      <AnalysisStatusPolling
        analysisId="abc"
        pollIntervalMs={POLL_INTERVAL_MS}
        fetchStatus={fetchStatus}
      />,
    );

    await waitFor(() => expect(fetchStatus.mock.calls.length).toBeGreaterThanOrEqual(3));
    expect(screen.getByTestId("analysis-status")).toHaveAttribute(
      "data-status",
      "evaluating",
    );

    unmount();
  });

  it("stops polling once status reaches completed", async () => {
    const fetchStatus = vi
      .fn()
      .mockResolvedValueOnce(makeStatus({ status: "analyzing" }))
      .mockResolvedValue(makeStatus({ status: "completed" }));
    const onCompleted = vi.fn();
    const onFailed = vi.fn();

    render(
      <AnalysisStatusPolling
        analysisId="abc"
        pollIntervalMs={POLL_INTERVAL_MS}
        fetchStatus={fetchStatus}
        onCompleted={onCompleted}
        onFailed={onFailed}
      />,
    );

    await waitFor(() => expect(onCompleted).toHaveBeenCalledTimes(1));

    const callsAtCompletion = fetchStatus.mock.calls.length;
    // Wait long enough that several intervals would have elapsed.
    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS * 10));

    expect(fetchStatus.mock.calls.length).toBe(callsAtCompletion);
    expect(onFailed).not.toHaveBeenCalled();
    expect(screen.getByTestId("analysis-status")).toHaveAttribute(
      "data-terminal",
      "true",
    );
    expect(screen.getByTestId("analysis-status-percent")).toHaveTextContent(
      "100%",
    );
  });

  it("stops polling once status reaches failed", async () => {
    const fetchStatus = vi
      .fn()
      .mockResolvedValueOnce(
        makeStatus({ status: "failed", error_message: "포즈 추적 실패" }),
      )
      .mockResolvedValue(makeStatus({ status: "failed" }));
    const onCompleted = vi.fn();
    const onFailed = vi.fn();

    render(
      <AnalysisStatusPolling
        analysisId="abc"
        pollIntervalMs={POLL_INTERVAL_MS}
        fetchStatus={fetchStatus}
        onCompleted={onCompleted}
        onFailed={onFailed}
      />,
    );

    await waitFor(() => expect(onFailed).toHaveBeenCalledTimes(1));

    const callsAtFailure = fetchStatus.mock.calls.length;
    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS * 10));

    expect(fetchStatus.mock.calls.length).toBe(callsAtFailure);
    expect(onCompleted).not.toHaveBeenCalled();
    expect(screen.getByTestId("analysis-status-error")).toHaveTextContent(
      "포즈 추적 실패",
    );
  });

  it("retries transient polling errors before failing", async () => {
    const fetchStatus = vi
      .fn()
      .mockRejectedValueOnce(new Error("temporary network error"))
      .mockResolvedValueOnce(makeStatus({ status: "completed" }));
    const onCompleted = vi.fn();
    const onFailed = vi.fn();

    render(
      <AnalysisStatusPolling
        analysisId="abc"
        pollIntervalMs={POLL_INTERVAL_MS}
        fetchStatus={fetchStatus}
        onCompleted={onCompleted}
        onFailed={onFailed}
      />,
    );

    await waitFor(() => expect(onCompleted).toHaveBeenCalledTimes(1));
    expect(onFailed).not.toHaveBeenCalled();
    expect(fetchStatus).toHaveBeenCalledTimes(2);
  });

  it("ignores a stale response from a previous analysis id", async () => {
    const first = deferred<AnalysisStatusResponse>();
    const fetchStatus = vi.fn((analysisId: string) => {
      if (analysisId === "abc") return first.promise;
      return Promise.resolve(makeStatus({ analysis_id: analysisId, status: "completed" }));
    });
    const onCompleted = vi.fn();

    const { rerender } = render(
      <AnalysisStatusPolling
        analysisId="abc"
        pollIntervalMs={POLL_INTERVAL_MS}
        fetchStatus={fetchStatus}
        onCompleted={onCompleted}
      />,
    );

    rerender(
      <AnalysisStatusPolling
        analysisId="def"
        pollIntervalMs={POLL_INTERVAL_MS}
        fetchStatus={fetchStatus}
        onCompleted={onCompleted}
      />,
    );

    await waitFor(() => expect(onCompleted).toHaveBeenCalledTimes(1));
    first.resolve(makeStatus({ analysis_id: "abc", status: "completed" }));
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS * 2));

    expect(onCompleted).toHaveBeenCalledTimes(1);
    expect(onCompleted.mock.calls[0][0].analysis_id).toBe("def");
  });

  it("invokes onFailed when fetch raises an error", async () => {
    const fetchStatus = vi.fn().mockRejectedValue(new Error("network error"));
    const onFailed = vi.fn();

    render(
      <AnalysisStatusPolling
        analysisId="abc"
        pollIntervalMs={POLL_INTERVAL_MS}
        fetchStatus={fetchStatus}
        onFailed={onFailed}
        maxErrorRetries={0}
      />,
    );

    await waitFor(() => expect(onFailed).toHaveBeenCalled());
    const callArgs = onFailed.mock.calls[0];
    expect(callArgs[0]).toBeNull();
    expect(callArgs[1]?.message).toBe("network error");
  });
});
