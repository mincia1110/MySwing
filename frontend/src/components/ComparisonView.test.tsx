import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ComparisonView } from "./ComparisonView";
import type { SwingPhaseResponse } from "../types/analysis";

/**
 * Reads the component stylesheet from disk. Vitest stubs CSS imports
 * (test.css = false), so the mobile layout contract is verified against the
 * file contents instead. Node type declarations are not installed, hence the
 * scoped suppressions; vitest provides these modules at runtime.
 */
async function readComparisonCss(): Promise<string> {
  // @ts-expect-error - node:fs type declarations are not installed
  const { readFileSync } = (await import("node:fs")) as {
    readFileSync: (path: string, encoding: string) => string;
  };
  // @ts-expect-error - node:path type declarations are not installed
  const { dirname, join } = (await import("node:path")) as {
    dirname: (path: string) => string;
    join: (...paths: string[]) => string;
  };
  // @ts-expect-error - node:url type declarations are not installed
  const { fileURLToPath } = (await import("node:url")) as {
    fileURLToPath: (url: string | URL) => string;
  };
  const testFilePath = fileURLToPath(import.meta.url);
  const cssPath = join(dirname(testFilePath), "ComparisonView.css");
  return readFileSync(cssPath, "utf8");
}

const userPhases: SwingPhaseResponse[] = [
  { phase: "stance", start_frame: 0, end_frame: 6, duration_ms: 200 },
  { phase: "load", start_frame: 6, end_frame: 16, duration_ms: 333 },
  { phase: "rotation", start_frame: 16, end_frame: 21, duration_ms: 167 },
];

describe("ComparisonView", () => {
  it("renders one row per phase merging user + reference phases", () => {
    render(
      <ComparisonView
        userPhases={userPhases}
        referencePhases={[
          { phase: "stance", duration_ms: 200 },
          { phase: "load", duration_ms: 300 },
          { phase: "rotation", duration_ms: 150 },
        ]}
      />,
    );

    expect(screen.getByTestId("comparison-row-stance")).toBeInTheDocument();
    expect(screen.getByTestId("comparison-row-load")).toBeInTheDocument();
    expect(screen.getByTestId("comparison-row-rotation")).toBeInTheDocument();
  });

  it("displays user and reference durations", () => {
    render(
      <ComparisonView
        userPhases={userPhases}
        referencePhases={[
          { phase: "stance", duration_ms: 200 },
          { phase: "load", duration_ms: 300 },
        ]}
      />,
    );

    expect(screen.getByTestId("comparison-user-stance")).toHaveTextContent(
      "200",
    );
    expect(screen.getByTestId("comparison-reference-stance")).toHaveTextContent(
      "200",
    );
    expect(screen.getByTestId("comparison-user-load")).toHaveTextContent("333");
    expect(screen.getByTestId("comparison-reference-load")).toHaveTextContent(
      "300",
    );
    // Phase only in user data should still render with reference dash.
    expect(
      screen.getByTestId("comparison-reference-rotation"),
    ).toHaveTextContent("-");
  });

  it("uses default reference set when none is supplied", () => {
    render(<ComparisonView userPhases={userPhases} />);
    expect(screen.getByTestId("comparison-row-impact")).toBeInTheDocument();
    expect(screen.getByTestId("comparison-row-follow_through")).toHaveTextContent(
      "팔로스루",
    );
  });

  it("renders an empty state when both arrays are empty", () => {
    render(<ComparisonView userPhases={[]} referencePhases={[]} />);
    expect(screen.getByTestId("comparison-view-empty")).toBeInTheDocument();
  });

  it("keeps the comparison bars present for every phase row", () => {
    render(<ComparisonView userPhases={userPhases} />);

    for (const phase of ["stance", "load", "rotation"]) {
      const row = screen.getByTestId(`comparison-row-${phase}`);
      expect(
        row.querySelectorAll(".comparison-view__bar-fill"),
      ).toHaveLength(2);
    }
  });

  it("uses a compact mobile layout instead of hiding comparison content", async () => {
    const css = await readComparisonCss();
    const mediaIndex = css.indexOf("@media (max-width: 640px)");
    expect(mediaIndex).toBeGreaterThanOrEqual(0);
    const mobileBlock = css.slice(mediaIndex);
    // Narrow screens must not hide any comparison content.
    expect(css).not.toMatch(/display:\s*none/);
    // The fixed desktop min-width is relaxed so bars stay visible at 390px.
    expect(mobileBlock).toContain("min-width: 0");
    expect(mobileBlock).toContain("overflow-x: visible");
  });
});
