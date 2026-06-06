import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { UploadProgress } from "./UploadProgress";

describe("UploadProgress", () => {
  it("clamps progress within [0, 100] and renders percentage", () => {
    render(<UploadProgress percent={45.7} label="uploading" />);
    expect(screen.getByTestId("upload-progress-percent")).toHaveTextContent("46%");
    const fill = screen.getByTestId("upload-progress-fill");
    expect(fill).toHaveStyle({ width: "46%" });
  });

  it("clamps negative values to 0", () => {
    render(<UploadProgress percent={-10} />);
    expect(screen.getByTestId("upload-progress-percent")).toHaveTextContent("0%");
  });

  it("clamps values above 100 to 100", () => {
    render(<UploadProgress percent={150} />);
    expect(screen.getByTestId("upload-progress-percent")).toHaveTextContent("100%");
  });

  it("renders an error message when provided", () => {
    render(<UploadProgress percent={50} error="Network error" />);
    expect(screen.getByRole("alert")).toHaveTextContent("Network error");
    expect(screen.getByTestId("upload-progress")).toHaveClass(
      "upload-progress--error",
    );
  });

  it("uses correct ARIA attributes for accessibility", () => {
    render(<UploadProgress percent={75} label="my-upload" />);
    const bar = screen.getByRole("progressbar", { name: "my-upload" });
    expect(bar).toHaveAttribute("aria-valuenow", "75");
    expect(bar).toHaveAttribute("aria-valuemin", "0");
    expect(bar).toHaveAttribute("aria-valuemax", "100");
  });
});
