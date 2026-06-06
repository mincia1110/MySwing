/**
 * Tests for FormField (reusable input component used by UserProfileForm).
 *
 * Validates label/input association, required indicator, error rendering,
 * and select options.
 */
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { FormField } from "./FormField";

describe("FormField", () => {
  it("renders a number input with label and unit", () => {
    render(
      <FormField
        id="height"
        type="number"
        label="키"
        value="180"
        onChange={() => undefined}
        required
        unit="cm"
        hint="100-220 범위"
      />,
    );

    const input = screen.getByTestId("form-field-input-height");
    expect(input).toHaveAttribute("type", "number");
    expect(input).toHaveValue(180);
    expect(screen.getByText("cm")).toBeInTheDocument();
    expect(screen.getByText("키")).toBeInTheDocument();
    expect(screen.getByLabelText("required")).toBeInTheDocument();
    expect(screen.getByTestId("form-field-hint-height")).toHaveTextContent(
      "100-220 범위",
    );
  });

  it("calls onChange with the new string value", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();

    render(
      <FormField
        id="weight"
        type="number"
        label="체중"
        value=""
        onChange={onChange}
      />,
    );

    await user.type(screen.getByTestId("form-field-input-weight"), "70");
    // userEvent.type fires onChange per character.
    expect(onChange).toHaveBeenCalledWith("7");
    expect(onChange).toHaveBeenCalledWith("0");
  });

  it("renders an error message and sets aria-invalid", () => {
    render(
      <FormField
        id="height"
        type="number"
        label="키"
        value="50"
        onChange={() => undefined}
        error="키는 100-220cm 사이여야 합니다."
      />,
    );

    const input = screen.getByTestId("form-field-input-height");
    expect(input).toHaveAttribute("aria-invalid", "true");
    expect(screen.getByTestId("form-field-error-height")).toHaveTextContent(
      "키는 100-220cm 사이여야 합니다.",
    );
    expect(screen.getByRole("alert")).toBeInTheDocument();
  });

  it("renders select options including optional placeholder", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();

    render(
      <FormField
        id="batting_direction"
        type="select"
        label="타격 방향"
        value=""
        onChange={onChange}
        placeholder="선택"
        options={[
          { value: "left", label: "Left" },
          { value: "right", label: "Right" },
        ]}
      />,
    );

    const select = screen.getByTestId(
      "form-field-input-batting_direction",
    ) as HTMLSelectElement;
    expect(select.options).toHaveLength(3); // placeholder + 2 options
    await user.selectOptions(select, "right");
    expect(onChange).toHaveBeenCalledWith("right");
  });
});
