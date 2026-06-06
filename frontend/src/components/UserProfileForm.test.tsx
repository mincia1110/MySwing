/**
 * Tests for UserProfileForm.
 *
 * Validates:
 *  - Required field validation (Requirements 2.1, 2.2)
 *  - Range validation (Requirements 2.5, 2.6)
 *  - Pre-population from existing profile (Requirement 2.7)
 *  - Successful submit (Requirements 2.1, 2.8)
 */
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";
import { UserProfileForm, validateProfileForm } from "./UserProfileForm";
import * as profileApi from "../api/userProfile";
import type { UserProfileResponse } from "../types/userProfile";

const TEST_USER_ID = "00000000-0000-0000-0000-000000000001";

const SAVED_PROFILE: UserProfileResponse = {
  id: "11111111-1111-1111-1111-111111111111",
  user_id: TEST_USER_ID,
  height: 180,
  bat_length: 33,
  batting_direction: "right",
  weight: 80,
  camera_direction: "side",
  age_group: "20s",
  level: "recreational",
  bat_weight: 30,
};

let getProfileSpy: MockInstance<typeof profileApi.getUserProfile>;
let saveProfileSpy: MockInstance<typeof profileApi.saveUserProfile>;

beforeEach(() => {
  getProfileSpy = vi.spyOn(
    profileApi,
    "getUserProfile",
  ) as MockInstance<typeof profileApi.getUserProfile>;
  saveProfileSpy = vi.spyOn(
    profileApi,
    "saveUserProfile",
  ) as MockInstance<typeof profileApi.saveUserProfile>;
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("validateProfileForm", () => {
  const baseValid = {
    height: "180",
    bat_length: "33",
    batting_direction: "right",
    weight: "",
    camera_direction: "",
    age_group: "",
    level: "",
    bat_weight: "",
  };

  it("returns no errors for a valid required-only payload", () => {
    expect(validateProfileForm(baseValid)).toEqual({});
  });

  it("flags missing required fields", () => {
    const errors = validateProfileForm({
      ...baseValid,
      height: "",
      bat_length: "",
      batting_direction: "",
    });
    expect(errors.height).toBeDefined();
    expect(errors.bat_length).toBeDefined();
    expect(errors.batting_direction).toBeDefined();
  });

  it("flags out-of-range height (below min)", () => {
    const errors = validateProfileForm({ ...baseValid, height: "50" });
    expect(errors.height).toMatch(/100-220/);
  });

  it("flags out-of-range height (above max)", () => {
    const errors = validateProfileForm({ ...baseValid, height: "250" });
    expect(errors.height).toMatch(/100-220/);
  });

  it("accepts bat_length in inches range (24-36)", () => {
    expect(
      validateProfileForm({ ...baseValid, bat_length: "30" }).bat_length,
    ).toBeUndefined();
  });

  it("accepts bat_length in cm range (61-91)", () => {
    expect(
      validateProfileForm({ ...baseValid, bat_length: "85" }).bat_length,
    ).toBeUndefined();
  });

  it("rejects bat_length outside both inch and cm ranges", () => {
    const errors = validateProfileForm({ ...baseValid, bat_length: "50" });
    expect(errors.bat_length).toBeDefined();
  });

  it("flags out-of-range bat_weight (below 16)", () => {
    const errors = validateProfileForm({ ...baseValid, bat_weight: "10" });
    expect(errors.bat_weight).toMatch(/16-36/);
  });

  it("flags out-of-range bat_weight (above 36)", () => {
    const errors = validateProfileForm({ ...baseValid, bat_weight: "40" });
    expect(errors.bat_weight).toMatch(/16-36/);
  });

  it("flags non-positive weight when provided", () => {
    expect(
      validateProfileForm({ ...baseValid, weight: "0" }).weight,
    ).toBeDefined();
    expect(
      validateProfileForm({ ...baseValid, weight: "-5" }).weight,
    ).toBeDefined();
  });

  it("ignores empty optional fields", () => {
    expect(validateProfileForm(baseValid)).toEqual({});
  });
});

describe("UserProfileForm", () => {
  it("shows validation errors for missing required fields on submit", async () => {
    getProfileSpy.mockResolvedValue(null);
    saveProfileSpy.mockResolvedValue(SAVED_PROFILE);
    const user = userEvent.setup();

    render(<UserProfileForm userId={TEST_USER_ID} />);

    // Wait for loading to complete.
    await screen.findByTestId("user-profile-form");

    const submitBtn = screen.getByTestId("user-profile-form-submit");
    await user.click(submitBtn);

    expect(
      await screen.findByTestId("form-field-error-height"),
    ).toHaveTextContent(/필수/);
    expect(screen.getByTestId("form-field-error-bat_length")).toHaveTextContent(
      /필수/,
    );
    expect(
      screen.getByTestId("form-field-error-batting_direction"),
    ).toHaveTextContent(/필수/);

    expect(saveProfileSpy).not.toHaveBeenCalled();
  });

  it("shows out-of-range error messages indicating valid range", async () => {
    getProfileSpy.mockResolvedValue(null);
    saveProfileSpy.mockResolvedValue(SAVED_PROFILE);
    const user = userEvent.setup();

    render(<UserProfileForm userId={TEST_USER_ID} />);
    await screen.findByTestId("user-profile-form");

    // Out-of-range height
    await user.type(screen.getByTestId("form-field-input-height"), "999");
    // Out-of-range bat_length (50 is neither inches nor cm valid)
    await user.type(
      screen.getByTestId("form-field-input-bat_length"),
      "50",
    );
    // Optional but out of range
    await user.type(
      screen.getByTestId("form-field-input-bat_weight"),
      "100",
    );

    // Submit to mark fields touched
    await user.click(screen.getByTestId("user-profile-form-submit"));

    expect(
      await screen.findByTestId("form-field-error-height"),
    ).toHaveTextContent(/100-220cm/);
    expect(screen.getByTestId("form-field-error-bat_length")).toHaveTextContent(
      /24-36/,
    );
    expect(screen.getByTestId("form-field-error-bat_length")).toHaveTextContent(
      /61-91/,
    );
    expect(screen.getByTestId("form-field-error-bat_weight")).toHaveTextContent(
      /16-36oz/,
    );
    expect(saveProfileSpy).not.toHaveBeenCalled();
  });

  it("pre-populates fields from a previously saved profile", async () => {
    getProfileSpy.mockResolvedValue(SAVED_PROFILE);
    saveProfileSpy.mockResolvedValue(SAVED_PROFILE);

    render(<UserProfileForm userId={TEST_USER_ID} />);

    await waitFor(() => {
      expect(screen.getByTestId("form-field-input-height")).toHaveValue(180);
    });

    expect(screen.getByTestId("form-field-input-bat_length")).toHaveValue(33);
    expect(screen.getByTestId("form-field-input-batting_direction")).toHaveValue(
      "right",
    );
    expect(screen.getByTestId("form-field-input-weight")).toHaveValue(80);
    expect(screen.getByTestId("form-field-input-camera_direction")).toHaveValue(
      "side",
    );
    expect(screen.getByTestId("form-field-input-age_group")).toHaveValue("20s");
    expect(screen.getByTestId("form-field-input-level")).toHaveValue(
      "recreational",
    );
    expect(screen.getByTestId("form-field-input-bat_weight")).toHaveValue(30);
  });

  it("submits the form and calls the save API with correct payload", async () => {
    getProfileSpy.mockResolvedValue(null);
    saveProfileSpy.mockResolvedValue(SAVED_PROFILE);
    const onSaved = vi.fn();
    const user = userEvent.setup();

    render(<UserProfileForm userId={TEST_USER_ID} onSaved={onSaved} />);
    await screen.findByTestId("user-profile-form");

    await user.type(screen.getByTestId("form-field-input-height"), "175");
    await user.type(screen.getByTestId("form-field-input-bat_length"), "32");
    await user.selectOptions(
      screen.getByTestId("form-field-input-batting_direction"),
      "left",
    );

    const submitBtn = screen.getByTestId("user-profile-form-submit");
    expect(submitBtn).not.toBeDisabled();

    await user.click(submitBtn);

    await waitFor(() => {
      expect(saveProfileSpy).toHaveBeenCalledTimes(1);
    });

    expect(saveProfileSpy).toHaveBeenCalledWith(TEST_USER_ID, {
      height: 175,
      bat_length: 32,
      batting_direction: "left",
      weight: undefined,
      camera_direction: undefined,
      age_group: undefined,
      level: undefined,
      bat_weight: undefined,
    });

    await screen.findByTestId("user-profile-form-submit-success");
    expect(onSaved).toHaveBeenCalledWith(SAVED_PROFILE);
  });

  it("includes optional fields in the payload when provided", async () => {
    getProfileSpy.mockResolvedValue(null);
    saveProfileSpy.mockResolvedValue(SAVED_PROFILE);
    const user = userEvent.setup();

    render(<UserProfileForm userId={TEST_USER_ID} />);
    await screen.findByTestId("user-profile-form");

    await user.type(screen.getByTestId("form-field-input-height"), "170");
    await user.type(screen.getByTestId("form-field-input-bat_length"), "85");
    await user.selectOptions(
      screen.getByTestId("form-field-input-batting_direction"),
      "right",
    );
    await user.type(screen.getByTestId("form-field-input-weight"), "72.5");
    await user.selectOptions(
      screen.getByTestId("form-field-input-camera_direction"),
      "front",
    );
    await user.type(screen.getByTestId("form-field-input-age_group"), "30대");
    await user.selectOptions(
      screen.getByTestId("form-field-input-level"),
      "high_school",
    );
    await user.type(screen.getByTestId("form-field-input-bat_weight"), "28");

    await user.click(screen.getByTestId("user-profile-form-submit"));

    await waitFor(() => {
      expect(saveProfileSpy).toHaveBeenCalledWith(TEST_USER_ID, {
        height: 170,
        bat_length: 85,
        batting_direction: "right",
        weight: 72.5,
        camera_direction: "front",
        age_group: "30대",
        level: "high_school",
        bat_weight: 28,
      });
    });
  });
});
