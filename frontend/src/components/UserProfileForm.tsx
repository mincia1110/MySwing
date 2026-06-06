/**
 * User profile input form (Requirements 2.1-2.7).
 *
 * Required fields:
 *  - height (cm, 100-220)
 *  - bat_length (24-36 inches OR 61-91 cm)
 *  - batting_direction ("left" | "right")
 *
 * Optional fields:
 *  - weight (kg, > 0)
 *  - camera_direction ("front" | "side" | "rear")
 *  - age_group (free text)
 *  - level ("professional" | "college" | "high_school" | "recreational")
 *  - bat_weight (oz, 16-36)
 *
 * Behavior:
 *  - On mount, calls getUserProfile(userId) and pre-populates fields when
 *    a previously saved profile exists (Requirement 2.7).
 *  - Validates fields in real-time; submit is blocked while errors exist
 *    (Requirements 2.2, 2.6).
 *  - On submit, calls saveUserProfile(userId, data) (Requirement 2.8).
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { getUserProfile, saveUserProfile } from "../api/userProfile";
import {
  isValidBatLength,
  VALIDATION_RANGES,
  type BattingDirection,
  type CameraDirection,
  type PlayerLevel,
  type UserProfileCreate,
  type UserProfileResponse,
} from "../types/userProfile";
import { FormField, type FormFieldOption } from "./FormField";
import "./UserProfileForm.css";

export interface UserProfileFormProps {
  userId: string;
  /** Called after a successful save with the saved profile. */
  onSaved?: (profile: UserProfileResponse) => void;
  /** Called when the initial GET or POST request fails. */
  onError?: (error: Error) => void;
}

interface FormValues {
  height: string;
  bat_length: string;
  batting_direction: string;
  weight: string;
  camera_direction: string;
  age_group: string;
  level: string;
  bat_weight: string;
}

type FormErrors = Partial<Record<keyof FormValues, string>>;

const EMPTY_VALUES: FormValues = {
  height: "",
  bat_length: "",
  batting_direction: "",
  weight: "",
  camera_direction: "",
  age_group: "",
  level: "",
  bat_weight: "",
};

const BATTING_DIRECTION_OPTIONS: FormFieldOption[] = [
  { value: "left", label: "왼손 타자 (Left)" },
  { value: "right", label: "오른손 타자 (Right)" },
];

const CAMERA_DIRECTION_OPTIONS: FormFieldOption[] = [
  { value: "front", label: "정면 (Front)" },
  { value: "side", label: "측면 (Side)" },
  { value: "rear", label: "후면 (Rear)" },
];

const LEVEL_OPTIONS: FormFieldOption[] = [
  { value: "professional", label: "프로 (Professional)" },
  { value: "college", label: "대학 (College)" },
  { value: "high_school", label: "고교 (High school)" },
  { value: "recreational", label: "동호인 (Recreational)" },
];

function profileToFormValues(profile: UserProfileResponse): FormValues {
  return {
    height: profile.height != null ? String(profile.height) : "",
    bat_length: profile.bat_length != null ? String(profile.bat_length) : "",
    batting_direction: profile.batting_direction ?? "",
    weight: profile.weight != null ? String(profile.weight) : "",
    camera_direction: profile.camera_direction ?? "",
    age_group: profile.age_group ?? "",
    level: profile.level ?? "",
    bat_weight: profile.bat_weight != null ? String(profile.bat_weight) : "",
  };
}

/**
 * Validate the form values. Returns an errors object mapping field name
 * to the first error message, or empty when all fields are valid.
 *
 * Validation rules (Requirements 2.1, 2.2, 2.5, 2.6):
 *  - height required, numeric, 100-220 cm
 *  - bat_length required, numeric, 24-36 in OR 61-91 cm
 *  - batting_direction required, "left" | "right"
 *  - weight optional, numeric, > 0 if provided
 *  - bat_weight optional, numeric, 16-36 oz if provided
 */
export function validateProfileForm(values: FormValues): FormErrors {
  const errors: FormErrors = {};

  // Required: height
  const heightTrimmed = values.height.trim();
  if (heightTrimmed === "") {
    errors.height = "키(height)는 필수 입력 항목입니다.";
  } else {
    const height = Number(heightTrimmed);
    if (!Number.isFinite(height)) {
      errors.height = "키는 숫자로 입력해주세요.";
    } else if (
      height < VALIDATION_RANGES.height.min ||
      height > VALIDATION_RANGES.height.max
    ) {
      errors.height = `키는 ${VALIDATION_RANGES.height.min}-${VALIDATION_RANGES.height.max}cm 사이여야 합니다.`;
    }
  }

  // Required: bat_length (accepts inches OR cm range)
  const batLengthTrimmed = values.bat_length.trim();
  if (batLengthTrimmed === "") {
    errors.bat_length = "배트 길이(bat length)는 필수 입력 항목입니다.";
  } else {
    const batLength = Number(batLengthTrimmed);
    if (!Number.isFinite(batLength)) {
      errors.bat_length = "배트 길이는 숫자로 입력해주세요.";
    } else if (!isValidBatLength(batLength)) {
      errors.bat_length =
        `배트 길이는 ${VALIDATION_RANGES.batLengthInches.min}-${VALIDATION_RANGES.batLengthInches.max}인치 또는 ${VALIDATION_RANGES.batLengthCm.min}-${VALIDATION_RANGES.batLengthCm.max}cm 사이여야 합니다.`;
    }
  }

  // Required: batting_direction
  if (values.batting_direction === "") {
    errors.batting_direction = "타격 방향(batting direction)은 필수 입력 항목입니다.";
  } else if (
    values.batting_direction !== "left" &&
    values.batting_direction !== "right"
  ) {
    errors.batting_direction = "타격 방향은 left 또는 right 여야 합니다.";
  }

  // Optional: weight (kg) - only if provided
  if (values.weight.trim() !== "") {
    const weight = Number(values.weight);
    if (!Number.isFinite(weight) || weight <= 0) {
      errors.weight = "체중은 0보다 큰 숫자여야 합니다.";
    }
  }

  // Optional: bat_weight (16-36 oz) - only if provided
  if (values.bat_weight.trim() !== "") {
    const batWeight = Number(values.bat_weight);
    if (!Number.isFinite(batWeight)) {
      errors.bat_weight = "배트 무게는 숫자로 입력해주세요.";
    } else if (
      batWeight < VALIDATION_RANGES.batWeight.min ||
      batWeight > VALIDATION_RANGES.batWeight.max
    ) {
      errors.bat_weight = `배트 무게는 ${VALIDATION_RANGES.batWeight.min}-${VALIDATION_RANGES.batWeight.max}oz 사이여야 합니다.`;
    }
  }

  // camera_direction is constrained by the select options; nothing to add.
  // age_group is free-form.
  // level is constrained by the select options.

  return errors;
}

/**
 * Convert validated form values into the API payload shape.
 * Empty optional fields are sent as `undefined` (omitted).
 */
function valuesToPayload(values: FormValues): UserProfileCreate {
  return {
    height: Number(values.height),
    bat_length: Number(values.bat_length),
    batting_direction: values.batting_direction as BattingDirection,
    weight: values.weight.trim() === "" ? undefined : Number(values.weight),
    camera_direction:
      values.camera_direction === ""
        ? undefined
        : (values.camera_direction as CameraDirection),
    age_group: values.age_group.trim() === "" ? undefined : values.age_group.trim(),
    level: values.level === "" ? undefined : (values.level as PlayerLevel),
    bat_weight:
      values.bat_weight.trim() === "" ? undefined : Number(values.bat_weight),
  };
}

export function UserProfileForm({
  userId,
  onSaved,
  onError,
}: UserProfileFormProps) {
  const [values, setValues] = useState<FormValues>(EMPTY_VALUES);
  const [isLoading, setIsLoading] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitSuccess, setSubmitSuccess] = useState(false);
  const [touched, setTouched] = useState<
    Partial<Record<keyof FormValues, boolean>>
  >({});

  // Pre-populate fields with previously saved profile (Requirement 2.7).
  useEffect(() => {
    let cancelled = false;
    setIsLoading(true);
    getUserProfile(userId)
      .then((profile) => {
        if (cancelled) return;
        if (profile) {
          setValues(profileToFormValues(profile));
        }
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        const error = err instanceof Error ? err : new Error(String(err));
        onError?.(error);
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [userId, onError]);

  // Real-time validation: errors recompute on every change.
  const errors = useMemo(() => validateProfileForm(values), [values]);

  const updateField = useCallback(
    <K extends keyof FormValues>(field: K, value: FormValues[K]) => {
      setValues((prev) => ({ ...prev, [field]: value }));
      setTouched((prev) => ({ ...prev, [field]: true }));
      setSubmitSuccess(false);
    },
    [],
  );

  const handleSubmit = useCallback(
    async (event: React.FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      setSubmitError(null);
      setSubmitSuccess(false);

      // Mark all fields as touched so any pending errors are revealed.
      setTouched({
        height: true,
        bat_length: true,
        batting_direction: true,
        weight: true,
        camera_direction: true,
        age_group: true,
        level: true,
        bat_weight: true,
      });

      const currentErrors = validateProfileForm(values);
      if (Object.keys(currentErrors).length > 0) {
        // Block submission per Requirements 2.2 and 2.6.
        return;
      }

      setIsSubmitting(true);
      try {
        const payload = valuesToPayload(values);
        const saved = await saveUserProfile(userId, payload);
        setSubmitSuccess(true);
        onSaved?.(saved);
      } catch (err: unknown) {
        const error = err instanceof Error ? err : new Error(String(err));
        const detail = extractErrorDetail(err) ?? error.message;
        setSubmitError(detail);
        onError?.(error);
      } finally {
        setIsSubmitting(false);
      }
    },
    [values, userId, onSaved, onError],
  );

  const showError = (field: keyof FormValues): string | null =>
    touched[field] ? errors[field] ?? null : null;

  if (isLoading) {
    return (
      <div className="user-profile-form" data-testid="user-profile-form-loading">
        <p>프로필 불러오는 중...</p>
      </div>
    );
  }

  return (
    <form
      className="user-profile-form"
      onSubmit={handleSubmit}
      noValidate
      data-testid="user-profile-form"
      aria-busy={isSubmitting}
    >
      <h2 className="user-profile-form__title">사용자 프로필</h2>
      <p className="user-profile-form__hint">
        분석 정확도 향상을 위해 신체 정보와 타격 특성을 입력해주세요.
        <span className="user-profile-form__required-hint"> * 표시는 필수 항목</span>
      </p>

      <fieldset className="user-profile-form__section" disabled={isSubmitting}>
        <legend>필수 정보</legend>

        <FormField
          id="height"
          type="number"
          label="키 (height)"
          value={values.height}
          onChange={(value) => updateField("height", value)}
          required
          unit="cm"
          min={VALIDATION_RANGES.height.min}
          max={VALIDATION_RANGES.height.max}
          step={0.1}
          hint={`유효 범위: ${VALIDATION_RANGES.height.min}-${VALIDATION_RANGES.height.max}cm`}
          error={showError("height")}
        />

        <FormField
          id="bat_length"
          type="number"
          label="배트 길이 (bat length)"
          value={values.bat_length}
          onChange={(value) => updateField("bat_length", value)}
          required
          unit="in 또는 cm"
          step={0.1}
          hint={`유효 범위: ${VALIDATION_RANGES.batLengthInches.min}-${VALIDATION_RANGES.batLengthInches.max}인치 또는 ${VALIDATION_RANGES.batLengthCm.min}-${VALIDATION_RANGES.batLengthCm.max}cm`}
          error={showError("bat_length")}
        />

        <FormField
          id="batting_direction"
          type="select"
          label="타격 방향 (batting direction)"
          value={values.batting_direction}
          onChange={(value) => updateField("batting_direction", value)}
          required
          options={BATTING_DIRECTION_OPTIONS}
          placeholder="선택해주세요"
          error={showError("batting_direction")}
        />
      </fieldset>

      <fieldset className="user-profile-form__section" disabled={isSubmitting}>
        <legend>선택 정보</legend>

        <FormField
          id="weight"
          type="number"
          label="체중 (weight)"
          value={values.weight}
          onChange={(value) => updateField("weight", value)}
          unit="kg"
          step={0.1}
          hint="권장: 분석 정확도를 위해 입력해주세요."
          error={showError("weight")}
        />

        <FormField
          id="camera_direction"
          type="select"
          label="촬영 방향 (camera direction)"
          value={values.camera_direction}
          onChange={(value) => updateField("camera_direction", value)}
          options={CAMERA_DIRECTION_OPTIONS}
          placeholder="선택 안 함"
          error={showError("camera_direction")}
        />

        <FormField
          id="age_group"
          type="text"
          label="연령대 (age group)"
          value={values.age_group}
          onChange={(value) => updateField("age_group", value)}
          placeholder="예: 20대, U-19"
          error={showError("age_group")}
        />

        <FormField
          id="level"
          type="select"
          label="수준 (level)"
          value={values.level}
          onChange={(value) => updateField("level", value)}
          options={LEVEL_OPTIONS}
          placeholder="선택 안 함"
          error={showError("level")}
        />

        <FormField
          id="bat_weight"
          type="number"
          label="배트 무게 (bat weight)"
          value={values.bat_weight}
          onChange={(value) => updateField("bat_weight", value)}
          unit="oz"
          min={VALIDATION_RANGES.batWeight.min}
          max={VALIDATION_RANGES.batWeight.max}
          step={0.1}
          hint={`유효 범위: ${VALIDATION_RANGES.batWeight.min}-${VALIDATION_RANGES.batWeight.max}oz`}
          error={showError("bat_weight")}
        />
      </fieldset>

      {submitError ? (
        <p
          className="user-profile-form__submit-error"
          role="alert"
          data-testid="user-profile-form-submit-error"
        >
          저장 실패: {submitError}
        </p>
      ) : null}

      {submitSuccess ? (
        <p
          className="user-profile-form__submit-success"
          role="status"
          data-testid="user-profile-form-submit-success"
        >
          프로필이 저장되었습니다.
        </p>
      ) : null}

      <div className="user-profile-form__actions">
        <button
          type="submit"
          className="user-profile-form__submit"
          disabled={isSubmitting}
          data-testid="user-profile-form-submit"
        >
          {isSubmitting ? "저장 중..." : "프로필 저장"}
        </button>
      </div>
    </form>
  );
}

/**
 * Best-effort extraction of `detail` from an axios error (FastAPI returns
 * { detail: "..." } for both validation and HTTPException).
 */
function extractErrorDetail(err: unknown): string | null {
  if (typeof err !== "object" || err === null) return null;
  const e = err as {
    response?: { data?: { detail?: unknown } };
  };
  const detail = e.response?.data?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail.length > 0) {
    // Pydantic validation: array of { msg: string, loc: ... }
    const first = detail[0] as { msg?: unknown; loc?: unknown };
    if (typeof first.msg === "string") {
      const loc = Array.isArray(first.loc) ? first.loc.join(".") : "";
      return loc ? `${loc}: ${first.msg}` : first.msg;
    }
  }
  return null;
}

export default UserProfileForm;
