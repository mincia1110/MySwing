/**
 * Type definitions for user profile API.
 *
 * Matches backend Pydantic schemas in app/schemas/user_profile.py.
 *
 * Required fields (Requirement 2.1):
 *  - height (cm, 100-220)
 *  - bat_length (24-36 inches OR 61-91 cm)
 *  - batting_direction ("left" | "right")
 *
 * Optional fields (Requirements 2.3, 2.4):
 *  - weight (kg)
 *  - camera_direction ("front" | "side" | "rear")
 *  - age_group (free-form string)
 *  - level ("professional" | "college" | "high_school" | "recreational")
 *  - bat_weight (oz, 16-36)
 */

export type BattingDirection = "left" | "right";
export type CameraDirection = "front" | "side" | "rear";
export type PlayerLevel =
  | "professional"
  | "college"
  | "high_school"
  | "recreational";

export interface UserProfileCreate {
  // Required (Requirement 2.1)
  height: number;
  bat_length: number;
  batting_direction: BattingDirection;

  // Optional recommended (Requirement 2.3)
  weight?: number | null;
  camera_direction?: CameraDirection | null;

  // Optional (Requirement 2.4)
  age_group?: string | null;
  level?: PlayerLevel | null;
  bat_weight?: number | null;
}

export interface UserProfileResponse {
  id: string;
  user_id: string;
  height: number;
  bat_length: number;
  batting_direction: BattingDirection;
  weight?: number | null;
  camera_direction?: CameraDirection | null;
  age_group?: string | null;
  level?: PlayerLevel | null;
  bat_weight?: number | null;
}

/**
 * Validation range constants - matches backend Requirement 2.5.
 */
export const VALIDATION_RANGES = {
  height: { min: 100, max: 220, unit: "cm" },
  /** Bat length accepts inches (24-36) OR cm (61-91). */
  batLengthInches: { min: 24, max: 36, unit: "in" },
  batLengthCm: { min: 61, max: 91, unit: "cm" },
  batWeight: { min: 16, max: 36, unit: "oz" },
} as const;

/**
 * Validate that a numeric bat_length value falls within either accepted range
 * (inches: 24-36 OR centimeters: 61-91).
 */
export function isValidBatLength(value: number): boolean {
  if (!Number.isFinite(value)) return false;
  return (
    (value >= VALIDATION_RANGES.batLengthInches.min &&
      value <= VALIDATION_RANGES.batLengthInches.max) ||
    (value >= VALIDATION_RANGES.batLengthCm.min &&
      value <= VALIDATION_RANGES.batLengthCm.max)
  );
}
