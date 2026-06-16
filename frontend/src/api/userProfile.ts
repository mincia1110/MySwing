/**
 * API client methods for user profile endpoints (Requirements 2.1-2.8).
 *
 * Backend endpoints:
 *  - GET  /api/v1/me/profile  -> 200 with profile or 404
 *  - POST /api/v1/me/profile  -> 201 created or 200 updated
 */

import type { AxiosInstance } from "axios";
import { apiClient as defaultClient } from "./client";
import type {
  UserProfileCreate,
  UserProfileResponse,
} from "../types/userProfile";

/**
 * Fetch the saved profile for a user. Returns null if no profile exists (404).
 *
 * Other errors (network failure, 500, etc.) are re-thrown so callers can
 * distinguish missing-profile from request failure.
 */
export async function getUserProfile(
  client: AxiosInstance = defaultClient,
): Promise<UserProfileResponse | null> {
  try {
    const response = await client.get<UserProfileResponse>("/me/profile");
    return response.data;
  } catch (err: unknown) {
    if (isAxiosNotFoundError(err)) {
      return null;
    }
    throw err;
  }
}

/**
 * Create or update the user profile.
 *
 * Returns the saved profile on success. On validation failure (400/422)
 * the underlying axios error is re-thrown so the caller can surface the
 * detail message to the user.
 */
export async function saveUserProfile(
  data: UserProfileCreate,
  client: AxiosInstance = defaultClient,
): Promise<UserProfileResponse> {
  const response = await client.post<UserProfileResponse>("/me/profile", data);
  return response.data;
}

/**
 * Type guard for axios 404 errors. Avoids importing axios just for the
 * `isAxiosError` helper (which is not always reliable across bundlers).
 */
function isAxiosNotFoundError(err: unknown): boolean {
  if (typeof err !== "object" || err === null) return false;
  const maybe = err as { response?: { status?: number }; isAxiosError?: boolean };
  return maybe.response?.status === 404;
}
