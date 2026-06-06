/**
 * Tests for userProfile API client (getUserProfile, saveUserProfile).
 *
 * Uses an axios instance with a mocked adapter to verify request shape
 * and response handling, including the 404 -> null behavior of getUserProfile.
 */
import axios, { AxiosError, type AxiosInstance } from "axios";
import { describe, expect, it } from "vitest";
import { getUserProfile, saveUserProfile } from "./userProfile";
import type { UserProfileResponse } from "../types/userProfile";

const SAMPLE_PROFILE: UserProfileResponse = {
  id: "11111111-1111-1111-1111-111111111111",
  user_id: "00000000-0000-0000-0000-000000000001",
  height: 180,
  bat_length: 33,
  batting_direction: "right",
  weight: 80,
  camera_direction: "side",
  age_group: "20s",
  level: "recreational",
  bat_weight: 30,
};

interface RequestRecord {
  url?: string;
  method?: string;
  data?: unknown;
}

/**
 * Build an axios instance whose adapter is a configurable function so each
 * test can stub the response/error and inspect the request shape.
 */
function makeMockClient(
  adapter: (config: { url?: string; method?: string; data?: unknown }) =>
    | { status: number; data: unknown }
    | { status: number; data: unknown; throwError: true },
): { client: AxiosInstance; lastRequest: RequestRecord } {
  const lastRequest: RequestRecord = {};
  const client = axios.create({ baseURL: "/api/v1" });
  client.defaults.adapter = async (config) => {
    lastRequest.url = config.url;
    lastRequest.method = config.method;
    lastRequest.data =
      typeof config.data === "string" ? JSON.parse(config.data) : config.data;
    const result = adapter({
      url: config.url,
      method: config.method,
      data: lastRequest.data,
    });
    if ("throwError" in result && result.throwError) {
      const err = new AxiosError(
        `Request failed with status ${result.status}`,
        String(result.status),
        config,
        null,
        {
          status: result.status,
          statusText: "",
          headers: {},
          config,
          data: result.data,
        },
      );
      throw err;
    }
    return {
      status: result.status,
      statusText: "OK",
      headers: {},
      config,
      data: result.data,
    };
  };
  return { client, lastRequest };
}

describe("getUserProfile", () => {
  it("returns the profile on 200", async () => {
    const { client, lastRequest } = makeMockClient(() => ({
      status: 200,
      data: SAMPLE_PROFILE,
    }));

    const result = await getUserProfile(SAMPLE_PROFILE.user_id, client);
    expect(result).toEqual(SAMPLE_PROFILE);
    expect(lastRequest.method).toBe("get");
    expect(lastRequest.url).toBe(
      `/users/${SAMPLE_PROFILE.user_id}/profile`,
    );
  });

  it("returns null on 404", async () => {
    const { client } = makeMockClient(() => ({
      status: 404,
      data: { detail: "Not found" },
      throwError: true,
    }));

    const result = await getUserProfile("missing-user", client);
    expect(result).toBeNull();
  });

  it("re-throws on other server errors", async () => {
    const { client } = makeMockClient(() => ({
      status: 500,
      data: { detail: "boom" },
      throwError: true,
    }));

    await expect(
      getUserProfile(SAMPLE_PROFILE.user_id, client),
    ).rejects.toBeInstanceOf(AxiosError);
  });
});

describe("saveUserProfile", () => {
  it("posts the payload to the correct URL and returns the saved profile", async () => {
    const { client, lastRequest } = makeMockClient(() => ({
      status: 201,
      data: SAMPLE_PROFILE,
    }));

    const payload = {
      height: 180,
      bat_length: 33,
      batting_direction: "right" as const,
    };

    const result = await saveUserProfile(SAMPLE_PROFILE.user_id, payload, client);

    expect(result).toEqual(SAMPLE_PROFILE);
    expect(lastRequest.method).toBe("post");
    expect(lastRequest.url).toBe(
      `/users/${SAMPLE_PROFILE.user_id}/profile`,
    );
    expect(lastRequest.data).toEqual(payload);
  });

  it("propagates validation errors from the server", async () => {
    const { client } = makeMockClient(() => ({
      status: 400,
      data: { detail: "height out of range" },
      throwError: true,
    }));

    await expect(
      saveUserProfile(
        SAMPLE_PROFILE.user_id,
        {
          height: 999,
          bat_length: 33,
          batting_direction: "right",
        },
        client,
      ),
    ).rejects.toBeInstanceOf(AxiosError);
  });
});
