/**
 * Tests for the API client (mocked axios + XMLHttpRequest).
 */
import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  vi,
  type Mock,
} from "vitest";
import {
  getMetadata,
  getPresignedUrl,
  uploadToS3,
  type UploadProgressEvent,
} from "./client";

interface FakeXhr {
  open: Mock;
  send: Mock;
  setRequestHeader: Mock;
  abort: Mock;
  upload: { onprogress: ((event: ProgressEvent) => void) | null };
  onload: (() => void) | null;
  onerror: (() => void) | null;
  onabort: (() => void) | null;
  status: number;
  statusText: string;
  responseText: string;
}

function makeFakeXhr(): FakeXhr {
  return {
    open: vi.fn(),
    send: vi.fn(),
    setRequestHeader: vi.fn(),
    abort: vi.fn(),
    upload: { onprogress: null },
    onload: null,
    onerror: null,
    onabort: null,
    status: 0,
    statusText: "",
    responseText: "",
  };
}

describe("uploadToS3", () => {
  let xhr: FakeXhr;

  beforeEach(() => {
    xhr = makeFakeXhr();
    vi.stubGlobal(
      "XMLHttpRequest",
      vi.fn(() => xhr) as unknown as typeof XMLHttpRequest,
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("uses PUT with correct content-type and reports progress", async () => {
    const file = new File(["abc"], "swing.mp4", { type: "video/mp4" });
    Object.defineProperty(file, "size", { value: 1000 });

    const events: UploadProgressEvent[] = [];
    const promise = uploadToS3("https://s3.example.com/upload", file, {
      onProgress: (e) => events.push(e),
    });

    expect(xhr.open).toHaveBeenCalledWith(
      "PUT",
      "https://s3.example.com/upload",
      true,
    );
    expect(xhr.setRequestHeader).toHaveBeenCalledWith(
      "Content-Type",
      "video/mp4",
    );

    // Simulate progress
    xhr.upload.onprogress?.({
      lengthComputable: true,
      loaded: 500,
      total: 1000,
    } as ProgressEvent);

    // Simulate success
    xhr.status = 200;
    xhr.onload?.();

    await expect(promise).resolves.toBeUndefined();

    expect(events.length).toBeGreaterThanOrEqual(2);
    expect(events[0].percent).toBe(50);
    // Final progress event should clamp to 100.
    expect(events[events.length - 1].percent).toBe(100);
  });

  it("rejects when the response is non-2xx", async () => {
    const file = new File(["x"], "f.mp4", { type: "video/mp4" });
    const promise = uploadToS3("https://s3.example.com/upload", file);

    xhr.status = 500;
    xhr.statusText = "Server Error";
    xhr.responseText = "boom";
    xhr.onload?.();

    await expect(promise).rejects.toThrow(/Upload failed with status 500/);
  });

  it("rejects on network errors", async () => {
    const file = new File(["x"], "f.mp4", { type: "video/mp4" });
    const promise = uploadToS3("https://s3.example.com/upload", file);
    xhr.onerror?.();
    await expect(promise).rejects.toThrow(/Network error/);
  });
});

describe("getPresignedUrl", () => {
  it("posts to /upload/presigned-url and returns the parsed body", async () => {
    const fakeAxios = {
      post: vi.fn().mockResolvedValue({
        data: {
          upload_url: "https://s3/upload",
          file_key: "videos/abc.mp4",
          expires_in: 3600,
        },
      }),
    };

    const result = await getPresignedUrl(
      { file_name: "swing.mp4", content_type: "video/mp4" },
      // The signature accepts AxiosInstance; using a structural mock is ok for tests.
      fakeAxios as unknown as Parameters<typeof getPresignedUrl>[1],
    );

    expect(fakeAxios.post).toHaveBeenCalledWith("/upload/presigned-url", {
      file_name: "swing.mp4",
      content_type: "video/mp4",
    });
    expect(result.file_key).toBe("videos/abc.mp4");
  });
});

describe("getMetadata", () => {
  it("URL-encodes the file_key path segments and posts to /videos/{file_key}/metadata", async () => {
    const fakeAxios = {
      post: vi.fn().mockResolvedValue({
        data: {
          file_name: "swing.mp4",
          duration_seconds: 5,
          resolution: { width: 1280, height: 720 },
          file_size_bytes: 1024,
          thumbnail_url: null,
        },
      }),
    };

    await getMetadata(
      "videos/abc def.mp4",
      fakeAxios as unknown as Parameters<typeof getMetadata>[1],
    );

    expect(fakeAxios.post).toHaveBeenCalledWith(
      "/videos/videos/abc%20def.mp4/metadata",
    );
  });
});
