import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { fetchHealth, streamAnswer, API_URL } from "./api";

function sseResponse(chunks: string[], { ok = true, status = 200 } = {}) {
  const encoder = new TextEncoder();
  let i = 0;
  const stream = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (i < chunks.length) {
        controller.enqueue(encoder.encode(chunks[i]));
        i += 1;
      } else {
        controller.close();
      }
    },
  });
  return { ok, status, body: stream, json: async () => ({}) } as unknown as Response;
}

describe("API_URL", () => {
  it("defaults to localhost:8000 when NEXT_PUBLIC_API_URL is unset", () => {
    expect(API_URL).toBe("http://localhost:8000");
  });
});

describe("fetchHealth", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("returns the parsed JSON body on success", async () => {
    const body = { status: "ok", documents_loaded: 4 };
    (fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: true,
      json: async () => body,
    });
    const result = await fetchHealth();
    expect(result).toEqual(body);
    expect(fetch).toHaveBeenCalledWith(`${API_URL}/health`);
  });

  it("throws when the response is not ok", async () => {
    (fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: false,
      status: 503,
      json: async () => ({}),
    });
    await expect(fetchHealth()).rejects.toThrow("/health returned 503");
  });
});

describe("streamAnswer", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("parses the [SOURCES] preamble then tokens then [DONE]", async () => {
    const sourcesJson = JSON.stringify([{ text: "chunk one", source: "a.txt" }]);
    (fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      sseResponse([`data: [SOURCES] ${sourcesJson}\n\n`, "data: Hel\n\n", "data: lo\n\n", "data: [DONE]\n\n"]),
    );

    const onSources = vi.fn();
    const onToken = vi.fn();
    await streamAnswer("hello", { onSources, onToken });

    expect(onSources).toHaveBeenCalledWith([{ text: "chunk one", source: "a.txt" }]);
    expect(onToken.mock.calls.map((c) => c[0])).toEqual(["Hel", "lo"]);
  });

  it("stops at [DONE] even if more data follows in the same chunk", async () => {
    (fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      sseResponse(["data: [SOURCES] []\n\n", "data: ok\n\n", "data: [DONE]\n\n"]),
    );
    const onToken = vi.fn();
    await streamAnswer("q", { onSources: vi.fn(), onToken });
    expect(onToken).toHaveBeenCalledTimes(1);
  });

  it("throws the [ERROR] message when the stream emits one", async () => {
    (fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      sseResponse(["data: [SOURCES] []\n\n", "data: [ERROR] Ollama is not reachable\n\n"]),
    );
    await expect(streamAnswer("q", { onSources: vi.fn(), onToken: vi.fn() })).rejects.toThrow(
      "Ollama is not reachable",
    );
  });

  it("splits SSE events across chunk boundaries correctly", async () => {
    // The "data: Hello\n\n" event arrives split across two underlying stream reads.
    (fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(
      sseResponse(["data: [SOURCES] []\n\n", "data: Hel", "lo\n\n", "data: [DONE]\n\n"]),
    );
    const onToken = vi.fn();
    await streamAnswer("q", { onSources: vi.fn(), onToken });
    expect(onToken).toHaveBeenCalledWith("Hello");
  });

  it("throws with the backend's detail message on a non-ok response", async () => {
    (fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: false,
      status: 429,
      json: async () => ({ detail: "Rate limit exceeded" }),
    });
    await expect(streamAnswer("q", { onSources: vi.fn(), onToken: vi.fn() })).rejects.toThrow(
      "Rate limit exceeded",
    );
  });

  it("falls back to the response's statusText when the error body isn't JSON", async () => {
    (fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      ok: false,
      status: 500,
      statusText: "Internal Server Error",
      json: async () => {
        throw new Error("not json");
      },
    });
    await expect(streamAnswer("q", { onSources: vi.fn(), onToken: vi.fn() })).rejects.toThrow(
      "Internal Server Error",
    );
  });

  it("throws when the response has no body", async () => {
    (fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({ ok: true, body: null });
    await expect(streamAnswer("q", { onSources: vi.fn(), onToken: vi.fn() })).rejects.toThrow(
      "Response had no body to stream.",
    );
  });

  it("includes X-API-Key when NEXT_PUBLIC_API_KEY is set", async () => {
    vi.stubEnv("NEXT_PUBLIC_API_KEY", "secret123");
    (fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(sseResponse(["data: [DONE]\n\n"]));
    await streamAnswer("q", { onSources: vi.fn(), onToken: vi.fn() });
    const [, init] = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(init.headers["X-API-Key"]).toBe("secret123");
  });

  it("omits X-API-Key when NEXT_PUBLIC_API_KEY is unset", async () => {
    vi.stubEnv("NEXT_PUBLIC_API_KEY", "");
    (fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(sseResponse(["data: [DONE]\n\n"]));
    await streamAnswer("q", { onSources: vi.fn(), onToken: vi.fn() });
    const [, init] = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(init.headers["X-API-Key"]).toBeUndefined();
  });

  it("sends the query in the request body and passes the abort signal through", async () => {
    (fetch as unknown as ReturnType<typeof vi.fn>).mockResolvedValue(sseResponse(["data: [DONE]\n\n"]));
    const controller = new AbortController();
    await streamAnswer("what is X?", { onSources: vi.fn(), onToken: vi.fn() }, controller.signal);

    const [url, init] = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(url).toBe(`${API_URL}/query/stream`);
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ query: "what is X?" });
    expect(init.signal).toBe(controller.signal);
  });
});
