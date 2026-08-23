/**
 * Thin client for service/api.py. No framework-specific data-fetching
 * magic here on purpose — this project's backend is a plain REST/SSE API,
 * so a plain fetch wrapper is the right amount of abstraction.
 */

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type HealthResponse = {
  status: string;
  embedding_backend: string;
  embedding_model: string;
  generation_backend: string;
  generation_model: string;
  documents_loaded: number;
};

export type Source = {
  text: string;
  source: string;
};

function authHeaders(): Record<string, string> {
  const key = process.env.NEXT_PUBLIC_API_KEY;
  return key ? { "X-API-Key": key } : {};
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_URL}/health`);
  if (!res.ok) throw new Error(`/health returned ${res.status}`);
  return res.json();
}

export type StreamHandlers = {
  onSources: (sources: Source[]) => void;
  onToken: (token: string) => void;
};

/**
 * Calls POST /query/stream and reads the Server-Sent Events response.
 * The first event is always `[SOURCES] <json>` (service/api.py emits it
 * before generation starts, since retrieval already ran) — this saves a
 * second, separately-billed generation call just to fetch sources, at the
 * cost of `streamAnswer` knowing about that one framing detail of the wire
 * format. Every event after that is either a raw answer token or the
 * terminal `[DONE]` / `[ERROR] <message>` sentinel.
 */
export async function streamAnswer(
  query: string,
  { onSources, onToken }: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_URL}/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ query }),
    signal,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail ?? `Request failed (${res.status})`);
  }
  if (!res.body) throw new Error("Response had no body to stream.");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\n\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const payload = line.slice("data: ".length);
      if (payload === "[DONE]") return;
      if (payload.startsWith("[ERROR]")) {
        throw new Error(payload.slice("[ERROR] ".length));
      }
      if (payload.startsWith("[SOURCES] ")) {
        onSources(JSON.parse(payload.slice("[SOURCES] ".length)));
        continue;
      }
      onToken(payload);
    }
  }
}
