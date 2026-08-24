import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { HealthBadge } from "./HealthBadge";
import * as api from "@/lib/api";

describe("HealthBadge", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("shows a checking state before the health check resolves", () => {
    vi.spyOn(api, "fetchHealth").mockReturnValue(new Promise(() => {}));
    render(<HealthBadge />);
    expect(screen.getByText(/checking backend/i)).toBeInTheDocument();
  });

  it("shows the document count once the backend responds", async () => {
    vi.spyOn(api, "fetchHealth").mockResolvedValue({
      status: "ok",
      embedding_backend: "local",
      embedding_model: "all-MiniLM-L6-v2",
      generation_backend: "groq",
      generation_model: "openai/gpt-oss-20b",
      documents_loaded: 4,
    });
    render(<HealthBadge />);
    await waitFor(() => expect(screen.getByText(/4 documents indexed/i)).toBeInTheDocument());
  });

  it("uses singular 'document' when exactly one is loaded", async () => {
    vi.spyOn(api, "fetchHealth").mockResolvedValue({
      status: "ok",
      embedding_backend: "local",
      embedding_model: "m",
      generation_backend: "groq",
      generation_model: "m",
      documents_loaded: 1,
    });
    render(<HealthBadge />);
    await waitFor(() => expect(screen.getByText(/1 document indexed/i)).toBeInTheDocument());
    expect(screen.queryByText(/1 documents indexed/i)).not.toBeInTheDocument();
  });

  it("shows an unreachable state when the health check fails", async () => {
    vi.spyOn(api, "fetchHealth").mockRejectedValue(new Error("network error"));
    render(<HealthBadge />);
    await waitFor(() => expect(screen.getByText(/backend unreachable/i)).toBeInTheDocument());
  });

  it("ignores a resolved health check after unmount (no state-update warning)", async () => {
    let resolveHealth: (value: api.HealthResponse) => void = () => {};
    vi.spyOn(api, "fetchHealth").mockReturnValue(
      new Promise((resolve) => {
        resolveHealth = resolve;
      }),
    );
    const { unmount } = render(<HealthBadge />);
    unmount();
    resolveHealth({
      status: "ok",
      embedding_backend: "local",
      embedding_model: "m",
      generation_backend: "groq",
      generation_model: "m",
      documents_loaded: 2,
    });
    await new Promise((r) => setTimeout(r, 0));
  });
});
