import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import Home from "./page";
import * as api from "@/lib/api";

// HealthBadge fires its own fetchHealth() on mount in every test here —
// stub it to "checking" (a pending promise) so it never resolves/rejects
// and interferes with assertions about the query flow, which is what
// these tests actually exercise.
function stubHealthPending() {
  vi.spyOn(api, "fetchHealth").mockReturnValue(new Promise(() => {}));
}

describe("Home page — query flow", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    stubHealthPending();
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("disables Ask while the query input is empty", () => {
    render(<Home />);
    expect(screen.getByRole("button", { name: "Ask" })).toBeDisabled();
  });

  it("enables Ask once text is typed", async () => {
    const user = userEvent.setup();
    render(<Home />);
    await user.type(screen.getByPlaceholderText(/approval threshold/i), "hello");
    expect(screen.getByRole("button", { name: "Ask" })).toBeEnabled();
  });

  it("streams tokens into the answer as they arrive", async () => {
    vi.spyOn(api, "streamAnswer").mockImplementation(async (_q, handlers) => {
      handlers.onSources([{ text: "passage", source: "a.txt" }]);
      handlers.onToken("Hello");
      handlers.onToken(" world");
    });

    const user = userEvent.setup();
    render(<Home />);
    await user.type(screen.getByPlaceholderText(/approval threshold/i), "hi");
    await user.click(screen.getByRole("button", { name: "Ask" }));

    await waitFor(() => expect(screen.getByText("Hello world")).toBeInTheDocument());
    expect(screen.getByText("a.txt")).toBeInTheDocument();
    expect(screen.getByText("passage")).toBeInTheDocument();
  });

  it("shows the Asking… label and disables the input while streaming", async () => {
    let resolveStream: () => void = () => {};
    vi.spyOn(api, "streamAnswer").mockReturnValue(
      new Promise((resolve) => {
        resolveStream = () => resolve();
      }),
    );

    const user = userEvent.setup();
    render(<Home />);
    const input = screen.getByPlaceholderText(/approval threshold/i);
    await user.type(input, "hi");
    await user.click(screen.getByRole("button", { name: "Ask" }));

    expect(screen.getByRole("button", { name: "Asking…" })).toBeInTheDocument();
    expect(input).toBeDisabled();

    resolveStream();
    await waitFor(() => expect(screen.getByRole("button", { name: "Ask" })).toBeInTheDocument());
  });

  it("shows the error message and hides the answer card when streaming fails", async () => {
    vi.spyOn(api, "streamAnswer").mockRejectedValue(new Error("Ollama is not reachable"));

    const user = userEvent.setup();
    render(<Home />);
    await user.type(screen.getByPlaceholderText(/approval threshold/i), "hi");
    await user.click(screen.getByRole("button", { name: "Ask" }));

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("Ollama is not reachable"));
    expect(screen.queryByText("Answer")).not.toBeInTheDocument();
  });

  it("falls back to a generic error message for a non-Error rejection", async () => {
    vi.spyOn(api, "streamAnswer").mockRejectedValue("a plain string, not an Error");

    const user = userEvent.setup();
    render(<Home />);
    await user.type(screen.getByPlaceholderText(/approval threshold/i), "hi");
    await user.click(screen.getByRole("button", { name: "Ask" }));

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("Something went wrong."));
  });

  it("does not call streamAnswer for a whitespace-only query", async () => {
    const spy = vi.spyOn(api, "streamAnswer");
    const user = userEvent.setup();
    render(<Home />);
    const input = screen.getByPlaceholderText(/approval threshold/i);
    await user.type(input, "   ");
    // Button stays disabled for whitespace-only input (query.trim() check),
    // so submit via Enter keypress on the form instead of the button.
    await user.type(input, "{Enter}");
    expect(spy).not.toHaveBeenCalled();
  });

  it("Clear button resets the answer, sources, and error state", async () => {
    vi.spyOn(api, "streamAnswer").mockImplementation(async (_q, handlers) => {
      handlers.onToken("an answer");
    });

    const user = userEvent.setup();
    render(<Home />);
    await user.type(screen.getByPlaceholderText(/approval threshold/i), "hi");
    await user.click(screen.getByRole("button", { name: "Ask" }));
    await waitFor(() => expect(screen.getByText("an answer")).toBeInTheDocument());

    await user.click(screen.getByRole("button", { name: "Clear" }));
    expect(screen.queryByText("an answer")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Clear" })).not.toBeInTheDocument();
  });

  it("does not show the Clear button before any question has been asked", () => {
    render(<Home />);
    expect(screen.queryByRole("button", { name: "Clear" })).not.toBeInTheDocument();
  });

  it("shows a warning when the model returns an empty response", async () => {
    vi.spyOn(api, "streamAnswer").mockResolvedValue(undefined);

    const user = userEvent.setup();
    render(<Home />);
    await user.type(screen.getByPlaceholderText(/approval threshold/i), "hi");
    await user.click(screen.getByRole("button", { name: "Ask" }));

    await waitFor(() => expect(screen.getByText(/empty response/i)).toBeInTheDocument());
  });

  it("shows the waking-up hint after 3s of no tokens, and clears it once one arrives", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    let sendToken: (() => void) | undefined;
    vi.spyOn(api, "streamAnswer").mockImplementation(
      (_q, handlers) =>
        new Promise((resolve) => {
          sendToken = () => {
            handlers.onToken("ok");
            resolve();
          };
        }),
    );

    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<Home />);
    await user.type(screen.getByPlaceholderText(/approval threshold/i), "hi");
    await user.click(screen.getByRole("button", { name: "Ask" }));

    expect(screen.queryByText(/sleeps when idle/i)).not.toBeInTheDocument();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(3000);
    });
    expect(screen.getByText(/sleeps when idle/i)).toBeInTheDocument();

    await act(async () => {
      sendToken?.();
    });
    expect(screen.queryByText(/sleeps when idle/i)).not.toBeInTheDocument();
  });

  it("does not show the waking-up hint once the answer resolves quickly", async () => {
    vi.spyOn(api, "streamAnswer").mockImplementation(async (_q, handlers) => {
      handlers.onToken("fast answer");
    });

    const user = userEvent.setup();
    render(<Home />);
    await user.type(screen.getByPlaceholderText(/approval threshold/i), "hi");
    await user.click(screen.getByRole("button", { name: "Ask" }));

    await waitFor(() => expect(screen.getByText("fast answer")).toBeInTheDocument());
    expect(screen.queryByText(/sleeps when idle/i)).not.toBeInTheDocument();
  });
});
