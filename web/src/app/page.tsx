"use client";

import { useRef, useState } from "react";
import { streamAnswer, type Source } from "@/lib/api";
import { HealthBadge } from "@/components/HealthBadge";

type Phase = "idle" | "streaming" | "done" | "error";

export default function Home() {
  const [query, setQuery] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [wakingUp, setWakingUp] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  async function ask(question: string) {
    const trimmed = question.trim();
    if (!trimmed || phase === "streaming") return;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setPhase("streaming");
    setAnswer("");
    setSources([]);
    setError(null);
    setWakingUp(false);

    // The API's free-tier host sleeps after inactivity — a cold start can
    // take 15-20s before the first token arrives. Anything slower than a
    // couple of seconds gets an explanation instead of a silent spinner,
    // per the same transparency principle the empty-response warning
    // follows: never leave someone guessing whether it's broken.
    const wakeTimer = setTimeout(() => setWakingUp(true), 3000);

    let receivedToken = false;
    try {
      await streamAnswer(
        trimmed,
        {
          onSources: setSources,
          onToken: (token) => {
            receivedToken = true;
            setAnswer((prev) => prev + token);
          },
        },
        controller.signal,
      );
      if (!receivedToken) setError("The model returned an empty response.");
      setPhase("done");
    } catch (err) {
      if (controller.signal.aborted) return;
      setError(err instanceof Error ? err.message : "Something went wrong.");
      setPhase("error");
    } finally {
      clearTimeout(wakeTimer);
      setWakingUp(false);
    }
  }

  const isBusy = phase === "streaming";

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-[640px] flex-col px-6 pt-20 pb-24 sm:px-8">
      <header className="mb-14 flex items-start justify-between gap-4">
        <div>
          <h1 className="text-[2.1rem] font-semibold tracking-tight text-ink sm:text-[2.4rem]">
            Ask your documents
          </h1>
          <p className="mt-2 max-w-[46ch] text-[15px] leading-relaxed text-ink-secondary">
            Answers are grounded strictly in your own corpus — nothing invented, nothing assumed.
          </p>
        </div>
      </header>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          ask(query);
        }}
        className="field flex items-center gap-2 px-4 py-3"
      >
        <SearchGlyph />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="What is the approval threshold for capital expenditures?"
          className="min-w-0 flex-1 bg-transparent text-[15px] text-ink placeholder:text-ink-tertiary focus:outline-none"
          disabled={isBusy}
          autoFocus
        />
        <button
          type="submit"
          disabled={isBusy || !query.trim()}
          className="btn-primary shrink-0 px-4 py-1.5 text-[14px]"
        >
          {isBusy ? "Asking…" : "Ask"}
        </button>
      </form>

      {wakingUp && (
        <p className="animate-fade-up mt-2 text-[13px] text-ink-tertiary">
          The backend runs on a free tier that sleeps when idle — waking it up, this can take up to 20s.
        </p>
      )}

      <div className="mt-3 flex items-center justify-between">
        <HealthBadge />
        {phase !== "idle" && (
          <button
            onClick={() => {
              setPhase("idle");
              setQuery("");
              setAnswer("");
              setSources([]);
              setError(null);
            }}
            className="text-[13px] text-ink-tertiary transition hover:text-ink-secondary"
          >
            Clear
          </button>
        )}
      </div>

      {error && (
        <p className="animate-fade-up mt-6 text-[14px] leading-relaxed text-red-600" role="alert">
          {error}
        </p>
      )}

      {(answer || isBusy) && !error && (
        <section className="animate-fade-up surface-card mt-10 p-6" aria-live="polite">
          <h2 className="mb-3 text-[13px] font-medium tracking-wide text-ink-tertiary uppercase">
            Answer
          </h2>
          <p className="text-[16px] leading-[1.65] whitespace-pre-wrap text-ink">
            {answer}
            {isBusy && <span className="cursor-blink text-accent">▍</span>}
          </p>
        </section>
      )}

      {sources.length > 0 && (
        <section className="animate-fade-up mt-8">
          <h2 className="mb-3 text-[13px] font-medium tracking-wide text-ink-tertiary uppercase">
            Sources
          </h2>
          <div className="divide-hairline surface-card divide-y overflow-hidden">
            {sources.map((s, i) => (
              <details key={i} open={i === 0} className="group px-5 py-4">
                <summary className="flex cursor-pointer list-none items-center justify-between gap-3">
                  <span className="truncate text-[14px] font-medium text-ink">{s.source}</span>
                  <ChevronGlyph />
                </summary>
                <p className="mt-3 text-[13.5px] leading-relaxed text-ink-secondary">{s.text}</p>
              </details>
            ))}
          </div>
        </section>
      )}
    </main>
  );
}

function SearchGlyph() {
  return (
    <svg
      width="17"
      height="17"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      className="shrink-0 text-ink-tertiary"
      aria-hidden
    >
      <circle cx="11" cy="11" r="7" />
      <path d="m21 21-4.3-4.3" />
    </svg>
  );
}

function ChevronGlyph() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="shrink-0 text-ink-tertiary transition-transform duration-200 group-open:rotate-90"
      aria-hidden
    >
      <path d="m9 18 6-6-6-6" />
    </svg>
  );
}
