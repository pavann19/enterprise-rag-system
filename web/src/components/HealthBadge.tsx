"use client";

import { useEffect, useState } from "react";
import { fetchHealth, type HealthResponse } from "@/lib/api";

type Status = "checking" | "online" | "offline";

/**
 * A quiet status indicator, not a dashboard. Apple's HIG "Responsibility"
 * principle asks for transparency about what a product does — this is the
 * minimum honest version of that: is the backend actually reachable right
 * now, shown as a single dot + word, not a paragraph of diagnostics.
 */
export function HealthBadge() {
  const [status, setStatus] = useState<Status>("checking");
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchHealth()
      .then((h) => {
        if (cancelled) return;
        setHealth(h);
        setStatus("online");
      })
      .catch(() => {
        if (cancelled) return;
        setStatus("offline");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const dotColor =
    status === "online" ? "bg-emerald-500" : status === "offline" ? "bg-red-500" : "bg-ink-tertiary";

  const label =
    status === "checking"
      ? "Checking backend…"
      : status === "offline"
        ? "Backend unreachable"
        : `${health?.documents_loaded ?? 0} document${health?.documents_loaded === 1 ? "" : "s"} indexed`;

  return (
    <div className="flex items-center gap-2 text-[13px] text-ink-secondary">
      <span className={`h-1.5 w-1.5 rounded-full ${dotColor}`} aria-hidden />
      <span>{label}</span>
    </div>
  );
}
