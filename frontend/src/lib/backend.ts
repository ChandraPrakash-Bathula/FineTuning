/**
 * The inference backend is optional. Everything on this site except live
 * generation comes from recorded artifacts, so the app must stay fully usable
 * when no GPU process is running.
 */
export interface Health {
  ok: boolean;
  adapters: string[];
  full_ft_loaded: boolean;
  load_bases?: string;
  /** cold | loading | ready | error | no_gpu — the bases warm up at startup */
  models?: "cold" | "loading" | "ready" | "error" | "no_gpu";
  models_detail?: string;
  models_seconds?: number;
}

export type BackendStatus =
  | { state: "checking" }
  | { state: "available"; health: Health }
  /** No API mounted at all -- the static build. Expected, not a fault. */
  | { state: "not_configured" }
  /** An API is supposed to be there and is not answering. */
  | { state: "unreachable"; reason: string };

/**
 * Where the API lives. Resolved at RUNTIME, not build time:
 *
 *   1. VITE_API_BASE baked in at build time
 *   2. same origin (the Docker Space, where FastAPI serves both)
 */
export function getApiBase(): string {
  return (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, "") ?? "";
}

export async function checkHealth(signal?: AbortSignal): Promise<BackendStatus> {
  try {
    const res = await fetch(`${getApiBase()}/api/health`, { signal });
    // A static deployment has no /api route, so the host answers 404 (or serves
    // the SPA shell). Neither means anything is broken.
    if (res.status === 404) return { state: "not_configured" };
    if (!res.ok) return { state: "unreachable", reason: `HTTP ${res.status}` };
    const body = await res.text();
    try {
      return { state: "available", health: JSON.parse(body) as Health };
    } catch {
      return { state: "not_configured" };   // got HTML, not the API
    }
  } catch (e) {
    return {
      state: "unreachable",
      reason: e instanceof Error && e.name === "AbortError" ? "timed out" : "not reachable",
    };
  }
}
