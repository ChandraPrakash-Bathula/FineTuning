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
 * Where the API lives. Resolved at RUNTIME, not build time, so a static
 * deployment (which is only files on a CDN and can never run a model itself)
 * can still be pointed at a backend the reader spins up elsewhere -- a Colab
 * T4 behind a tunnel, a rented GPU, anything.
 *
 *   1. whatever the reader saved in this browser
 *   2. VITE_API_BASE baked in at build time
 *   3. same origin (the Docker Space, where FastAPI serves both)
 */
const STORAGE_KEY = "ftlab.apiBase";

export function getApiBase(): string {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) return saved.replace(/\/$/, "");
  } catch {
    /* private mode or blocked storage: fall through */
  }
  return (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, "") ?? "";
}

export function setApiBase(url: string | null) {
  try {
    if (url) localStorage.setItem(STORAGE_KEY, url.trim().replace(/\/$/, ""));
    else localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* nothing we can do; the caller still gets the in-memory value */
  }
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

export interface LiveResult {
  variant_id: string;
  predicted_sql: string | null;
  gold_sql: string | null;
  result_rows: unknown[][] | null;
  gold_rows: unknown[][] | null;
  exact_match: boolean;
  execution_correct: boolean;
  invalid_sql: boolean;
  error: string | null;
}

export interface RunResponse {
  question: string;
  /** false for a custom question: there is no gold SQL, so there is no score */
  scoreable: boolean;
  results: LiveResult[];
}

export async function runLive(body: {
  variants: string[];
  example_id?: number;
  question?: string;
  table?: { header: string[]; types?: string[]; rows: unknown[][] };
}): Promise<RunResponse> {
  const res = await fetch(`${getApiBase()}/api/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`/api/run: HTTP ${res.status} ${await res.text()}`);
  return (await res.json()) as RunResponse;
}
