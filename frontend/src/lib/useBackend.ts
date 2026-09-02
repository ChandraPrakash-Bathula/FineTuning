import { useCallback, useEffect, useState } from "react";
import { checkHealth, type BackendStatus } from "./backend";

/**
 * Reports whether an inference backend is reachable (same-origin on a Docker
 * Space, or VITE_API_BASE baked in at build time) so the SQL lab can say so.
 *
 * While the bases are loading the health endpoint is re-polled, because that
 * transition can take minutes and the user should see it finish.
 */
export function useBackend() {
  const [backend, setBackend] = useState<BackendStatus>({ state: "checking" });

  const probe = useCallback(async (timeoutMs = 6000) => {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), timeoutMs);
    try {
      setBackend(await checkHealth(ctl.signal));
    } finally {
      clearTimeout(t);
    }
  }, []);

  useEffect(() => { void probe(2500); }, [probe]);

  // Keep polling while the models warm up.
  useEffect(() => {
    if (backend.state !== "available" || backend.health.models !== "loading") return;
    const id = setInterval(() => void probe(), 5000);
    return () => clearInterval(id);
  }, [backend, probe]);

  return { backend, refresh: probe };
}
