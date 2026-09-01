import { useCallback, useEffect, useState } from "react";
import { checkHealth, getApiBase, setApiBase, type BackendStatus } from "./backend";

/**
 * One backend connection shared by everything that needs it, so the SQL lab and
 * the live panel never disagree about whether a GPU is reachable.
 *
 * While the bases are loading the health endpoint is re-polled, because that
 * transition can take minutes and the user should see it finish.
 */
export function useBackend() {
  const [backend, setBackend] = useState<BackendStatus>({ state: "checking" });
  const [apiBase, setBase] = useState(() => getApiBase());

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

  const connect = useCallback(async (url: string | null) => {
    setApiBase(url);
    setBase(url ?? "");
    setBackend({ state: "checking" });
    await probe();
  }, [probe]);

  return { backend, connect, apiBase, refresh: probe };
}
