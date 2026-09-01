import { useEffect, useMemo, useState } from "react";
import type { Variant } from "../lib/types";
import {
  groupAnswers, loadDisagreement, loadQuestions, loadVariantProbes,
  type Disagreement, type Question, type VariantProbes,
} from "../lib/probes";
import { variantColor } from "../lib/palette";
import { parseSchema } from "../lib/format";
import { columnDelta, equivalenceReason } from "../lib/sqldiff";
import SqlText from "./SqlText";
import Provenance from "./Provenance";
import { type BackendStatus } from "../lib/backend";

const KIND_COPY = {
  equivalent: {
    title: "Different SQL, same answer",
    blurb: "Every variant wrote a different query and every one returns the gold rows. Differing SQL does not imply differing behaviour.",
  },
  split: {
    title: "Genuinely split",
    blurb: "Some variants get it right and others do not — the cases where the choice of method actually shows up.",
  },
  all_wrong: {
    title: "All eleven miss",
    blurb: "Disagreement with nothing correct. These mark the ceiling of the task, not of any one method.",
  },
} as const;

type Kind = keyof typeof KIND_COPY;

export default function SqlLab({ variants, dark, backend, apiBase, onConnect }: {
  variants: Variant[];
  dark: boolean;
  backend: BackendStatus;
  apiBase: string;
  onConnect: (url: string | null) => Promise<void>;
}) {
  const [questions, setQuestions] = useState<Question[] | null>(null);
  const [dis, setDis] = useState<Disagreement | null>(null);
  const [probes, setProbes] = useState<Record<string, VariantProbes>>({});
  const [loading, setLoading] = useState(false);
  const [qid, setQid] = useState<number | null>(null);
  const [search, setSearch] = useState("");
  const [kind, setKind] = useState<Kind>("split");
  const [err, setErr] = useState<string | null>(null);
  const [apiDraft, setApiDraft] = useState(apiBase);
  const [connecting, setConnecting] = useState(false);

  const connect = async (url: string | null) => {
    setConnecting(true);
    setApiDraft(url ?? "");
    try {
      await onConnect(url);
    } finally {
      setConnecting(false);
    }
  };


  useEffect(() => {
    Promise.all([loadQuestions(), loadDisagreement()])
      .then(([q, d]) => {
        setQuestions(q);
        setDis(d);
        setQid((cur) => cur ?? d.featured.split[0]?.id ?? 0);
      })
      .catch((e) => setErr(String(e)));
  }, []);

  // Fetch only the variants actually selected, once each.
  useEffect(() => {
    const missing = variants.filter((v) => !probes[v.id]);
    if (!missing.length) return;
    setLoading(true);
    Promise.all(missing.map((v) => loadVariantProbes(v.id).then((d) => [v.id, d] as const)))
      .then((pairs) => setProbes((p) => ({ ...p, ...Object.fromEntries(pairs) })))
      .catch((e) => setErr(String(e)))
      .finally(() => setLoading(false));
  }, [variants, probes]);

  const ready = variants.filter((v) => probes[v.id]);
  const groups = useMemo(() => {
    if (qid === null || !ready.length) return [];
    return groupAnswers(qid, ready.map((v) => ({ id: v.id, data: probes[v.id] })));
  }, [qid, ready, probes]);

  const nCorrect = useMemo(
    () => ready.reduce(
      (n, v) => n + (qid !== null && probes[v.id].execution_correct[qid] ? 1 : 0), 0),
    [ready, probes, qid]);

  const results = useMemo(() => {
    if (!questions || !search.trim()) return null;
    const needle = search.toLowerCase();
    return questions.filter((q) => q.question.toLowerCase().includes(needle)).slice(0, 40);
  }, [questions, search]);

  if (err) return <p className="body">Could not load the probe run: {err}</p>;
  if (!questions || !dis) return <p className="muted">Loading recorded probe run…</p>;

  const q = qid === null ? null : questions[qid];
  const featured = dis.featured[kind] ?? [];
  const shownViaTab = qid !== null && featured.some((c) => c.id === qid);

  return (
    <div className="lab">
      <div className="lab-intro">
        <p className="body">
          All {dis.n_questions.toLocaleString()} held-out questions, answered by every
          variant during Phase 1. These are <Provenance source="recorded" /> outputs
          replayed from disk — greedy decoding, so a question always gives the same SQL.
        </p>
        <div className={`backend-status backend-${backend.state}`}>
          <span className="state-dot" aria-hidden />
          {backend.state === "checking" && <span>Checking for an inference backend…</span>}

          {backend.state === "not_configured" && (
            <span>
              <strong>Recorded outputs.</strong> No inference backend is attached, so
              every answer below is replayed from the Phase 1 probe run rather than
              generated now — all {dis.n_questions.toLocaleString()} held-out questions.
              A static page is only files; running a 7B model needs a GPU process
              somewhere. Point this page at one below and live generation switches on.
            </span>
          )}

          {backend.state === "unreachable" && (
            <span>
              <strong>Backend not answering</strong> ({backend.reason}). Recorded answers
              below still work.
            </span>
          )}

          {backend.state === "available" && (
            <span>
              {backend.health.models === "ready" ? (
                <>
                  <strong>Live inference ready</strong> — {backend.health.adapters.length}{" "}
                  adapters resident
                  {backend.health.full_ft_loaded ? " plus full_ft" : ""}, warmed in{" "}
                  {backend.health.models_seconds}s.
                </>
              ) : backend.health.models === "loading" ? (
                <>
                  <strong>Backend warming up</strong> — loading the bases (
                  {backend.health.models_seconds}s elapsed; the first load pulls about
                  16&nbsp;GB). Recorded answers below work now.
                </>
              ) : backend.health.models === "error" ? (
                <>
                  <strong>Backend reachable, models failed to load.</strong>{" "}
                  {backend.health.models_detail}
                </>
              ) : backend.health.models === "no_gpu" ? (
                <>
                  <strong>Backend reachable, no GPU.</strong> Live generation needs CUDA
                  and at least 16&nbsp;GB of VRAM; everything below is replayed from the
                  recorded run.
                </>
              ) : (
                <>Backend reachable — {backend.health.adapters.length} adapters registered.</>
              )}
            </span>
          )}
        </div>

        <details className="api-connect" open={backend.state !== "available"}>
          <summary>Connect an inference backend</summary>
          <p className="muted">
            Run <code>backend/app.py</code> on any machine with a CUDA GPU — a free Colab
            T4 works with <code>LOAD_BASES=qlora</code> — expose it (cloudflared, ngrok),
            and paste the URL. It is stored in this browser only.
          </p>
          <form
            className="api-form"
            onSubmit={(e) => { e.preventDefault(); connect(apiDraft.trim() || null); }}
          >
            <input
              value={apiDraft}
              onChange={(e) => setApiDraft(e.target.value)}
              placeholder="https://your-tunnel.trycloudflare.com"
              aria-label="inference backend URL"
              spellCheck={false}
            />
            <button className="chip primary" type="submit" disabled={connecting}>
              {connecting ? "connecting…" : "connect"}
            </button>
            {apiBase && (
              <button className="chip" type="button" onClick={() => connect(null)}>
                clear
              </button>
            )}
          </form>
        </details>

        <div className="lab-counts">
          <Count n={dis.counts.unanimous} of={dis.n_questions} label="identical SQL from all 11" />
          <Count n={dis.counts.split} of={dis.n_questions} label="genuinely split" />
          <Count n={dis.counts.equivalent} of={dis.n_questions} label="different SQL, all correct" />
          <Count n={dis.counts.all_wrong} of={dis.n_questions} label="all eleven wrong" />
        </div>
      </div>

      <div className="lab-picker">
        <div className="lab-kinds" role="tablist" aria-label="featured case type">
          {(Object.keys(KIND_COPY) as Kind[]).map((k) => (
            <button key={k} role="tab" aria-selected={kind === k}
                    className={`chip${kind === k ? " on" : ""}`}
                    onClick={() => {
                      setKind(k);
                      // jump to the first case of this kind, so the answers below
                      // always match the tab that is selected
                      const first = dis.featured[k]?.[0]?.id;
                      if (first !== undefined) setQid(first);
                    }}>
              {KIND_COPY[k].title}
            </button>
          ))}
        </div>
        <p className="muted lab-blurb">
          {shownViaTab
            ? KIND_COPY[kind].blurb
            : "Reached by search, not one of the cases above — its category may differ from the tab selected."}
        </p>
        <div className="lab-cases">
          {featured.map((c) => (
            <button key={c.id} className={`lab-case${c.id === qid ? " on" : ""}`}
                    onClick={() => setQid(c.id)}>
              <span className="lab-case-id">Q{c.id}</span>
              <span className="lab-case-q">{questions[c.id].question}</span>
              <span className="lab-case-meta">
                {c.n_distinct} queries · {c.n_correct}/11 correct
              </span>
            </button>
          ))}
        </div>

        <label className="lab-search">
          <span className="muted">or search all {dis.n_questions.toLocaleString()}</span>
          <input value={search} onChange={(e) => setSearch(e.target.value)}
                 placeholder="e.g. how many players…" aria-label="search questions" />
        </label>
        {results && (
          <div className="lab-results">
            {results.length === 0 && <p className="muted">No question contains that text.</p>}
            {results.map((r) => (
              <button key={r.id} className={`lab-result${r.id === qid ? " on" : ""}`}
                      onClick={() => { setQid(r.id); setSearch(""); }}>
                <span className="lab-case-id">Q{r.id}</span> {r.question}
              </button>
            ))}
          </div>
        )}
      </div>

      {!variants.length && (
        <p className="body">Select at least one variant above to see its answers.</p>
      )}

      {q && variants.length > 0 && (
        <div className="lab-question">
          <h3>{q.question}</h3>
          <div className="q-schema">
            {parseSchema(q.schema).map((c) => (
              <span key={c.col} className="col-pill">
                <span className="mono">{c.col}</span>
                <span className="col-name">{c.name}</span>
              </span>
            ))}
          </div>
          <div className="gold">
            <span className="muted">gold</span> <SqlText sql={q.gold_sql} />
          </div>

          {loading && <p className="muted">Loading recorded answers…</p>}

          {ready.length > 0 && (
            <div className="lab-scoreline">
              <span className={nCorrect === 0 ? "bad" : nCorrect === ready.length ? "ok" : ""}>
                <strong>{nCorrect} of {ready.length}</strong> selected{" "}
                {ready.length === 1 ? "model returns" : "models return"} the gold rows
              </span>
              <span className="muted">
                {groups.length === 1
                  ? "all wrote the same query"
                  : `${groups.length} distinct queries`}
              </span>
            </div>
          )}

          <div className="lab-answers">
            {groups.map((g) => (
              <div key={g.sql} className="out-group"
                   style={{ borderLeftColor: g.executionCorrect ? "var(--good)" : "var(--critical)" }}>
                <div className="out-verdict">
                  <span className={`verdict ${g.executionCorrect ? "ok" : "bad"}`}>
                    {g.executionCorrect ? "✓ returns the gold rows" : "✗ wrong rows"}
                  </span>
                  {g.exactMatch && <span className="badge">exact match</span>}
                  {g.invalidSql && <span className="badge bad">invalid SQL</span>}
                  <span className="muted">
                    {g.variantIds.length === 1 ? "1 model" : `${g.variantIds.length} models`}
                  </span>
                </div>
                <SqlText sql={g.sql} gold={q.gold_sql} />
                <ColumnNote sql={g.sql} gold={q.gold_sql} schema={q.schema} />
                <div className="out-who">
                  {g.variantIds.map((id) => {
                    const v = variants.find((x) => x.id === id)!;
                    return (
                      <span key={id} className="who">
                        <span className="swatch"
                              style={{ background: variantColor(v.method, v.rank, dark) }} />
                        {v.label}
                      </span>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>

          {groups.length > 1 && groups.every((g) => g.executionCorrect) && (
            <div className="callout">
              <strong>{groups.length} different queries, all correct.</strong>
              <span>
                {equivalenceReason(groups.map((g) => g.sql)) ??
                  "These queries differ textually but return the same rows."}{" "}
                Execution accuracy scores the rows returned, not the SQL text — which is
                why a query can look nothing like the gold answer and still be right.
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/** Names the columns a wrong query read, using the table's own headers. */
function ColumnNote({ sql, gold, schema }: { sql: string; gold: string; schema: string[] }) {
  const { extra, missing } = columnDelta(sql, gold);
  if (!extra.length && !missing.length) return null;
  const names = new Map(parseSchema(schema).map((c) => [c.col, c.name]));
  const label = (c: string) => `${c}${names.get(c) ? ` (${names.get(c)})` : ""}`;
  return (
    <p className="col-note">
      {missing.length > 0 && <>gold reads <strong>{missing.map(label).join(", ")}</strong></>}
      {missing.length > 0 && extra.length > 0 && " · "}
      {extra.length > 0 && <>this reads <strong>{extra.map(label).join(", ")}</strong></>}
    </p>
  );
}

function Count({ n, of, label }: { n: number; of: number; label: string }) {
  return (
    <div className="lab-count">
      <div className="lab-count-n">{n}</div>
      <div className="lab-count-l">{label}</div>
      <div className="muted">{((n / of) * 100).toFixed(1)}%</div>
    </div>
  );
}
