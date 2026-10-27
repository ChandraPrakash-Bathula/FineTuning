import { useEffect, useState } from "react";
import { RANKS } from "../lib/arch";
import { loadServingBf16, loadVariantProbes } from "../lib/probes";
import { pct } from "../lib/format";
import Provenance from "./Provenance";
import { usePaired, fmtP } from "../lib/paired";

interface Row { rank: number; fourBit: number; bf16: number }

/**
 * A QLoRA adapter is trained against a 4-bit NF4 base. It will happily load onto
 * a bf16 base, and the build template even suggests doing so for simplicity.
 * Both serving modes were run over all 1,000 held-out questions, so the cost of
 * that shortcut is measurable rather than hypothetical.
 */
export default function ServingBaseComparison() {
  const [rows, setRows] = useState<Row[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const paired = usePaired();
  const g = paired?.groups.serving_4bit_vs_bf16;

  useEffect(() => {
    Promise.all(
      RANKS.map(async (r) => {
        const id = `qlora_r${r}`;
        const [q, b] = await Promise.all([loadVariantProbes(id), loadServingBf16(id)]);
        return {
          rank: r,
          fourBit: q.metrics.sql_execution_accuracy,
          bf16: b.metrics.sql_execution_accuracy,
        };
      }),
    ).then(setRows).catch((e) => setErr(String(e)));
  }, []);

  if (err) return <p className="body">Could not load the serving comparison: {err}</p>;
  if (!rows) return <p className="muted">Loading both probe runs…</p>;

  const meanDrop = rows.reduce((s, r) => s + (r.fourBit - r.bf16), 0) / rows.length;
  const allDown = rows.every((r) => r.bf16 < r.fourBit);

  return (
    <>
      <div className="scroll-x">
        <table>
          <thead>
            <tr>
              <th>QLoRA adapter</th>
              <th>Served on 4-bit NF4 <Provenance source="recorded" /></th>
              <th>Served on bf16 <Provenance source="recorded" /></th>
              <th>Difference</th>
              <th>McNemar p</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const s = paired?.serving.find((x) => x.a === `qlora_r${r.rank}`);
              return (
                <tr key={r.rank}>
                  <td>r = {r.rank}</td>
                  <td className="hi">{pct(r.fourBit)}</td>
                  <td>{pct(r.bf16)}</td>
                  <td>{((r.bf16 - r.fourBit) * 100).toFixed(1)} pts</td>
                  <td>{s ? fmtP(s.mcnemar_p) : "…"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="body">
        {allDown ? "Every rank loses accuracy" : "Accuracy shifts"} when the adapter is
        served on a bf16 base instead of the 4-bit base it was trained against: on
        average {(meanDrop * 100).toFixed(1)} points over the same{" "}
        {(1000).toLocaleString()} questions.
      </p>
      <p className="body">
        Is that real? Tested one rank at a time, mostly not: each drop is a point or two,
        and only {paired ? paired.serving.filter((x) => x.mcnemar_p < 0.05).length : "…"} of
        the five reaches p &lt; 0.05. The stronger evidence is that all five go the same
        way. To test that directly, average the five adapters' results per question
        and run a paired sign-flip test:{" "}
        {g ? <strong>{(g.diff * 100).toFixed(1)} points, p = {fmtP(g.p)}, 95% interval{" "}
          {(g.bootstrap_ci95[0] * 100).toFixed(1)} to {(g.bootstrap_ci95[1] * 100).toFixed(1)}</strong> : "…"}.
      </p>
      <p className="muted">
        The 4-bit column is the same probe run behind every headline accuracy in this
        app. Script: <code>scripts/paired_stats.py</code>.
      </p>

      <div className="callout warn">
        <strong>An adapter is trained against a specific base. It is not portable for free.</strong>
        <span>
          The LoRA weights learned to correct a <em>quantised</em> model. Put them on top
          of the full-precision weights and they are correcting an error that is no longer
          there. In this sweep the drop was small, consistent across ranks, and entirely
          avoidable: serve QLoRA adapters on the 4-bit base, as this project's own
          evaluation did.
        </span>
      </div>
    </>
  );
}
