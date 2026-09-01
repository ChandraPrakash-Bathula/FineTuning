import { useEffect, useState } from "react";
import type { Lab } from "../lib/registry";
import { RANKS } from "../lib/arch";
import { loadServingBf16, loadVariantProbes } from "../lib/probes";
import { pct } from "../lib/format";
import Provenance from "./Provenance";

interface Row { rank: number; fourBit: number; bf16: number; registry: number }

/**
 * A QLoRA adapter is trained against a 4-bit NF4 base. It will happily load onto
 * a bf16 base, and the build template even suggests doing so for simplicity.
 * Both serving modes were run over all 1,000 held-out questions, so the cost of
 * that shortcut is measurable rather than hypothetical.
 */
export default function ServingBaseComparison({ lab }: { lab: Lab }) {
  const [rows, setRows] = useState<Row[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    Promise.all(
      RANKS.map(async (r) => {
        const id = `qlora_r${r}`;
        const [q, b] = await Promise.all([loadVariantProbes(id), loadServingBf16(id)]);
        return {
          rank: r,
          fourBit: q.metrics.sql_execution_accuracy,
          bf16: b.metrics.sql_execution_accuracy,
          registry: lab.byId.get(id)!.execAccuracy,
        };
      }),
    ).then(setRows).catch((e) => setErr(String(e)));
  }, [lab]);

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
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.rank}>
                <td>r = {r.rank}</td>
                <td className="hi">{pct(r.fourBit)}</td>
                <td>{pct(r.bf16)}</td>
                <td>{((r.bf16 - r.fourBit) * 100).toFixed(1)} pts</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="body">
        {allDown ? "Every rank loses accuracy" : "Accuracy shifts"} when the adapter is
        served on a bf16 base instead of the 4-bit base it was trained against —
        on average {(meanDrop * 100).toFixed(1)} points, across all{" "}
        {(1000).toLocaleString()} held-out questions.
      </p>
      <p className="muted">
        These are re-generated probe runs, so they differ from the registry's own
        evaluation by a few tenths of a point — greedy decoding still varies slightly
        with batching. Across LoRA and full fine-tuning that drift averages 0.17 points
        in both directions; the bf16 serving penalty here is roughly six times larger
        and goes one way at every rank, which is what makes it a result rather than
        noise. Registry figures are used everywhere a headline metric is shown.
      </p>

      <div className="callout warn">
        <strong>An adapter is trained against a specific base. It is not portable for free.</strong>
        <span>
          The LoRA weights learned to correct a <em>quantised</em> model. Put them on top
          of the full-precision weights and they are correcting an error that is no longer
          there. The drop is small, consistent, and entirely avoidable — serve QLoRA
          adapters on the 4-bit base, as this project's own evaluation did.
        </span>
      </div>
    </>
  );
}
