import type { Variant } from "../lib/types";
import { variantColor } from "../lib/palette";
import { params, size, duration, pct } from "../lib/format";

/** The table view. Required as relief for the sub-3:1 light-mode series colour,
 *  and the fastest way to read eleven variants at once. */
export default function ComparisonTable({
  variants, dark,
}: { variants: Variant[]; dark: boolean }) {
  if (!variants.length) return null;

  const best = Math.max(...variants.map((v) => v.execAccuracy));
  const leanest = Math.min(...variants.map((v) => v.peakMemoryGb));

  return (
    <div className="card">
      <div className="card-head">
        <h2>Every number, side by side</h2>
        <div className="sub">Straight from <code>registry.json</code> — nothing recomputed here.</div>
      </div>
      <div className="scroll-x">
        <table>
          <thead>
            <tr>
              <th>Variant</th>
              <th>Trainable</th>
              <th>% of base</th>
              <th>Peak memory</th>
              <th>Checkpoint</th>
              <th>Train time</th>
              <th>Exact</th>
              <th>Execution</th>
            </tr>
          </thead>
          <tbody>
            {variants.map((v) => (
              <tr key={v.id}>
                <td>
                  <span className="who">
                    <span className="swatch" style={{ background: variantColor(v.method, v.rank, dark) }} />
                    {v.label}
                  </span>
                </td>
                <td>{params(v.trainable)}</td>
                <td>{v.trainablePct < 1 ? v.trainablePct.toFixed(3) : v.trainablePct.toFixed(0)}%</td>
                <td className={v.peakMemoryGb === leanest ? "hi" : undefined}>
                  {v.peakMemoryGb.toFixed(1)} GB
                </td>
                <td>{size(v.checkpointMb)}</td>
                <td>{duration(v.trainingMinutes)}</td>
                <td>{pct(v.exactMatch)}</td>
                <td className={v.execAccuracy === best ? "hi" : undefined}>{pct(v.execAccuracy)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
