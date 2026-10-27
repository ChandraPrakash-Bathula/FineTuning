import { useState } from "react";
import { BLOCK } from "./QuantViz";

const BITS = [2, 3, 4, 8] as const;

/** Evenly spaced levels on [-1, 1]: the simplest possible quantiser. */
function uniformLevels(bits: number): number[] {
  const k = 2 ** bits;
  return Array.from({ length: k }, (_, i) => -1 + (2 * i) / (k - 1));
}

const nearest = (levels: number[], x: number) =>
  levels.reduce((best, lv) => (Math.abs(lv - x) < Math.abs(best - x) ? lv : best), levels[0]);

/** Quantisation from first principles, before NF4: bits -> levels -> rounding error. */
export default function QuantPrimer() {
  const [bits, setBits] = useState<number>(4);
  const levels = uniformLevels(bits);
  const absmax = Math.max(...BLOCK.map(Math.abs));
  const rows = BLOCK.map((w) => {
    const q = nearest(levels, w / absmax) * absmax;
    return { w, q, err: q - w };
  });
  const meanErr = rows.reduce((s, r) => s + Math.abs(r.err), 0) / rows.length;

  return (
    <section className="card">
      <h2>First: what quantisation means</h2>
      <p className="body">
        Every weight is a number, and every number in memory takes a fixed number of
        bits. In bf16 a weight takes 16 bits, so it can be one of 65,536 values. To
        quantise is to allow far fewer values, called <strong>levels</strong>, and round
        each weight to the nearest one. You then store only which level it is: with 16
        levels, that index fits in 4 bits.
      </p>
      <p className="body">
        One extra step is needed. Weights are small numbers (here, around ±0.1), so each
        block of weights is first divided by its largest absolute value, the{" "}
        <strong>absmax</strong>. That puts every weight between −1 and +1, where the
        levels live. The absmax is stored once per block so the weights can be scaled
        back.
      </p>

      <div className="rank-picker">
        <span className="muted">bits per weight</span>
        {BITS.map((b) => (
          <button key={b} className={`chip${b === bits ? " on" : ""}`}
                  aria-pressed={b === bits} onClick={() => setBits(b)}>{b}</button>
        ))}
        <span className="muted">→ {levels.length} evenly spaced levels</span>
      </div>

      <svg viewBox="0 0 520 34" width="100%" aria-label={`${levels.length} levels between -1 and 1`}>
        <line x1="10" y1="16" x2="510" y2="16" stroke="var(--axis)" />
        {levels.length <= 64 && levels.map((lv, i) => {
          const x = 10 + ((lv + 1) / 2) * 500;
          return <line key={i} x1={x} y1={10} x2={x} y2={22} stroke="var(--series-qlora)" strokeWidth={1.2} />;
        })}
        {levels.length > 64 && (
          <rect x="10" y="10" width="500" height="12" fill="var(--series-qlora)" opacity="0.35" />
        )}
        {BLOCK.map((w, i) => (
          <circle key={i} cx={10 + ((w / absmax + 1) / 2) * 500} cy={16} r="3" fill="var(--text-primary)" />
        ))}
        <text x="10" y="33" fontSize="9.5" fill="var(--text-muted)">−1</text>
        <text x="510" y="33" fontSize="9.5" textAnchor="end" fill="var(--text-muted)">+1</text>
      </svg>
      <p className="muted">
        dots = the 8 sample weights after dividing by absmax ({absmax.toFixed(3)}) ·
        ticks = allowed levels{levels.length > 64 && " (too dense to draw)"}
      </p>

      <div className="scroll-x">
        <table>
          <thead>
            <tr><th>Original weight</th><th>After {bits}-bit round trip</th><th>Error</th></tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                <td className="mono">{r.w.toFixed(4)}</td>
                <td className="mono">{r.q.toFixed(4)}</td>
                <td className="mono">{r.err >= 0 ? "+" : ""}{r.err.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="body">
        Average error at {bits} bits: <strong>{meanErr.toFixed(4)}</strong>. Fewer bits
        means fewer levels, bigger gaps between them, and bigger rounding errors. Try 2
        and 8. QLoRA uses 4 bits, and the next section shows how it places those 16
        levels better than evenly.
      </p>
    </section>
  );
}
