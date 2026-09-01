import { useMemo, useState } from "react";

/** The 16 NF4 levels used by bitsandbytes (QLoRA, Dettmers et al. 2023).
 *  Spaced so that equal probability mass of a standard normal falls in each
 *  bucket -- which is why they crowd near zero, where weights actually live. */
export const NF4_LEVELS = [
  -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
  -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
  0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
  0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
  0.7229568362236023, 1.0,
];

const BLOCK = [0.021, -0.044, 0.008, 0.115, -0.067, 0.031, -0.002, 0.052];

export default function QuantViz() {
  // NOT the absmax cell -- that one normalises to exactly 1.0 and shows zero
// error, which is the opposite of the lesson.
  const [pick, setPick] = useState(0);
  const absmax = useMemo(() => Math.max(...BLOCK.map(Math.abs)), []);
  const w = BLOCK[pick];
  const norm = w / absmax;
  const idx = NF4_LEVELS.reduce(
    (best, lv, i) => (Math.abs(lv - norm) < Math.abs(NF4_LEVELS[best] - norm) ? i : best), 0);
  const level = NF4_LEVELS[idx];
  const restored = level * absmax;
  const err = restored - w;

  return (
    <div className="quant">
      <div className="quant-block">
        <div className="muted">one block of 64 weights (8 shown)</div>
        <div className="quant-cells">
          {BLOCK.map((v, i) => (
            <button key={i} className={`qcell${i === pick ? " on" : ""}`}
                    onClick={() => setPick(i)}>{v.toFixed(3)}</button>
          ))}
        </div>
        <div className="muted">
          absmax of the block = <strong>{absmax.toFixed(3)}</strong> — stored once, in
          full precision, and used to rescale the whole block
        </div>
      </div>

      <ol className="quant-steps">
        <li>
          <span className="qs-k">normalise</span>
          <span className="qs-v mono">{w.toFixed(3)} / {absmax.toFixed(3)} = {norm.toFixed(4)}</span>
        </li>
        <li>
          <span className="qs-k">snap to the nearest of 16 levels</span>
          <span className="qs-v mono">index {idx} → {level.toFixed(4)}</span>
        </li>
        <li>
          <span className="qs-k">store</span>
          <span className="qs-v mono">4 bits, not 16</span>
        </li>
        <li>
          <span className="qs-k">dequantise on use</span>
          <span className="qs-v mono">{level.toFixed(4)} × {absmax.toFixed(3)} = {restored.toFixed(4)}</span>
        </li>
        <li className="qs-err">
          <span className="qs-k">error introduced</span>
          <span className="qs-v mono">{err >= 0 ? "+" : ""}{err.toFixed(4)}</span>
        </li>
      </ol>

      <div className="nf4-axis">
        <div className="muted">the 16 NF4 levels — note how they cluster near zero</div>
        <svg viewBox="0 0 520 46" width="100%">
          <line x1="10" y1="30" x2="510" y2="30" stroke="var(--axis)" strokeWidth="1" />
          {NF4_LEVELS.map((lv, i) => {
            const x = 10 + ((lv + 1) / 2) * 500;
            return (
              <g key={i}>
                <line x1={x} y1={24} x2={x} y2={36} stroke={i === idx ? "var(--series-qlora)" : "var(--axis)"}
                      strokeWidth={i === idx ? 2.5 : 1} />
                {i === idx && (
                  <text x={Math.min(495, Math.max(25, x))} y={17} textAnchor="middle"
                        fontSize="10" fill="var(--series-qlora)">{lv.toFixed(3)}</text>
                )}
              </g>
            );
          })}
          {(() => {
            const x = 10 + ((norm + 1) / 2) * 500;
            return <circle cx={x} cy={30} r="4" fill="var(--text-primary)" />;
          })()}
          <text x="10" y="46" fontSize="9.5" fill="var(--text-muted)">−1</text>
          <text x="510" y="46" fontSize="9.5" textAnchor="end" fill="var(--text-muted)">+1</text>
        </svg>
        <div className="muted">
          black dot = the normalised weight · orange tick = the level it snaps to
        </div>
      </div>
    </div>
  );
}
