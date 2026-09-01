import { useEffect, useRef, useState } from "react";
import { ARCH, alphaFor, loraParams, denseParams, TARGET_MODULES } from "../lib/arch";
import { Matrix, Times, Plus, Arrow, ScaleBar } from "./Matrix";

const Q = TARGET_MODULES[0]; // q_proj, 3584 x 3584 -- the matrix we walk through

const STEPS = [
  {
    title: "One weight matrix",
    body: "Inside the model sit thousands of weight matrices. Here is one: the query projection of a single attention layer. Every cell is a number the model learned during pre-training.",
  },
  {
    title: "Full fine-tuning updates all of it",
    body: "To teach the model something new, the obvious move is to keep training every one of those numbers. Each needs a gradient, and Adam keeps two more values per weight on top.",
  },
  {
    title: "So freeze it instead",
    body: "LoRA's premise: the update a fine-tune applies is highly redundant. It has low intrinsic rank. If that is true, we never need to touch W at all.",
  },
  {
    title: "Add two thin matrices",
    body: "B has as many rows as W and only r columns. A has r rows and as many columns as W. Only these two are trained.",
  },
  {
    title: "Multiply them",
    body: "B × A produces a matrix the exact shape of W. Watch it fill in: every cell of the output is a dot product of one row of B and one column of A.",
  },
  {
    title: "Same shape, far fewer numbers",
    body: "ΔW is as big as W, but it was generated from a tiny number of parameters. That is the whole trick — the update lives on a low-rank surface.",
  },
  {
    title: "Scale, then add",
    body: "The update is scaled by alpha/r before it is added, so changing r does not change how strongly the adapter speaks.",
  },
  {
    title: "Merge, and inference is free",
    body: "Once trained, W + (alpha/r)·BA can be folded back into a single matrix of the original shape. The deployed model has the same architecture and the same latency as before.",
  },
] as const;

export default function LoRAAnimation({ rank }: { rank: number }) {
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [sweep, setSweep] = useState(0);
  const raf = useRef<number>(0);

  // The multiply step animates a sweep; every other step is static.
  useEffect(() => {
    if (step !== 4) { setSweep(step > 4 ? 1 : 0); return; }
    setSweep(0);
    const t0 = performance.now();
    const tick = (t: number) => {
      const p = Math.min(1, (t - t0) / 1600);
      setSweep(p);
      if (p < 1) raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [step]);

  useEffect(() => {
    if (!playing) return;
    const id = setTimeout(() => {
      if (step >= STEPS.length - 1) setPlaying(false);
      else setStep((s) => s + 1);
    }, step === 4 ? 2600 : 2000);
    return () => clearTimeout(id);
  }, [playing, step]);

  const S = 150;
  const rW = Math.max(9, (rank / ARCH.hidden) * S);
  const dense = denseParams(Q);
  const lora = loraParams(Q, rank);
  const alpha = alphaFor(rank);

  const showFull = step === 1;
  const frozen = step >= 2;
  const showBA = step >= 3;
  const showDelta = step >= 4;
  const showMerge = step >= 6;

  return (
    <div className="card anim">
      <div className="card-head">
        <h2>{STEPS[step].title}</h2>
        <div className="sub">
          Step {step + 1} of {STEPS.length} · <code>q_proj</code>, layer 1 of {ARCH.layers}
        </div>
      </div>

      <div className="anim-stage">
        <svg viewBox="0 0 726 250" width="100%" role="img"
             aria-label={STEPS[step].title}>
          {/* ---- W ---- */}
          <Matrix
            x={70} y={45} rows={Q.out} cols={Q.in} scale={S}
            label="W"
            fill={showFull ? "var(--series-full)" : "var(--series-lora)"}
            frozen={frozen}
          />
          <text x={70 + S / 2} y={228} textAnchor="middle" fontSize="10.5"
                fill="var(--text-muted)">
            {showFull ? `${(dense / 1e6).toFixed(1)}M trainable` : frozen ? "0 trainable" : `${(dense / 1e6).toFixed(1)}M weights`}
          </text>

          {/* ---- B x A ---- */}
          {showBA && (
            <g className="fade-in">
              <Plus x={247} y={125} />
              <Matrix x={276} y={45} rows={Q.out} cols={rank} scale={S}
                      label="B" fill="var(--series-lora)" dims={false} />
              <text x={276 + rW / 2} y={212} textAnchor="middle" fontSize="9.5"
                    fill="var(--text-secondary)">{Q.out.toLocaleString()}×{rank}</text>
              <Times x={276 + rW + 22} y={125} />
              <Matrix x={276 + rW + 44} y={45 + S / 2 - Math.max(9, rW) / 2}
                      rows={rank} cols={Q.in} scale={S}
                      label="A" fill="var(--series-lora)" dims={false} />
              <text x={276 + rW + 44 + S / 2} y={212} textAnchor="middle" fontSize="9.5"
                    fill="var(--text-secondary)">{rank}×{Q.in.toLocaleString()}</text>
              <text x={276 + rW + 44 + S / 2} y={228} textAnchor="middle"
                    fontSize="10.5" fill="var(--text-muted)">
                {lora.toLocaleString()} trainable
              </text>
            </g>
          )}

          {/* ---- delta W ---- */}
          {showDelta && (
            <g className="fade-in">
              <Arrow x={276 + rW + 44 + S + 12} y={125} w={22} />
              <Matrix
                x={276 + rW + 44 + S + 74} y={45}
                rows={Q.out} cols={Q.in} scale={S}
                label={showMerge ? `W + ${(alpha / rank).toFixed(1)}·BA` : "ΔW = BA"}
                fill={showMerge ? "var(--series-full)" : "var(--series-lora)"}
                sweep={1 - sweep}
              />
            </g>
          )}
        </svg>
      </div>

      <p className="anim-body">{STEPS[step].body}</p>

      {step >= 3 && (
        <ScaleBar
          big={dense} small={lora}
          bigLabel="W" smallLabel={`B and A at r=${rank}`}
        />
      )}
      {step >= 3 && (
        <p className="muted anim-scale-note">
          The B and A slivers are drawn at a minimum visible width — at true scale a
          rank-{rank} column beside a 3,584-wide matrix would be under a pixel. The bar
          above is to scale.
        </p>
      )}

      <div className="anim-controls">
        <button className="chip" onClick={() => { setPlaying(false); setStep((s) => Math.max(0, s - 1)); }}
                disabled={step === 0}>← back</button>
        <button className="chip primary" onClick={() => setPlaying((p) => !p)}>
          {playing ? "pause" : step === STEPS.length - 1 ? "replay" : "play"}
        </button>
        <button className="chip" onClick={() => { setPlaying(false); setStep((s) => Math.min(STEPS.length - 1, s + 1)); }}
                disabled={step === STEPS.length - 1}>next →</button>
        <div className="dots">
          {STEPS.map((_, i) => (
            <button key={i} className={`dot${i === step ? " on" : ""}`}
                    aria-label={`step ${i + 1}`}
                    onClick={() => { setPlaying(false); setStep(i); }} />
          ))}
        </div>
      </div>
    </div>
  );
}
