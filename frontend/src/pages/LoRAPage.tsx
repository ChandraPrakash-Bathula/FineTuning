import { useState } from "react";
import type { Lab } from "../lib/registry";
import {
  ARCH, ADAPTED_MATRICES, RANKS, TARGET_MODULES, alphaFor, denseParams,
  loraParams, perLayerDense, perLayerLora, totalLora,
} from "../lib/arch";
import { params, size, pct } from "../lib/format";
import LoRAAnimation from "../components/LoRAAnimation";
import LoRAArchDiagram from "../components/LoRAArchDiagram";
import { Matrix, matrixSize, sharedUnit } from "../components/Matrix";
import RankCurve from "../components/RankCurve";

export default function LoRAPage({ lab, dark, onNext }: { lab: Lab; dark: boolean; onNext: () => void }) {
  const [rank, setRank] = useState(16);
  const v = lab.byId.get(`lora_r${rank}`)!;
  const full = lab.byId.get("full_ft")!;
  const alpha = alphaFor(rank);

  return (
    <article className="page">
      <header className="page-head">
        <div className="eyebrow">Part 2</div>
        <h1>LoRA: train the update, not the model</h1>
        <p className="lede">
          Instead of moving every weight, freeze them all and learn a small,
          low-rank correction that gets added on top.
        </p>
      </header>

      <LoRAArchDiagram />

      <div className="rank-picker">
        <span className="muted">rank r</span>
        {RANKS.map((r) => (
          <button key={r} className={`chip${r === rank ? " on" : ""}`}
                  aria-pressed={r === rank} onClick={() => setRank(r)}>{r}</button>
        ))}
        <span className="muted">alpha = 2r = {alpha}, so alpha/r = 2.0 at every rank</span>
      </div>

      <LoRAAnimation rank={rank} />

      <section className="card">
        <h2>Every matrix LoRA touches, at r={rank}</h2>
        <p className="body">
          The sweep adapted the four attention projections in all {ARCH.layers} layers —{" "}
          <strong>{ADAPTED_MATRICES} matrices</strong>. Note that <code>k_proj</code> and{" "}
          <code>v_proj</code> are much shorter than the others: Qwen2.5 uses grouped-query
          attention, so they output only {ARCH.kvHeads} heads × {ARCH.headDim} ={" "}
          {ARCH.kvHeads * ARCH.headDim} features instead of {ARCH.hidden}.
        </p>

        <div className="module-grid">
          {TARGET_MODULES.map((m) => {
            // A single pixels-per-element unit across W, B and A, so the row
            // count they share is drawn at the same height in all three.
            const u = sharedUnit(90, m.out, m.in);
            const wBox = matrixSize(m.out, m.in, 90, 7, u);
            const bBox = matrixSize(m.out, rank, 90, 7, u);
            const mid = 68;  // vertical centre, leaving headroom for the W/B/A labels
            return (
              <div key={m.name} className="module">
                <h3><code>{m.name}</code></h3>
                {/* dims live in the HTML captions below, not as SVG text --
                    overlaying them on the figure collided at these sizes */}
                <svg viewBox="0 0 250 138" width="100%" aria-hidden>
                  <Matrix x={12} y={mid - wBox.h / 2} rows={m.out} cols={m.in}
                          unit={u} minPx={7} fill="var(--series-lora)" frozen
                          dims={false} label="W" />
                  <Matrix x={140} y={mid - bBox.h / 2} rows={m.out} cols={rank}
                          unit={u} minPx={7} fill="var(--series-lora)" dims={false} label="B" />
                  <Matrix x={140 + bBox.w + 12} y={mid - Math.max(7, rank * u) / 2}
                          rows={rank} cols={m.in} unit={u} minPx={7}
                          fill="var(--series-lora)" dims={false} label="A" />
                  <text x={12 + wBox.w / 2} y={132} textAnchor="middle" fontSize="9.5"
                        fill="var(--text-muted)">{params(denseParams(m))} frozen</text>
                  <text x={190} y={132} textAnchor="middle" fontSize="9.5"
                        fill="var(--series-lora)">
                    {loraParams(m, rank).toLocaleString()} trained
                  </text>
                </svg>
                <div className="module-dims mono">
                  W {m.out.toLocaleString()}×{m.in.toLocaleString()}
                </div>
                <div className="module-dims mono">
                  B {m.out.toLocaleString()}×{rank} · A {rank}×{m.in.toLocaleString()}
                </div>
              </div>
            );
          })}
        </div>

        <div className="scroll-x">
          <table>
            <thead>
              <tr><th>Scope</th><th>Frozen weights</th><th>LoRA trainable</th><th>Ratio</th></tr>
            </thead>
            <tbody>
              <tr>
                <td>One layer (4 matrices)</td>
                <td>{params(perLayerDense)}</td>
                <td>{perLayerLora(rank).toLocaleString()}</td>
                <td>{(perLayerDense / perLayerLora(rank)).toFixed(0)}×</td>
              </tr>
              <tr>
                <td>All {ARCH.layers} layers</td>
                <td>{params(perLayerDense * ARCH.layers)}</td>
                <td>{totalLora(rank).toLocaleString()}</td>
                <td>{(perLayerDense * ARCH.layers / totalLora(rank)).toFixed(0)}×</td>
              </tr>
              <tr>
                <td>Whole model</td>
                <td>{params(ARCH.totalParams)}</td>
                <td>{params(v.trainable)}</td>
                <td>{(ARCH.totalParams / v.trainable).toFixed(0)}×</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="muted">
          The bottom row is exactly what the training run reported:{" "}
          <strong>{v.trainable.toLocaleString()}</strong> trainable parameters,{" "}
          {v.trainablePct.toFixed(3)}% of the model.
        </p>
      </section>

      <section className="card">
        <h2>What each rank actually buys</h2>
        <p className="body">
          Rank is the one knob. It sets how much room the correction has to express
          itself — and it is the only thing changing across these five runs.
        </p>
        <div className="rank-cards">
          {RANKS.map((r) => {
            const rv = lab.byId.get(`lora_r${r}`)!;
            const w = Math.max(6, (r / 64) * 46);
            return (
              <button key={r} className={`rank-card${r === rank ? " on" : ""}`}
                      onClick={() => setRank(r)}>
                <div className="rank-card-head">
                  <strong>r = {r}</strong>
                  <span className="muted">α = {alphaFor(r)}</span>
                </div>
                <svg viewBox="0 0 60 58" width="60" height="58" aria-hidden>
                  <rect x={4} y={6} width={w} height={44} rx="1.5"
                        fill="var(--series-lora)" opacity="0.9" />
                  <rect x={4 + w + 4} y={6} width={46} height={w} rx="1.5"
                        fill="var(--series-lora)" opacity="0.55" />
                </svg>
                <dl className="rank-stats">
                  <div><dt>trainable</dt><dd>{params(rv.trainable)}</dd></div>
                  <div><dt>checkpoint</dt><dd>{size(rv.checkpointMb)}</dd></div>
                  <div><dt>peak mem</dt><dd>{rv.peakMemoryGb.toFixed(1)} GB</dd></div>
                  <div><dt>execution</dt><dd>{pct(rv.execAccuracy)}</dd></div>
                </dl>
              </button>
            );
          })}
        </div>
        <div className="callout">
          <strong>Sixteen times the rank did not buy sixteen times anything.</strong>
          <span>
            r=4 scored {pct(lab.byId.get("lora_r4")!.execAccuracy)} and r=64 scored{" "}
            {pct(lab.byId.get("lora_r64")!.execAccuracy)} — a gap of{" "}
            {((lab.byId.get("lora_r64")!.execAccuracy - lab.byId.get("lora_r4")!.execAccuracy) * 100).toFixed(1)}{" "}
            points, well inside the ±{lab.standardErrorPts.toFixed(1)} standard error on{" "}
            {lab.evalN.toLocaleString()} examples. On this task, capacity was not the
            binding constraint.
          </span>
        </div>
      </section>

      <RankCurve all={lab.variants} ranks={lab.ranks} dark={dark}
                 sePts={lab.standardErrorPts} />

      <section className="card">
        <h2>Against the full fine-tune</h2>
        <div className="versus">
          <Versus label="Trainable parameters"
                  a={params(full.trainable)} b={params(v.trainable)}
                  ratio={`${(full.trainable / v.trainable).toFixed(0)}× fewer`} />
          <Versus label="Checkpoint you ship"
                  a={size(full.checkpointMb)} b={size(v.checkpointMb)}
                  ratio={`${(full.checkpointMb / v.checkpointMb).toFixed(0)}× smaller`} />
          <Versus label="Peak GPU memory"
                  a={`${full.peakMemoryGb.toFixed(1)} GB`} b={`${v.peakMemoryGb.toFixed(1)} GB`}
                  ratio={`${(full.peakMemoryGb - v.peakMemoryGb).toFixed(0)} GB less`} />
          <Versus label="Execution accuracy"
                  a={pct(full.execAccuracy)} b={pct(v.execAccuracy)}
                  ratio="statistically tied" />
        </div>
        <p className="body">
          The checkpoint difference is the one that changes how you work. At{" "}
          {size(v.checkpointMb)} you can keep a folder of adapters, one per task, and
          hot-swap them against a single copy of the base model.
        </p>
        <button className="chip primary next-btn" onClick={onNext}>
          Continue to QLoRA →
        </button>
      </section>
    </article>
  );
}

function Versus({ label, a, b, ratio }: { label: string; a: string; b: string; ratio: string }) {
  return (
    <div className="vs">
      <div className="vs-label">{label}</div>
      <div className="vs-row">
        <span className="vs-a">{a}</span>
        <span className="vs-arrow">→</span>
        <span className="vs-b">{b}</span>
      </div>
      <div className="vs-ratio">{ratio}</div>
    </div>
  );
}
