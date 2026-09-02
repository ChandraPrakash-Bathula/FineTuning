import { useMemo, useState } from "react";
import type { Lab } from "../lib/registry";
import { METHOD_LABEL } from "../lib/registry";
import type { Method } from "../lib/types";
import { ARCH, RANKS, TARGET_MODULES, loraParams } from "../lib/arch";
import type { GpuId } from "../lib/cost";
import { Matrix, sharedUnit, matrixSize } from "../components/Matrix";
import ExperimentRunner from "../components/ExperimentRunner";
import MetricComparison from "../components/MetricComparison";
import ComparisonTable from "../components/ComparisonTable";
import TrainingCurves from "../components/TrainingCurves";
import SqlLab from "../components/SqlLab";
import { useBackend } from "../lib/useBackend";
import Provenance from "../components/Provenance";
import { variantColor, methodColor } from "../lib/palette";

const METHODS: Method[] = ["full_ft", "lora", "qlora"];

const PROS: Record<Method, { good: string[]; bad: string[] }> = {
  full_ft: {
    good: ["Every parameter is free to move — the most expressive option available",
           "No adapter machinery: the output is just a model",
           "The default baseline in most papers, so results compare easily"],
    bad: ["Needs an 80 GB accelerator; will not fit on a 40 GB card",
          "A full model checkpoint per task — ten tasks means ten copies",
          "The most expensive of the three to rent",
          "Scored no better than the adapters on this task"],
  },
  lora: {
    good: ["Hundreds of times fewer trainable parameters",
           "Adapter checkpoints small enough to keep a folder of",
           "Fastest of the three — far fewer gradients to compute",
           "Many adapters can share one copy of the base model"],
    bad: ["The base model still sits in memory in bf16 during training",
          "Rank is one more hyperparameter to justify",
          "Only the targeted modules adapt — here, attention only"],
  },
  qlora: {
    good: ["Lowest memory of the three; fits on hardware people own",
           "Adapter checkpoints identical in size to LoRA's",
           "Accuracy indistinguishable from the alternatives on this task"],
    bad: ["Slowest per step — every matmul dequantises first",
          "Quantisation is lossy; the frozen base is permanently approximated",
          "Serving the adapter on a bf16 base changes its accuracy (see the QLoRA page)"],
  },
};

export default function Playground({ lab, dark }: { lab: Lab; dark: boolean }) {
  const gpuId: GpuId = "h100";
  const [method, setMethod] = useState<Method>("lora");
  const [rank, setRank] = useState(16);
  const [compare, setCompare] = useState<Set<string>>(
    () => new Set(["full_ft", "lora_r16", "qlora_r16"]));
  const { backend } = useBackend();

  const activeId = method === "full_ft" ? "full_ft" : `${method}_r${rank}`;
  const active = lab.byId.get(activeId);
  const selected = useMemo(
    () => lab.variants.filter((v) => compare.has(v.id)), [lab, compare]);

  const toggle = (id: string) => {
    const next = new Set(compare);
    next.has(id) ? next.delete(id) : next.add(id);
    setCompare(next);
  };

  return (
    <article className="page">
      <header className="page-head">
        <div className="eyebrow">Part 4</div>
        <h1>Fine-Tuning Laboratory</h1>
        <p className="lede">
          Eleven experiments were run. Select one, replay what actually happened, and
          compare it against the others. Nothing here trains a model.
        </p>
      </header>

      <section className="card setup">
        <h2>The setup</h2>
        <div className="setup-grid">
          <div><dt>Model</dt><dd><code>{lab.baseModel}</code></dd></div>
          <div><dt>Parameters</dt><dd>{ARCH.totalParams.toLocaleString()}</dd></div>
          <div><dt>Architecture</dt><dd>{ARCH.layers} layers · hidden {ARCH.hidden.toLocaleString()}</dd></div>
          <div><dt>Dataset</dt><dd><code>{lab.dataset}</code></dd></div>
          <div>
            <dt>Optimiser steps</dt>
            <dd title={`Bounded to ${lab.totalSteps.toLocaleString()} by the logging interval and the recorded warmup steps`}>
              ≈ {lab.totalSteps.toLocaleString()} <Provenance source="calculated" />
            </dd>
          </div>
          <div>
            <dt>Training examples</dt>
            <dd title={`Not recorded in registry.json. Bounded to ${lab.trainSizeRange[0].toLocaleString()}–${lab.trainSizeRange[1].toLocaleString()} by steps x batch size / epochs.`}>
              ≈ {lab.trainSizeApprox.toLocaleString()} <Provenance source="calculated" />
            </dd>
          </div>
          <div><dt>Held-out eval</dt><dd>{lab.evalN.toLocaleString()} examples</dd></div>
          <div>
            <dt>Epochs · batch</dt>
            <dd>{lab.epochs} × {lab.effectiveBatchSize}, identical across every run</dd>
          </div>
        </div>
      </section>

      <section className="card step-card">
        <div className="step-label">Select an experiment</div>
        <div className="exp-select">
          <div className="exp-methods" role="radiogroup" aria-label="method">
            {METHODS.map((m) => (
              <button key={m} role="radio" aria-checked={method === m}
                      className={`method-btn${method === m ? " on" : ""}`}
                      style={{ ["--m" as string]: methodColor(m, dark) }}
                      onClick={() => setMethod(m)}>
                <span className="swatch" style={{ background: methodColor(m, dark) }} />
                {METHOD_LABEL[m]}
              </button>
            ))}
          </div>
          <label className={`exp-rank${method === "full_ft" ? " disabled" : ""}`}>
            <span className="muted">rank</span>
            <select value={rank} onChange={(e) => setRank(Number(e.target.value))}
                    disabled={method === "full_ft"} aria-label="LoRA rank">
              {RANKS.map((r) => <option key={r} value={r}>r = {r}</option>)}
            </select>
          </label>
        </div>

        {!active ? (
          <div className="callout warn">
            <strong>Experiment unavailable.</strong>
            <span>No artifact was found for <code>{activeId}</code> in the registry.</span>
          </div>
        ) : (
          <>
            <dl className="exp-state">
              <div><dt>Method</dt><dd>{METHOD_LABEL[active.method]}</dd></div>
              {active.rank !== null && <div><dt>Rank</dt><dd>{active.rank}</dd></div>}
              {active.alpha !== null && (
                <div><dt>Alpha</dt><dd>{active.alpha} (α/r = {active.scaling?.toFixed(1)})</dd></div>
              )}
              <div><dt>Artifact</dt><dd><code>{active.id}</code></dd></div>
              <div><dt>Precision</dt><dd>{active.precision}
                {active.quantization ? ` · ${active.quantization}` : ""}</dd></div>
              <div><dt>Measured experiment</dt><dd>Yes <Provenance source="measured" /></dd></div>
            </dl>

            {active.method !== "full_ft" && (
              <div className="module-strip">
                {TARGET_MODULES.map((m) => {
                  // one unit for the whole figure, so 512 rows look like 512 rows
                  const u = sharedUnit(70, m.out, m.in);
                  const wBox = matrixSize(m.out, m.in, 70, 6, u);
                  const bBox = matrixSize(m.out, rank, 70, 6, u);
                  return (
                    <div key={m.name} className="mini-module">
                      <svg viewBox="0 0 170 96" width="100%" aria-hidden>
                        <Matrix x={16} y={48 - wBox.h / 2} rows={m.out} cols={m.in}
                                unit={u} minPx={6} fill={methodColor(active.method, dark)}
                                frozen dims={false} />
                        <Matrix x={100} y={48 - bBox.h / 2} rows={m.out} cols={rank}
                                unit={u} minPx={6} fill={methodColor(active.method, dark)} dims={false} />
                        <Matrix x={100 + bBox.w + 6} y={48 - Math.max(6, rank * u) / 2}
                                rows={rank} cols={m.in} unit={u} minPx={6}
                                fill={methodColor(active.method, dark)} dims={false} />
                      </svg>
                      <div className="mini-name mono">{m.name}</div>
                      <div className="mini-count">
                        {loraParams(m, rank).toLocaleString()} × {ARCH.layers}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            <ExperimentRunner v={active} gpuId={gpuId}
                              color={methodColor(active.method, dark)} />
            <ProsCons m={active.method} />
          </>
        )}
      </section>

      <section className="card">
        <h2>Compare experiments</h2>
        <p className="body">
          Every variant in the registry. Colour follows the variant, so changing the
          selection never repaints the ones that stay.
        </p>
        <div className="compare-picker">
          {lab.variants.map((v) => (
            <label key={v.id} className="check">
              <input type="checkbox" checked={compare.has(v.id)}
                     onChange={() => toggle(v.id)} />
              <span className="swatch" style={{ background: variantColor(v.method, v.rank, dark) }} />
              {v.label}
            </label>
          ))}
        </div>
        <div className="compare-actions">
          <button className="chip" onClick={() => setCompare(new Set(lab.variants.map((v) => v.id)))}>
            select all {lab.variants.length}
          </button>
          <button className="chip" onClick={() => setCompare(new Set(["full_ft", "lora_r4", "qlora_r4"]))}>
            cheapest of each method
          </button>
          <button className="chip" onClick={() => setCompare(new Set())}>clear</button>
        </div>
      </section>

      {selected.length > 0 && (
        <>
          <MetricComparison variants={selected} dark={dark} gpuId={gpuId}
                            sePts={lab.standardErrorPts} evalN={lab.evalN} />
          <ComparisonTable variants={selected} dark={dark} />
          <TrainingCurves variants={selected} dark={dark} />
        </>
      )}

      <section className="card">
        <h2>SQL laboratory</h2>
        <SqlLab variants={selected} dark={dark} backend={backend} />
      </section>

      <div className="callout closing">
        <strong>
          Eleven experiments, {(lab.accuracyRange[0] * 100).toFixed(1)}–
          {(lab.accuracyRange[1] * 100).toFixed(1)}% execution accuracy, and a{" "}
          {Math.round(lab.checkpointRange[1] / lab.checkpointRange[0])}× spread in
          checkpoint size.
        </strong>
        <span>
          Which would you deploy? And note what this cannot tell you: every run saw the
          same ≈{lab.trainSizeApprox.toLocaleString()} examples. Whether more capacity pays off
          on a harder task, or more data on this one, remains open.
        </span>
      </div>
    </article>
  );
}

function ProsCons({ m }: { m: Method }) {
  const { good, bad } = PROS[m];
  return (
    <div className="proscons">
      <div className="pros">
        <h3>Pros</h3>
        <ul>{good.map((g) => <li key={g}>{g}</li>)}</ul>
      </div>
      <div className="cons">
        <h3>Cons</h3>
        <ul>{bad.map((b) => <li key={b}>{b}</li>)}</ul>
      </div>
    </div>
  );
}
