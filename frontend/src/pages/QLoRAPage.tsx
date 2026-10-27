import { useState } from "react";
import type { Lab } from "../lib/registry";
import { ARCH, RANKS } from "../lib/arch";
import { pct, size, duration } from "../lib/format";
import QuantViz from "../components/QuantViz";
import QLoRAArchDiagram from "../components/QLoRAArchDiagram";
import ServingBaseComparison from "../components/ServingBaseComparison";
import { usePaired, pairOf, fmtP } from "../lib/paired";

const BLOCK_SIZE = 64;

export default function QLoRAPage({ lab, onNext }: { lab: Lab; onNext: () => void }) {
  const [rank, setRank] = useState(16);
  const q = lab.byId.get(`qlora_r${rank}`)!;
  const l = lab.byId.get(`lora_r${rank}`)!;
  const paired = usePaired();
  const qVsL = paired && pairOf(paired, `qlora_r${rank}`, `lora_r${rank}`);

  const n = ARCH.totalParams;
  const bf16Gb = (n * 2) / 1e9;
  const nf4Gb = (n * 0.5) / 1e9;
  // one fp32 absmax per 64-weight block
  const constGb = ((n / BLOCK_SIZE) * 4) / 1e9;
  // double quant: absmax stored in 8 bits, with a second-level fp32 per 256 blocks
  const dqGb = ((n / BLOCK_SIZE) * 1 + (n / (BLOCK_SIZE * 256)) * 4) / 1e9;

  return (
    <article className="page">
      <header className="page-head">
        <div className="eyebrow">Part 3</div>
        <h1>QLoRA: shrink the part you froze</h1>
        <p className="lede">
          LoRA stopped training the base model but still had to <em>hold</em> it —
          all {bf16Gb.toFixed(1)} GB of it. QLoRA stores those frozen weights in four
          bits instead of sixteen.
        </p>
      </header>

      <QLoRAArchDiagram />

      <section className="card">
        <h2>Four bits, chosen for how weights are actually distributed</h2>
        <p className="body">
          Neural network weights are roughly normally distributed around zero. Splitting
          the range into 16 evenly spaced buckets would waste most of them on the tails
          where almost no weights live. NF4 instead places its 16 levels so that an
          equal share of a normal distribution falls into each — dense near zero, sparse
          at the edges.
        </p>
        <QuantViz />
      </section>

      <section className="card">
        <h2>The memory arithmetic</h2>
        <div className="scroll-x">
          <table>
            <thead>
              <tr><th>Storing {n.toLocaleString()} frozen weights</th><th>Bits each</th><th>Size</th></tr>
            </thead>
            <tbody>
              <tr><td>bf16 — what LoRA holds</td><td>16</td><td>{bf16Gb.toFixed(2)} GB</td></tr>
              <tr><td>NF4 values</td><td>4</td><td>{nf4Gb.toFixed(2)} GB</td></tr>
              <tr>
                <td>+ one fp32 absmax per {BLOCK_SIZE}-weight block</td>
                <td>{(32 / BLOCK_SIZE).toFixed(2)}</td>
                <td>{constGb.toFixed(2)} GB</td>
              </tr>
              <tr>
                <td>+ double quantisation — those constants quantised too</td>
                <td>{((dqGb * 1e9 * 8) / n).toFixed(2)}</td>
                <td>{dqGb.toFixed(2)} GB</td>
              </tr>
              <tr className="row-total">
                <td>NF4 + double quant, total</td>
                <td>{(((nf4Gb + dqGb) * 1e9 * 8) / n).toFixed(2)}</td>
                <td>{(nf4Gb + dqGb).toFixed(2)} GB</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="body">
          That is <strong>{(bf16Gb - nf4Gb - dqGb).toFixed(1)} GB</strong> saved on the
          frozen weights alone. The adapters stay in bf16 — they are the part being
          trained, and they are tiny anyway.
        </p>

        <div className="callout warn">
          <strong>The measured saving is smaller than the arithmetic: {(l.peakMemoryGb - q.peakMemoryGb).toFixed(1)} GB, not {(bf16Gb - nf4Gb - dqGb).toFixed(1)} GB.</strong>
          <span>
            Peak memory is not just weights. Activations, gradient-checkpoint buffers and
            the temporarily dequantised tiles needed for each matmul are all unchanged by
            quantisation. At r={rank}: LoRA peaked at {l.peakMemoryGb.toFixed(1)} GB,
            QLoRA at {q.peakMemoryGb.toFixed(1)} GB. Storage maths and peak-memory
            measurements are different quantities, and this is exactly where students
            expect them to match and they do not.
          </span>
        </div>
      </section>

      <section className="card">
        <h2>The base you serve it on is part of the method</h2>
        <p className="body">
          A QLoRA adapter loads onto a bf16 base without complaint. Both serving modes
          were run over the full held-out set, so the cost of that convenience is
          measurable.
        </p>
        <ServingBaseComparison />
      </section>

      <section className="card">
        <h2>What it costs you</h2>
        <div className="rank-picker">
          <span className="muted">rank r</span>
          {RANKS.map((r) => (
            <button key={r} className={`chip${r === rank ? " on" : ""}`}
                    aria-pressed={r === rank} onClick={() => setRank(r)}>{r}</button>
          ))}
        </div>
        <div className="scroll-x">
          <table>
            <thead>
              <tr><th>at r={rank}</th><th>LoRA</th><th>QLoRA</th><th>Difference</th></tr>
            </thead>
            <tbody>
              <tr>
                <td>Peak GPU memory</td>
                <td>{l.peakMemoryGb.toFixed(1)} GB</td>
                <td className="hi">{q.peakMemoryGb.toFixed(1)} GB</td>
                <td>{(l.peakMemoryGb - q.peakMemoryGb).toFixed(1)} GB less</td>
              </tr>
              <tr>
                <td>Training time</td>
                <td className="hi">{duration(l.trainingMinutes)}</td>
                <td>{duration(q.trainingMinutes)}</td>
                <td>{(q.trainingMinutes - l.trainingMinutes).toFixed(0)} min slower</td>
              </tr>
              <tr>
                <td>Checkpoint</td>
                <td>{size(l.checkpointMb)}</td>
                <td>{size(q.checkpointMb)}</td>
                <td>identical — same adapter shapes</td>
              </tr>
              <tr>
                <td>Execution accuracy</td>
                <td>{pct(l.execAccuracy)}</td>
                <td>{pct(q.execAccuracy)}</td>
                <td>
                  {((q.execAccuracy - l.execAccuracy) * 100).toFixed(1)} pts
                  {qVsL && <> · paired p = {fmtP(qVsL.mcnemar_p)}</>}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="body">
          The trade is explicit: you buy memory with time. Every forward and backward
          pass has to dequantise blocks back to bf16 before it can multiply them, and
          that arithmetic is not free.
        </p>
        <div className="callout">
          <strong>Why it matters: {q.peakMemoryGb.toFixed(1)} GB fits on a 24 GB card.</strong>
          <span>
            Full fine-tuning needed {lab.byId.get("full_ft")!.peakMemoryGb.toFixed(0)} GB
            and an 80 GB accelerator. QLoRA moves this task onto hardware people
            actually own.
          </span>
        </div>
        <button className="chip primary next-btn" onClick={onNext}>
          Go to the playground →
        </button>
      </section>
    </article>
  );
}
