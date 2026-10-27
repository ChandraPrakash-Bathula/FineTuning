import type { Lab } from "../lib/registry";
import Provenance from "./Provenance";

/** Parameter counts read from each model's safetensors index on the Hub. */
const QWEN = [
  { id: "0.5B", params: 494_032_768 },
  { id: "1.5B", params: 1_543_714_304 },
  { id: "3B", params: 3_085_938_688 },
  { id: "7B", params: 7_615_616_512 },
];

// Bytes held per parameter before activations, matching this sweep's setup
// (bf16 weights, adamw_torch keeping its two states in the parameter dtype).
const FULL_BPP = 2 + 2 + 2 + 2; // weights, gradients, Adam m, Adam v
const LORA_BPP = 2;             // frozen bf16 weights; adapters are negligible
const QLORA_BPP = 0.5 + 0.127 / 8; // NF4 values + double-quantised block constants

const T4_GB = 16;

/** The same lesson at smaller scale: what each method needs before activations. */
export default function ModelScale({ lab }: { lab: Lab }) {
  const gb = (p: number, bpp: number) => (p * bpp) / 1e9;
  const measured = {
    full: lab.byId.get("full_ft")!.peakMemoryGb,
    lora: lab.byId.get("lora_r16")!.peakMemoryGb,
    qlora: lab.byId.get("qlora_r16")!.peakMemoryGb,
  };
  const cell = (v: number) => (
    <td className={v < T4_GB ? "hi" : ""}>{v.toFixed(1)} GB</td>
  );

  return (
    <section className="card">
      <h2>Too big to try yourself? Scale it down</h2>
      <p className="body">
        This lesson uses a 7B model because that is where the costs become impossible to
        ignore. The same methods work on smaller models in the same Qwen2.5 family. The
        memory each method needs grows with the parameter count, so you can work it out
        before you rent anything.
      </p>
      <div className="scroll-x">
        <table>
          <thead>
            <tr>
              <th>Qwen2.5-Instruct <Provenance source="calculated" /></th>
              <th>Parameters</th><th>Full fine-tune</th><th>LoRA</th><th>QLoRA</th>
            </tr>
          </thead>
          <tbody>
            {QWEN.map((m) => (
              <tr key={m.id}>
                <td>{m.id}</td>
                <td>{(m.params / 1e9).toFixed(2)} B</td>
                {cell(gb(m.params, FULL_BPP))}
                {cell(gb(m.params, LORA_BPP))}
                {cell(gb(m.params, QLORA_BPP))}
              </tr>
            ))}
            <tr className="row-total">
              <td>7B, measured peak <Provenance source="measured" /></td>
              <td></td>
              <td>{measured.full.toFixed(1)} GB</td>
              <td>{measured.lora.toFixed(1)} GB</td>
              <td>{measured.qlora.toFixed(1)} GB</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="muted">
        Calculated rows count only what scales with parameters: weights, plus gradients
        and two Adam states for full fine-tuning ({FULL_BPP} bytes per parameter in this
        sweep's bf16 setup; the common fp32-master setup needs about 16). Activations
        come on top. At 7B they added the gap between the calculated and measured rows,
        at batch 8 × 512 tokens. A smaller batch shrinks them. Green = under 16 GB, the
        size of a free Colab T4, before activations.
      </p>
      <div className="callout">
        <strong>Run a small version yourself</strong>
        <span>
          <code>notebooks/mini_lab.ipynb</code> fine-tunes Qwen2.5-0.5B-Instruct with
          LoRA on a slice of WikiSQL and scores it by execution accuracy, using the same
          prompt format and metric as this lesson. It runs on a free Colab T4 or an
          Apple-silicon laptop. Expect different absolute numbers: a 0.5B model, a
          smaller training set and fewer steps are a different experiment.
        </span>
      </div>
    </section>
  );
}
