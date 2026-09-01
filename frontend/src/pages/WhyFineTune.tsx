import type { Lab } from "../lib/registry";
import { ARCH } from "../lib/arch";
import { GPU_RATES, costUsd, money } from "../lib/cost";
import { size, duration } from "../lib/format";

export default function WhyFineTune({ lab, onNext }: { lab: Lab; onNext: () => void }) {
  const full = lab.byId.get("full_ft")!;
  const weightsGb = (ARCH.totalParams * 2) / 1e9; // bf16: 2 bytes per parameter

  return (
    <article className="page">
      <header className="page-head">
        <div className="eyebrow">Part 1</div>
        <h1>A model only knows what it was shown</h1>
        <p className="lede">
          A pre-trained model has read an enormous amount of text — but not
          everything, and not your things. Whatever was missing from that data is
          missing from the model.
        </p>
      </header>

      <section className="card">
        <h2>The gap we care about here</h2>
        <p className="body">
          Our task is translating an English question into SQL against a specific
          table, with columns addressed as <code>col0</code>, <code>col1</code>, and the
          table always called <code>"table"</code>. That last part is a house
          convention. No amount of general pre-training teaches a model a convention it
          has never seen.
        </p>
        <div className="example-pair">
          <div>
            <div className="muted">a question</div>
            <p className="q">What position does the player who played for butler cc (ks) play?</p>
          </div>
          <div>
            <div className="muted">the answer we need</div>
            <code className="sql">SELECT col3 FROM "table" WHERE col5 = 'Butler CC (KS)'</code>
          </div>
        </div>
        <p className="body">
          To close a gap like this you keep training the model on data that shows the
          behaviour you want. That is fine-tuning. Every run on this site trained on the
          same <code>{lab.dataset}</code> split for {lab.epochs} epochs —
          about {lab.trainSizeApprox.toLocaleString()} examples. The registry does not
          record the split size directly, so that figure is derived from the recorded
          step count and batch size, not measured.
        </p>
      </section>

      <section className="card">
        <h2>The obvious approach: retrain everything</h2>
        <p className="body">
          {ARCH.totalParams.toLocaleString()} parameters, all of them unfrozen. Every
          weight gets a gradient. The optimiser keeps running statistics on top of
          that, and activations have to be held for the backward pass.
        </p>

        <div className="mem-stack">
          <MemRow label="The weights themselves" detail={`${ARCH.totalParams.toLocaleString()} × 2 bytes (bf16)`} gb={weightsGb} max={full.peakMemoryGb} />
          <MemRow label="+ gradients, optimiser state, activations" detail="gradient checkpointing was enabled, which trades compute to hold this down" gb={full.peakMemoryGb - weightsGb} max={full.peakMemoryGb} accent />
          <div className="mem-total">
            <span>Measured peak</span>
            <strong>{full.peakMemoryGb.toFixed(1)} GB</strong>
          </div>
        </div>

        <h3 className="sub-head">Which cards can even run it</h3>
        <div className="gpu-fit">
          {GPU_RATES.map((g) => {
            const fits = full.peakMemoryGb <= g.memoryGb;
            return (
              <div key={g.id} className={`gpu${fits ? " fits" : " nope"}`}>
                <div className="gpu-name">{g.label}</div>
                <div className="gpu-verdict">
                  {fits ? `✓ fits (${g.memoryGb} GB)` : `✗ ${full.peakMemoryGb.toFixed(0)} GB won't fit in ${g.memoryGb}`}
                </div>
              </div>
            );
          })}
        </div>
        <p className="muted">
          Rates are typical on-demand rentals, not measurements. The run itself was
          done on an H200.
        </p>
      </section>

      <section className="card">
        <h2>What that one run cost</h2>
        <div className="kpi-row">
          <Kpi v={duration(full.trainingMinutes)} l="training time" />
          <Kpi v={`${full.peakMemoryGb.toFixed(1)} GB`} l="peak GPU memory" />
          <Kpi v={size(full.checkpointMb)} l="checkpoint, per task" />
          <Kpi v={money(costUsd(full.trainingMinutes, "h100"))} l="at $2.99/hr on an H100" />
        </div>
        <p className="body">
          The money is survivable. The <strong>{size(full.checkpointMb)} checkpoint</strong> is
          the part that hurts. That is a full copy of the model for one task. Ten tasks
          means ten copies, {size(full.checkpointMb * 10)} on disk, and a fresh set of
          weights to load every time you switch.
        </p>
        <div className="callout">
          <strong>So: can we get the new behaviour without moving 7.62 billion numbers?</strong>
          <span>That question is what LoRA answers.</span>
        </div>
        <button className="chip primary next-btn" onClick={onNext}>
          Continue to LoRA →
        </button>
      </section>
    </article>
  );
}

function MemRow({ label, detail, gb, max, accent }: {
  label: string; detail: string; gb: number; max: number; accent?: boolean;
}) {
  return (
    <div className="mem-row">
      <div className="mem-bar-wrap">
        <div className={`mem-bar${accent ? " accent" : ""}`} style={{ width: `${(gb / max) * 100}%` }} />
      </div>
      <div className="mem-meta">
        <span className="mem-label">{label}</span>
        <span className="mem-gb">{gb.toFixed(1)} GB</span>
      </div>
      <div className="muted">{detail}</div>
    </div>
  );
}

function Kpi({ v, l }: { v: string; l: string }) {
  return (
    <div className="stat">
      <div className="stat-v">{v}</div>
      <div className="stat-l">{l}</div>
    </div>
  );
}
