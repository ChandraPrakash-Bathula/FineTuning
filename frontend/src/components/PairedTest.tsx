import { useEffect, useState } from "react";
import type { Lab } from "../lib/registry";
import { loadPaired, pairOf, fmtP, pts, type PairedStats } from "../lib/paired";
import Provenance from "./Provenance";

/** Is the gap between two variants real? Answered with a paired test, because
 *  every variant answered the same 1,000 questions. */
export default function PairedTest({ lab }: { lab: Lab }) {
  const [stats, setStats] = useState<PairedStats | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [a, setA] = useState("full_ft");
  const [b, setB] = useState("lora_r8");

  useEffect(() => { loadPaired().then(setStats, (e) => setError(String(e))); }, []);

  if (error) return <div className="card"><p className="body">Could not load the paired tests: {error}</p></div>;
  if (!stats) return <div className="card"><p className="muted">Loading paired tests…</p></div>;

  const label = (id: string) => lab.byId.get(id)?.label ?? id;
  const p = a === b ? null : pairOf(stats, a, b);
  const n = stats.n_questions;
  const sig = (x: number) => x < stats.alpha;
  const adapters = lab.variants.filter((v) => v.method !== "full_ft").map((v) => v.id);
  const nSigRaw = stats.pairs.filter((x) => sig(x.mcnemar_p)).length;
  const nSigHolm = stats.pairs.filter((x) => sig(x.mcnemar_p_holm)).length;
  const g = stats.groups.adapters_vs_full_ft;

  const picker = (value: string, set: (s: string) => void, aria: string) => (
    <select value={value} onChange={(e) => set(e.target.value)} aria-label={aria}>
      {lab.variants.map((v) => <option key={v.id} value={v.id}>{v.label}</option>)}
    </select>
  );

  return (
    <section className="card">
      <h2>Is a gap real? Use a paired test</h2>
      <p className="body">
        All eleven models answered the <strong>same {n.toLocaleString()} questions</strong>.
        So their scores are not independent: a question that is easy for one model is
        usually easy for the others. A fair comparison looks only at the questions where
        two models <em>disagree</em>.
      </p>

      <div className="rank-picker">
        <span className="muted">compare</span>
        {picker(a, setA, "first variant")}
        <span className="muted">with</span>
        {picker(b, setB, "second variant")}
        <Provenance source="recorded" />
      </div>

      {!p ? (
        <p className="muted">Pick two different variants.</p>
      ) : (
        <>
          <div className="paired-grid">
            <div className="scroll-x">
              <table className="contingency">
                <thead>
                  <tr><th></th><th>{label(b)} right</th><th>{label(b)} wrong</th></tr>
                </thead>
                <tbody>
                  <tr>
                    <td>{label(a)} right</td>
                    <td className="dim">{p.both_correct}</td>
                    <td className="disc">{p.a_only}</td>
                  </tr>
                  <tr>
                    <td>{label(a)} wrong</td>
                    <td className="disc">{p.b_only}</td>
                    <td className="dim">{p.neither_correct}</td>
                  </tr>
                </tbody>
              </table>
              <p className="muted">
                Grey cells: both agree, so they say nothing about which is better.
                Highlighted cells: the {p.a_only + p.b_only} questions that decide it.
              </p>
            </div>

            <dl className="paired-result">
              <div>
                <dt>Accuracy difference</dt>
                <dd>{pts(p.diff)} pts <span className="muted">({(p.acc_a * 100).toFixed(1)}% vs {(p.acc_b * 100).toFixed(1)}%)</span></dd>
              </div>
              <div>
                <dt>McNemar's exact test</dt>
                <dd className={sig(p.mcnemar_p) ? "warn-text" : ""}>p = {fmtP(p.mcnemar_p)}</dd>
              </div>
              <div>
                <dt>Paired bootstrap, 95% interval</dt>
                <dd>{pts(p.bootstrap_ci95[0])} to {pts(p.bootstrap_ci95[1])} pts</dd>
              </div>
              <div>
                <dt>After correcting for {stats.n_pairs} comparisons (Holm)</dt>
                <dd>p = {fmtP(p.mcnemar_p_holm)}</dd>
              </div>
              <div className="was">
                <dt>Unpaired z-test, for contrast</dt>
                <dd>z = {p.unpaired_z.toFixed(2)}, p = {fmtP(p.unpaired_p)}</dd>
              </div>
            </dl>
          </div>

          <ol className="body paired-steps">
            <li>
              If the two models were equally good, each of the {p.a_only + p.b_only} disagreements
              would be a coin flip. {label(a)} won {p.a_only} and {label(b)} won {p.b_only}.
              McNemar's test asks how surprising a split that uneven would be:{" "}
              <strong>p = {fmtP(p.mcnemar_p)}</strong>.
            </li>
            <li>
              The bootstrap resamples the {n.toLocaleString()} questions{" "}
              {stats.bootstrap_resamples.toLocaleString()} times, keeping each question's pair
              of answers together. {p.bootstrap_ci95[0] > 0 || p.bootstrap_ci95[1] < 0
                ? "The interval does not include 0."
                : "The interval includes 0, so no difference is a plausible answer."}
            </li>
            <li>
              The unpaired test treats the two scores as if they came from different
              questions. That throws away the pairing, so it overstates the noise
              {sig(p.mcnemar_p) && !sig(p.unpaired_p) && <> — here it calls a paired-significant gap "not significant"</>}.
            </li>
          </ol>
        </>
      )}

      <h3 className="sub-head">Full fine-tuning against every adapter</h3>
      <div className="scroll-x">
        <table>
          <thead>
            <tr>
              <th>Adapter</th><th>Adapter − full FT</th><th>Only adapter right</th>
              <th>Only full FT right</th><th>McNemar p</th><th>Holm p</th><th>95% interval</th>
            </tr>
          </thead>
          <tbody>
            {adapters.map((id) => {
              const r = pairOf(stats, id, "full_ft")!;
              return (
                <tr key={id} className={id === a || id === b ? "row-on" : ""}>
                  <td>{label(id)}</td>
                  <td>{pts(r.diff)}</td>
                  <td>{r.a_only}</td>
                  <td>{r.b_only}</td>
                  <td className={sig(r.mcnemar_p) ? "warn-text" : ""}>{fmtP(r.mcnemar_p)}</td>
                  <td>{fmtP(r.mcnemar_p_holm)}</td>
                  <td>{pts(r.bootstrap_ci95[0])} to {pts(r.bootstrap_ci95[1])}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="body">
        One question, asked once, avoids testing all {stats.n_pairs} pairs: <em>do the
        adapters, on average, score differently from full fine-tuning?</em> For each
        question, take the share of the ten adapters that got it right, subtract whether
        full fine-tuning got it right, and test the average with a sign-flip permutation
        test.
      </p>
      <dl className="paired-result inline">
        <div><dt>Mean adapter − full FT</dt><dd>{pts(g.diff)} pts</dd></div>
        <div><dt>p (sign-flip, paired)</dt><dd className={sig(g.p) ? "warn-text" : ""}>{fmtP(g.p)}</dd></div>
        <div><dt>95% interval</dt><dd>{pts(g.bootstrap_ci95[0])} to {pts(g.bootstrap_ci95[1])} pts</dd></div>
      </dl>

      <div className="callout warn">
        <strong>What the paired test changes</strong>
        <span>
          The first version of this lesson used an unpaired test, called every gap "not
          significant", and concluded that full fine-tuning bought nothing. Done
          properly, the comparison says more: on these {n.toLocaleString()} questions,{" "}
          <strong>full fine-tuning scored lower than every adapter, about{" "}
          {(g.diff * 100).toFixed(0)} points below their average</strong> (p ={" "}
          {fmtP(g.p)}). Picking out single pairs is weaker evidence: {nSigRaw} of the{" "}
          {stats.n_pairs} pairs reach p &lt; 0.05, but with {stats.n_pairs} tests about{" "}
          {(stats.n_pairs * stats.alpha).toFixed(1)} would by chance alone, and after Holm's
          correction {nSigHolm === 0 ? "none" : nSigHolm} survive.
        </span>
      </div>

      <div className="callout">
        <strong>What these tests can and cannot tell you</strong>
        <span>
          They measure one source of noise: which questions happened to be in the test
          set. They say nothing about training noise. Every run used one seed (42), and
          full fine-tuning was trained at one learning rate (1e-5) for three epochs. A
          different seed or learning rate could move it by more than these gaps. So the
          evidence supports "in this setup, these trained models differ on these
          questions". It does not support "LoRA beats full fine-tuning" in general, on
          other models, or on other tasks. What would you need to run to test that?
        </span>
      </div>
      <p className="muted">
        Computed from the recorded probe runs, which hold a right/wrong result for every
        question; every accuracy in this app comes from the same runs. The evaluation
        run right after training stored only totals, which are not enough for a paired
        test, and differs from these by up to 0.4 points (full FT: 75.4% here, 75.3% at
        training). Re-generating alone moves a score that much, so gaps of that size
        are noise. Script: <code>scripts/paired_stats.py</code>.
      </p>
    </section>
  );
}
