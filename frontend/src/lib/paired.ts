/** Paired significance tests, precomputed by scripts/paired_stats.py from the
 *  per-question probe results. See that script for the method. */
import { useEffect, useState } from "react";

export interface PairedResult {
  a: string;
  b: string;
  acc_a: number;
  acc_b: number;
  diff: number;
  both_correct: number;
  neither_correct: number;
  /** questions a got right and b got wrong */
  a_only: number;
  b_only: number;
  mcnemar_p: number;
  mcnemar_p_holm: number;
  bootstrap_ci95: [number, number];
  unpaired_z: number;
  unpaired_p: number;
}

/** A group of models against another, paired by question (sign-flip test). */
export interface GroupResult {
  a: string;
  b: string;
  diff: number;
  p: number;
  bootstrap_ci95: [number, number];
}

export interface PairedStats {
  n_questions: number;
  n_pairs: number;
  bootstrap_resamples: number;
  alpha: number;
  accuracy: Record<string, number>;
  pairs: PairedResult[];
  /** each QLoRA adapter on its 4-bit training base vs a bf16 base */
  serving: PairedResult[];
  groups: {
    adapters_vs_full_ft: GroupResult;
    serving_4bit_vs_bf16: GroupResult;
  };
}

/** For components that only need a number or two: null until loaded. */
export function usePaired(): PairedStats | null {
  const [s, setS] = useState<PairedStats | null>(null);
  useEffect(() => { loadPaired().then(setS, () => setS(null)); }, []);
  return s;
}

let cached: Promise<PairedStats> | null = null;

export function loadPaired(): Promise<PairedStats> {
  return (cached ??= fetch(`${import.meta.env.BASE_URL}paired_stats.json`).then((r) => {
    if (!r.ok) throw new Error(`paired_stats.json: HTTP ${r.status}`);
    return r.json() as Promise<PairedStats>;
  }));
}

/** The pair (a, b) oriented so that `a` is the first argument; a_only/b_only,
 *  diff and the interval are flipped to match. */
export function pairOf(s: PairedStats, a: string, b: string): PairedResult | null {
  const p = s.pairs.find((x) => x.a === a && x.b === b);
  if (p) return p;
  const q = s.pairs.find((x) => x.a === b && x.b === a);
  if (!q) return null;
  return {
    ...q, a, b, acc_a: q.acc_b, acc_b: q.acc_a, diff: -q.diff,
    a_only: q.b_only, b_only: q.a_only,
    bootstrap_ci95: [-q.bootstrap_ci95[1], -q.bootstrap_ci95[0]],
    unpaired_z: -q.unpaired_z,
  };
}

export function fmtP(p: number): string {
  if (p < 0.001) return "< 0.001";
  return p < 0.01 ? p.toFixed(3) : p.toFixed(2);
}

/** Signed percentage points, one decimal. */
export function pts(x: number): string {
  const v = x * 100;
  return `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(1)}`;
}
