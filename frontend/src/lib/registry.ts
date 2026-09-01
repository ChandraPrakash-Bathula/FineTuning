import type { Registry, RawVariant, Variant, Method } from "./types";

// Phase 1 reports decimal units (76.4 GB, 15,243 MB) -- match it exactly
// so the app's figures agree with registry.json and the Phase 1 charts.
const GB = 1e9;
const MB = 1e6;

export const METHOD_LABEL: Record<Method, string> = {
  full_ft: "Full fine-tune",
  lora: "LoRA",
  qlora: "QLoRA",
};

function flatten(v: RawVariant): Variant {
  const c = v.config;
  return {
    id: v.variant_id,
    method: v.method,
    rank: v.rank,
    label: v.rank === null ? METHOD_LABEL[v.method] : `${METHOD_LABEL[v.method]} r=${v.rank}`,
    alpha: c.lora_alpha ?? null,
    scaling: c.lora_alpha != null && v.rank ? c.lora_alpha / v.rank : null,
    targetModules: c.target_modules ?? null,
    precision: c.precision ?? "—",
    quantization: c.quantization
      ? `${c.quantization.bits}-bit ${c.quantization.quant_type}`
      : null,
    learningRate: c.learning_rate ?? null,
    epochs: c.epochs ?? null,
    trainable: v.parameters.trainable,
    total: v.parameters.total,
    trainablePct: v.parameters.trainable_percentage,
    peakMemoryGb: v.resources.peak_memory_allocated_bytes / GB,
    checkpointMb: v.resources.checkpoint_size_bytes / MB,
    trainingMinutes: v.resources.training_time_seconds / 60,
    execAccuracy: v.evaluation.sql_execution_accuracy,
    exactMatch: v.evaluation.sql_exact_match,
    evalLoss: v.evaluation.eval_loss,
    tokenAccuracy: v.evaluation.token_accuracy,
    invalidSql: v.evaluation.num_invalid_sql,
    evalN: v.evaluation.num_examples,
    trainCurve: v.training_curve?.train ?? [],
    evalCurve: (v.training_curve?.eval ?? []).map((p) => ({ step: p.step, loss: p.eval_loss })),
  };
}

export interface Lab {
  baseModel: string;
  dataset: string;
  variants: Variant[];
  byId: Map<string, Variant>;
  /** Standard error on the execution-accuracy estimate, in percentage points. */
  standardErrorPts: number;
  evalN: number;
  ranks: number[];
  /** Total optimiser steps. Bounded, not exact -- see trainSizeRange. */
  totalSteps: number;
  /**
   * The training-set size is NOT recorded in registry.json.
   *
   * Two recorded quantities bound it. Loss is logged every `logEvery` steps, so
   * the last logged step puts the total in [last, last + logEvery). And
   * warmup_steps_resolved = floor(warmup_ratio x total) pins a second interval.
   * Intersecting them gives a range of a few hundred examples, which is what
   * gets shown -- labelled "calculated", never "measured".
   */
  trainSizeApprox: number;
  trainSizeRange: [number, number];
  epochs: number;
  effectiveBatchSize: number;
  accuracyRange: [number, number];
  checkpointRange: [number, number];
}

export function buildLab(reg: Registry): Lab {
  const variants = reg.variants
    .filter((v) => v.status === "completed")
    .map(flatten);

  // SE of a proportion at the observed accuracy, not a hardcoded constant:
  // sqrt(p(1-p)/n), expressed in points.
  const n = variants[0]?.evalN ?? 1000;
  const pBar = variants.reduce((s, v) => s + v.execAccuracy, 0) / (variants.length || 1);
  const standardErrorPts = Math.sqrt((pBar * (1 - pBar)) / n) * 100;

  const ranks = [...new Set(variants.map((v) => v.rank).filter((r): r is number => r !== null))]
    .sort((a, b) => a - b);

  const first = reg.variants[0];
  const curve = first?.training_curve?.train ?? [];
  const logEvery = curve.length > 1 ? curve[1].step - curve[0].step : 25;
  const lastLogged = curve.length ? curve[curve.length - 1].step : 0;
  const epochs = (first?.config.epochs as number) ?? 1;
  const effectiveBatchSize = (first?.config.effective_batch_size as number) ?? 1;

  // Interval 1: logging granularity.
  let lo = lastLogged;
  let hi = lastLogged + logEvery - 1;

  // Interval 2: warmup_steps_resolved = floor(warmup_ratio * total), when recorded.
  const warmup = first?.config.warmup_steps_resolved as number | undefined;
  const ratio = first?.config.warmup_ratio as number | undefined;
  if (warmup && ratio) {
    lo = Math.max(lo, Math.ceil(warmup / ratio));
    hi = Math.min(hi, Math.ceil((warmup + 1) / ratio) - 1);
  }
  if (hi < lo) hi = lo; // recorded values disagree; fall back to the tighter bound

  const totalSteps = Math.round((lo + hi) / 2);
  const perExample = effectiveBatchSize / epochs;
  const trainSizeApprox = Math.round(totalSteps * perExample);
  const trainSizeRange: [number, number] = [
    Math.round(lo * perExample), Math.round(hi * perExample),
  ];

  const accs = variants.map((v) => v.execAccuracy);
  const ckpts = variants.map((v) => v.checkpointMb);

  return {
    baseModel: reg.base_model,
    dataset: reg.dataset,
    variants,
    byId: new Map(variants.map((v) => [v.id, v])),
    standardErrorPts,
    evalN: n,
    ranks,
    totalSteps,
    trainSizeApprox,
    trainSizeRange,
    epochs,
    effectiveBatchSize,
    accuracyRange: [Math.min(...accs), Math.max(...accs)],
    checkpointRange: [Math.min(...ckpts), Math.max(...ckpts)],
  };
}

export async function loadLab(): Promise<Lab> {
  const res = await fetch(`${import.meta.env.BASE_URL}registry.json`);
  if (!res.ok) throw new Error(`registry.json: HTTP ${res.status}`);
  return buildLab((await res.json()) as Registry);
}
