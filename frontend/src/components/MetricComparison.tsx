import { useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, ErrorBar, LabelList, ReferenceArea,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { Variant } from "../lib/types";
import { variantColor } from "../lib/palette";
import { params, size, duration, pct } from "../lib/format";
import { costUsd, money, type GpuId } from "../lib/cost";
import { axisProps, gridProps, Tip } from "./chart";
import Provenance, { type Source } from "./Provenance";

/** Only metrics that actually exist in the registry, plus the one derived cost. */
interface Metric {
  id: string;
  label: string;
  source: Source;
  value: (v: Variant, gpuId: GpuId) => number;
  fmt: (n: number) => string;
  log?: boolean;
  /** accuracy metrics carry a standard error and must not be read as a ranking */
  errorPts?: boolean;
  note?: string;
}

export const METRICS: Metric[] = [
  { id: "memory", label: "Peak GPU memory", source: "measured",
    value: (v) => v.peakMemoryGb, fmt: (n) => `${n.toFixed(1)} GB` },
  { id: "params", label: "Trainable parameters", source: "measured",
    value: (v) => v.trainable, fmt: params, log: true, note: "log scale" },
  { id: "checkpoint", label: "Checkpoint size", source: "measured",
    value: (v) => v.checkpointMb, fmt: size, log: true, note: "log scale" },
  { id: "time", label: "Training time", source: "measured",
    value: (v) => v.trainingMinutes, fmt: duration },
  { id: "cost", label: "Rental cost", source: "estimated",
    value: (v, g) => costUsd(v.trainingMinutes, g), fmt: money,
    note: "measured training time × the configured hourly rate" },
  { id: "exec", label: "Execution accuracy", source: "measured",
    value: (v) => v.execAccuracy, fmt: (n) => pct(n), errorPts: true },
  { id: "exact", label: "Exact match", source: "measured",
    value: (v) => v.exactMatch, fmt: (n) => pct(n), errorPts: true },
];

export default function MetricComparison({ variants, dark, gpuId, sePts, evalN }: {
  variants: Variant[]; dark: boolean; gpuId: GpuId; sePts: number; evalN: number;
}) {
  const [metricId, setMetricId] = useState("memory");
  const m = METRICS.find((x) => x.id === metricId)!;
  if (!variants.length) return null;

  const se = sePts / 100;
  const data = variants.map((v) => ({
    ...v, value: m.value(v, gpuId), err: m.errorPts ? se : undefined,
  }));
  const values = data.map((d) => d.value);
  const max = Math.max(...values);
  const min = Math.min(...values);

  // Accuracy gets a padded domain and a noise band; everything else starts at 0.
  const domain: [number, number] = m.errorPts
    ? [min - Math.max(se * 2.2, 0.012), max + Math.max(se * 2.2, 0.012)]
    : m.log
      ? [Math.max(min / 3, 0.1), max * 1.6]
      : [0, max * 1.12];

  return (
    <div className="card">
      <div className="card-head chart-head">
        <div>
          <h2>Compare on one metric</h2>
          <div className="sub">
            {m.label} <Provenance source={m.source} />
            {m.note && <> · {m.note}</>}
          </div>
        </div>
        <label className="metric-select">
          <span className="muted">metric</span>
          <select value={metricId} onChange={(e) => setMetricId(e.target.value)}
                  aria-label="comparison metric">
            {METRICS.map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
          </select>
        </label>
      </div>

      <ResponsiveContainer width="100%" height={Math.max(170, variants.length * 32 + 42)}>
        <BarChart data={data} layout="vertical" margin={{ left: 4, right: 84, top: 4, bottom: 4 }}>
          <CartesianGrid {...gridProps} horizontal={false} vertical />
          {m.errorPts && (
            <ReferenceArea x1={Math.max(0, min - se)} x2={max + se}
                           fill="var(--text-muted)" fillOpacity={0.09} stroke="none" />
          )}
          <XAxis type="number" scale={m.log ? "log" : "linear"} domain={domain}
                 {...axisProps} tickFormatter={m.fmt} allowDataOverflow />
          <YAxis type="category" dataKey="label" width={120} {...axisProps} axisLine={false} />
          <Tooltip cursor={{ fill: "var(--raised)" }}
                   content={({ active, payload }) =>
                     active && payload?.length ? (
                       <Tip title={payload[0].payload.label}
                            rows={[[m.label, m.fmt(payload[0].payload.value)
                              + (m.errorPts ? ` ± ${sePts.toFixed(1)} pts` : "")]]} />
                     ) : null} />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={15} isAnimationActive={false}>
            {data.map((d) => (
              <Cell key={d.id} fill={variantColor(d.method, d.rank, dark)} />
            ))}
            <LabelList dataKey="value" position="right"
                       formatter={(x) => (typeof x === "number" ? m.fmt(x) : "")}
                       style={{ fill: "var(--text-secondary)", fontSize: 11 }} />
            {m.errorPts && (
              <ErrorBar dataKey="err" direction="x" width={4} strokeWidth={2}
                        stroke="var(--axis)" />
            )}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {m.errorPts && (
        <p className="chart-note">
          The shaded band spans every selected variant ± one standard error
          (±{sePts.toFixed(1)} points on {evalN.toLocaleString()} examples). Differences
          smaller than about {(sePts * 2).toFixed(0)} points are not distinguishable at
          this eval size — this chart is not a ranking.
        </p>
      )}
    </div>
  );
}
