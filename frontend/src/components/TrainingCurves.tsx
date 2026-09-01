import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { Method, Variant } from "../lib/types";
import { METHOD_LABEL } from "../lib/registry";
import { variantColor } from "../lib/palette";
import { ChartCard, Legend, Tip, axisProps, gridProps } from "./chart";

/** Downsample a long curve so 400+ points don't turn into noise. */
function thin<T>(xs: T[], target = 90): T[] {
  if (xs.length <= target) return xs;
  const step = xs.length / target;
  return Array.from({ length: target }, (_, i) => xs[Math.floor(i * step)]).concat(
    xs[xs.length - 1],
  );
}

function Facet({ method, variants, dark }: { method: Method; variants: Variant[]; dark: boolean }) {
  // Merge each variant's curve onto a shared step axis.
  const steps = new Set<number>();
  const series = variants.map((v) => {
    const pts = thin(v.trainCurve);
    pts.forEach((p) => steps.add(p.step));
    return { v, map: new Map(pts.map((p) => [p.step, p.loss])) };
  });
  const data = [...steps]
    .sort((a, b) => a - b)
    .map((step) => {
      const row: Record<string, number | null> = { step };
      series.forEach(({ v, map }) => (row[v.id] = map.get(step) ?? null));
      return row;
    });

  const losses = variants.flatMap((v) => v.trainCurve.map((p) => p.loss)).filter((l) => l > 0);
  const domain: [number, number] = [
    Math.min(...losses) * 0.8,
    Math.max(...losses) * 1.25,
  ];

  return (
    <div className="card">
      <div className="card-head">
        <h3>{METHOD_LABEL[method]}</h3>
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={data} margin={{ left: 0, right: 8, top: 6, bottom: 4 }}>
          <CartesianGrid {...gridProps} />
          <XAxis dataKey="step" type="number" scale="linear" domain={["dataMin", "dataMax"]}
                 {...axisProps} tickFormatter={(n: number) => (n >= 1000 ? `${Math.round(n / 1000)}k` : `${n}`)} />
          <YAxis
            {...axisProps}
            axisLine={false}
            width={48}
            scale="log"
            domain={domain}
            allowDataOverflow
            tickFormatter={(n: number) => (n >= 1 ? n.toFixed(1) : n.toFixed(3))}
          />
          <Tooltip
            cursor={{ stroke: "var(--axis)" }}
            content={({ active, payload, label }) =>
              active && payload?.length ? (
                <Tip
                  title={`step ${label}`}
                  rows={payload
                    .filter((p) => p.value != null)
                    .map((p) => [
                      variants.find((v) => v.id === p.dataKey)?.label ?? String(p.dataKey),
                      (p.value as number).toFixed(4),
                    ])}
                />
              ) : null
            }
          />
          {variants.map((v) => (
            <Line
              key={v.id}
              dataKey={v.id}
              stroke={variantColor(v.method, v.rank, dark)}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
      <Legend
        items={variants.map((v) => ({
          color: variantColor(v.method, v.rank, dark),
          label: v.rank === null ? "loss" : `r=${v.rank}`,
        }))}
      />
    </div>
  );
}

export default function TrainingCurves({ variants, dark }: { variants: Variant[]; dark: boolean }) {
  const methods = (["full_ft", "lora", "qlora"] as Method[]).filter((m) =>
    variants.some((v) => v.method === m),
  );
  if (!methods.length) return null;

  return (
    <ChartCard
      title="Training loss"
      note="Log loss axis, faceted by method so rank reads as shade within one hue, never as a new colour."
    >
      <div className="facets">
        {methods.map((m) => (
          <Facet key={m} method={m} variants={variants.filter((v) => v.method === m)} dark={dark} />
        ))}
      </div>
      <div className="chart-note">
        Loss is on a log axis: every run falls roughly two orders of magnitude, and the
        mid-training spikes (QLoRA r=64 hits 5.28 at step 3,025) would otherwise flatten
        the whole curve. Those spikes recover within a few hundred steps.
      </div>
    </ChartCard>
  );
}
