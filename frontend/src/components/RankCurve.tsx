import {
  CartesianGrid, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";
import type { Variant } from "../lib/types";
import { methodColor } from "../lib/palette";
import { ChartCard, Legend, Tip, axisProps, gridProps } from "./chart";

export default function RankCurve({
  all, ranks, dark, sePts,
}: { all: Variant[]; ranks: number[]; dark: boolean; sePts: number }) {
  const se = sePts / 100;
  const full = all.find((v) => v.method === "full_ft");
  const byMethod = (m: "lora" | "qlora") =>
    Object.fromEntries(all.filter((v) => v.method === m).map((v) => [v.rank!, v]));
  const lora = byMethod("lora");
  const qlora = byMethod("qlora");

  const data = ranks.map((r) => ({
    rank: r,
    lora: lora[r]?.execAccuracy ?? null,
    qlora: qlora[r]?.execAccuracy ?? null,
  }));

  const loraColor = methodColor("lora", dark);
  const qloraColor = methodColor("qlora", dark);
  const fullColor = methodColor("full_ft", dark);

  const vals = data.flatMap((d) => [d.lora, d.qlora]).filter((x): x is number => x != null);
  const lo = Math.min(...vals, full?.execAccuracy ?? 1);
  const hi = Math.max(...vals, full?.execAccuracy ?? 0);
  const pad = Math.max(se * 2.2, 0.012);

  return (
    <ChartCard
      title="Does rank buy accuracy?"
      note="Execution accuracy against LoRA rank, both methods overlaid."
    >
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={data} margin={{ left: 4, right: 8, top: 8, bottom: 14 }}>
          <CartesianGrid {...gridProps} />
          {full && (
            <ReferenceArea
              y1={full.execAccuracy - se}
              y2={full.execAccuracy + se}
              fill={fullColor}
              fillOpacity={0.12}
              stroke="none"
            />
          )}
          {full && (
            <ReferenceLine
              y={full.execAccuracy}
              stroke={fullColor}
              strokeWidth={2}
              strokeDasharray="5 4"
              label={{
                value: `full fine-tune ${(full.execAccuracy * 100).toFixed(1)}%`,
                position: "insideTopLeft",
                fill: "var(--text-secondary)",
                fontSize: 11,
              }}
            />
          )}
          <XAxis
            dataKey="rank"
            type="number"
            scale="log"
            domain={[3, 85]}
            ticks={ranks}
            {...axisProps}
            label={{ value: "rank (log)", position: "insideBottom", offset: -2,
                     fill: "var(--text-muted)", fontSize: 11 }}
          />
          <YAxis
            domain={[lo - pad, hi + pad]}
            {...axisProps}
            axisLine={false}
            tickFormatter={(n: number) => `${(n * 100).toFixed(1)}%`}
          />
          <Tooltip
            cursor={{ stroke: "var(--axis)", strokeWidth: 1 }}
            content={({ active, payload, label }) =>
              active && payload?.length ? (
                <Tip
                  title={`rank ${label}`}
                  rows={payload.map((p) => [
                    p.dataKey === "lora" ? "LoRA" : "QLoRA",
                    `${((p.value as number) * 100).toFixed(1)}%`,
                  ])}
                />
              ) : null
            }
          />
          <Line type="monotone" dataKey="lora" stroke={loraColor} strokeWidth={2}
                dot={{ r: 4, fill: loraColor, stroke: "var(--surface-1)", strokeWidth: 2 }}
                isAnimationActive={false} connectNulls />
          <Line type="monotone" dataKey="qlora" stroke={qloraColor} strokeWidth={2}
                dot={{ r: 4, fill: qloraColor, stroke: "var(--surface-1)", strokeWidth: 2 }}
                isAnimationActive={false} connectNulls />
        </LineChart>
      </ResponsiveContainer>
      <Legend
        items={[
          { color: loraColor, label: "LoRA" },
          { color: qloraColor, label: "QLoRA" },
          { color: fullColor, label: "Full fine-tune (± 1 s.e.)" },
        ]}
      />
      <div className="chart-note">
        The band is one standard error around the full fine-tune. Every adapter run
        lands inside or beside it: sixteen times more rank does not move the metric.
      </div>
    </ChartCard>
  );
}
