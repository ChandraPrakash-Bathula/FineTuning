import { useEffect, useRef, useState } from "react";
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { Variant } from "../lib/types";
import { costUsd, gpu, money, type GpuId } from "../lib/cost";
import { size, duration, pct, params } from "../lib/format";
import { axisProps, gridProps, Tip } from "./chart";
import Provenance from "./Provenance";
import { prefersReducedMotion } from "../lib/motion";

/** The states an experiment can be in. Never a fake percentage. */
export type Phase = "idle" | "loading" | "replaying" | "complete";

const PHASE_COPY: Record<Phase, string> = {
  idle: "Not loaded",
  loading: "Loading experiment record",
  replaying: "Replaying measured run",
  complete: "Experiment loaded",
};

const REPLAY_MS = 4200;

export default function ExperimentRunner({ v, color, gpuId }: {
  v: Variant; color: string; gpuId: GpuId;
}) {
  const [phase, setPhase] = useState<Phase>("idle");
  const [p, setP] = useState(0);
  const raf = useRef(0);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  // Reset whenever the selected experiment changes.
  useEffect(() => {
    setPhase("idle");
    setP(0);
    cancelAnimationFrame(raf.current);
    clearTimeout(timer.current);
  }, [v.id]);

  useEffect(() => () => {
    cancelAnimationFrame(raf.current);
    clearTimeout(timer.current);
  }, []);

  const run = () => {
    cancelAnimationFrame(raf.current);
    clearTimeout(timer.current);
    setPhase("loading");
    setP(0);
    timer.current = setTimeout(() => {
      if (prefersReducedMotion()) { setP(1); setPhase("complete"); return; }
      setPhase("replaying");
      const t0 = performance.now();
      const tick = (t: number) => {
        const prog = Math.min(1, (t - t0) / REPLAY_MS);
        setP(prog);
        if (prog < 1) raf.current = requestAnimationFrame(tick);
        else setPhase("complete");
      };
      raf.current = requestAnimationFrame(tick);
    }, 420);
  };

  const curve = v.trainCurve;
  const shown = phase === "complete" ? curve : curve.slice(0, Math.max(1, Math.floor(p * curve.length)));
  const at = shown.at(-1);
  const card = gpu(gpuId);
  const fits = v.peakMemoryGb <= card.memoryGb;
  const losses = curve.map((d) => d.loss).filter((l) => l > 0);
  const domain: [number, number] = [Math.min(...losses) * 0.8, Math.max(...losses) * 1.25];

  return (
    <div className="runner" style={{ ["--accent" as string]: color }}>
      <div className="runner-state">
        <span className={`state-dot state-${phase}`} aria-hidden />
        <span className="state-label">{PHASE_COPY[phase]}</span>
        <span className="muted">
          artifact <code>{v.id}</code> · {curve.length} recorded loss points
        </span>
        {phase !== "idle" && (
          <button className="chip runner-reset" onClick={run}>
            {phase === "complete" ? "replay" : "restart"}
          </button>
        )}
      </div>

      {phase === "idle" ? (
        <button className="train-btn" onClick={run}>
          Run experiment
          <span className="train-btn-sub">replays the measured run — nothing is trained</span>
        </button>
      ) : (
        <>
          <div className="replay-chart">
            <div className="replay-head">
              <span>Training loss <Provenance source="measured" /></span>
              <span className="mono replay-readout">
                {at ? `step ${at.step.toLocaleString()} · loss ${at.loss.toFixed(4)}` : "—"}
              </span>
            </div>
            <ResponsiveContainer width="100%" height={190}>
              <LineChart data={shown} margin={{ left: 0, right: 10, top: 6, bottom: 2 }}>
                <CartesianGrid {...gridProps} />
                <XAxis dataKey="step" type="number" scale="linear"
                       domain={[curve[0]?.step ?? 0, curve.at(-1)?.step ?? 1]}
                       {...axisProps} allowDataOverflow
                       tickFormatter={(n: number) => (n >= 1000 ? `${Math.round(n / 1000)}k` : `${n}`)} />
                <YAxis {...axisProps} axisLine={false} width={48} scale="log"
                       domain={domain} allowDataOverflow
                       tickFormatter={(n: number) => (n >= 1 ? n.toFixed(1) : n.toFixed(3))} />
                <Tooltip cursor={{ stroke: "var(--axis)" }}
                         content={({ active, payload, label }) =>
                           active && payload?.length ? (
                             <Tip title={`step ${label}`}
                                  rows={[["loss", (payload[0].value as number).toFixed(4)]]} />
                           ) : null} />
                <Line dataKey="loss" stroke={color} strokeWidth={2} dot={false}
                      isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
            <p className="muted">
              The real recorded curve, log axis, spikes preserved as they occurred.
            </p>
          </div>

          {phase === "complete" && (
            <div className="runner-metrics">
              <div className="kpi-row">
                <Kpi v={duration(v.trainingMinutes)} l="training time" s="measured" />
                <Kpi v={`${v.peakMemoryGb.toFixed(1)} GB`} l="peak GPU memory" s="measured"
                     warn={!fits} sub={fits ? undefined : `exceeds ${card.label}`} />
                <Kpi v={size(v.checkpointMb)} l="checkpoint" s="measured" />
                <Kpi v={money(costUsd(v.trainingMinutes, gpuId))}
                     l={`on ${card.label} at $${card.usdPerHour}/hr`} s="estimated" />
              </div>
              <div className="kpi-row second">
                <Kpi v={params(v.trainable)} l="trainable parameters" s="measured" />
                <Kpi v={pct(v.execAccuracy)} l="execution accuracy" s="measured" />
                <Kpi v={pct(v.exactMatch)} l="exact match" s="measured" />
                <Kpi v={v.evalLoss.toFixed(4)} l="eval loss" s="measured" />
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function Kpi({ v, l, s, sub, warn }: {
  v: string; l: string; s: "measured" | "estimated"; sub?: string; warn?: boolean;
}) {
  return (
    <div className="stat">
      <div className={`stat-v${warn ? " warn" : ""}`}>{v}</div>
      <div className="stat-l">
        {l} <Provenance source={s} />
        {sub && <><br /><span className="warn-text">{sub}</span></>}
      </div>
    </div>
  );
}
