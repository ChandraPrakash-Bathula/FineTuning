import type { ReactNode } from "react";

/** Recharts axis/grid props bound to the theme tokens, so charts stay recessive. */
export const axisProps = {
  tick: { fill: "var(--text-muted)", fontSize: 11 },
  stroke: "var(--axis)",
  tickLine: false,
} as const;

export const gridProps = {
  stroke: "var(--grid)",
  strokeDasharray: "0",
  vertical: false,
} as const;

export function Tip({ title, rows }: { title: string; rows: [string, ReactNode][] }) {
  return (
    <div className="tip">
      <div className="tip-title">{title}</div>
      {rows.map(([k, v]) => (
        <div key={k} className="tip-row">
          <span className="tip-k">{k}</span>
          <span className="tip-v">{v}</span>
        </div>
      ))}
    </div>
  );
}

export function Legend({ items }: { items: { color: string; label: string }[] }) {
  return (
    <div className="legend">
      {items.map((i) => (
        <span key={i.label} className="legend-item">
          <span className="swatch" style={{ background: i.color }} />
          {i.label}
        </span>
      ))}
    </div>
  );
}

export function ChartCard({
  title, note, children, right,
}: { title: string; note?: ReactNode; children: ReactNode; right?: ReactNode }) {
  return (
    <div className="card">
      <div className="card-head chart-head">
        <div>
          <h2>{title}</h2>
          {note && <div className="sub">{note}</div>}
        </div>
        {right}
      </div>
      {children}
    </div>
  );
}
