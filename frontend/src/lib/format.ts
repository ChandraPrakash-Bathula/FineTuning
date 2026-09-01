export const pct = (x: number, dp = 1) => `${(x * 100).toFixed(dp)}%`;

export function params(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

/** Decimal units throughout, matching Phase 1's own reporting. */
export function size(mb: number): string {
  return mb >= 1000 ? `${(mb / 1000).toFixed(2)} GB` : `${mb.toFixed(0)} MB`;
}

export function duration(min: number): string {
  const h = Math.floor(min / 60);
  const m = Math.round(min % 60);
  return h ? `${h}h ${m}m` : `${m}m`;
}

/** "col3: Position" -> { col: "col3", name: "Position" } */
export function parseSchema(rows: string[]) {
  return rows.map((r) => {
    const i = r.indexOf(":");
    return i < 0
      ? { col: r.trim(), name: "" }
      : { col: r.slice(0, i).trim(), name: r.slice(i + 1).trim() };
  });
}
