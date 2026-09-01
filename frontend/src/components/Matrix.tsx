/**
 * SVG matrix primitives for the lesson figures.
 *
 * Honesty note: a rank-4 sliver next to a 3584-wide matrix is a third of a
 * pixel. Every matrix here therefore has a minimum drawn thickness, and any
 * figure that clamps says so and pairs itself with a to-scale area bar.
 */
interface MatrixProps {
  rows: number;
  cols: number;
  /** drawn size in px for the LONG side; the short side scales from it */
  scale?: number;
  fill?: string;
  label?: string;
  /** show the row/col counts along the edges */
  dims?: boolean;
  frozen?: boolean;
  x?: number;
  y?: number;
  /** 0..1 sweep for the multiplication animation */
  sweep?: number;
  minPx?: number;
  /**
   * Pixels per element. Pass the SAME unit to every matrix in a figure so a
   * shared dimension draws at a shared size -- otherwise W (512x3584) and
   * B (512x16) render their common 512 rows at wildly different heights.
   */
  unit?: number;
}

const CELL = 5;

export function matrixSize(rows: number, cols: number, scale = 150, minPx = 9, unit?: number) {
  const u = unit ?? scale / Math.max(rows, cols);
  return { w: Math.max(minPx, cols * u), h: Math.max(minPx, rows * u) };
}

/** One pixels-per-element value that keeps every matrix in a figure comparable. */
export const sharedUnit = (scale: number, ...dims: number[]) =>
  scale / Math.max(...dims);

export function Matrix({
  rows, cols, scale = 150, fill = "var(--series)", label, dims = true,
  frozen = false, x = 0, y = 0, sweep, minPx = 9, unit,
}: MatrixProps) {
  const { w, h } = matrixSize(rows, cols, scale, minPx, unit);
  const id = `g${rows}x${cols}${label ?? ""}`.replace(/\W/g, "");

  return (
    <g transform={`translate(${x},${y})`}>
      <defs>
        <pattern id={id} width={CELL} height={CELL} patternUnits="userSpaceOnUse">
          <rect width={CELL} height={CELL} fill="none" />
          <path d={`M ${CELL} 0 L 0 0 0 ${CELL}`} fill="none"
                stroke="var(--surface-1)" strokeWidth="0.6" opacity="0.55" />
        </pattern>
      </defs>

      <rect
        width={w} height={h} rx="2"
        fill={frozen ? "var(--grid)" : fill}
        stroke={frozen ? "var(--axis)" : fill}
        strokeWidth="1"
        opacity={frozen ? 1 : 0.92}
      />
      <rect width={w} height={h} rx="2" fill={`url(#${id})`} pointerEvents="none" />

      {/* multiplication sweep: the column of the output being computed */}
      {sweep !== undefined && sweep > 0 && (
        <rect
          x={0} y={0} width={w * Math.min(1, sweep)} height={h}
          fill="var(--surface-1)" opacity="0.34" pointerEvents="none"
        />
      )}

      {frozen && w > 46 && h > 18 && (
        <text x={w / 2} y={h / 2 + 4} textAnchor="middle"
              fontSize="11" fill="var(--text-muted)">frozen</text>
      )}

      {label && (
        <text x={w / 2} y={-8} textAnchor="middle" fontSize="12"
              fontWeight="600" fill="var(--text-primary)">{label}</text>
      )}

      {dims && (
        <>
          {/* columns along the bottom edge */}
          <text x={w / 2} y={h + 14} textAnchor="middle" fontSize="10.5"
                fill="var(--text-secondary)"
                style={{ fontVariantNumeric: "tabular-nums" }}>
            {cols.toLocaleString()}
          </text>
          {/* rows down the left edge */}
          <text x={-7} y={h / 2 + 3.5} textAnchor="end" fontSize="10.5"
                fill="var(--text-secondary)"
                style={{ fontVariantNumeric: "tabular-nums" }}>
            {rows.toLocaleString()}
          </text>
        </>
      )}
    </g>
  );
}

export function Times({ x, y }: { x: number; y: number }) {
  return (
    <text x={x} y={y} textAnchor="middle" fontSize="17" fill="var(--text-muted)">×</text>
  );
}

export function Plus({ x, y }: { x: number; y: number }) {
  return (
    <text x={x} y={y} textAnchor="middle" fontSize="17" fill="var(--text-muted)">+</text>
  );
}

export function Arrow({ x, y, w = 30 }: { x: number; y: number; w?: number }) {
  return (
    <g>
      <line x1={x} y1={y} x2={x + w} y2={y} stroke="var(--axis)" strokeWidth="1.5" />
      <path d={`M ${x + w} ${y} l -6 -4 v 8 z`} fill="var(--axis)" />
    </g>
  );
}

/** An honest, to-scale comparison of two cell counts. */
export function ScaleBar({ big, small, bigLabel, smallLabel }: {
  big: number; small: number; bigLabel: string; smallLabel: string;
}) {
  const frac = small / big;
  return (
    <div className="scalebar">
      <div className="scalebar-track">
        <div className="scalebar-fill" style={{ width: `${Math.max(frac * 100, 0.14)}%` }} />
      </div>
      <div className="scalebar-legend">
        <span>{smallLabel} is <strong>{(frac * 100).toFixed(2)}%</strong> of {bigLabel}</span>
        <span className="muted">1 / {Math.round(1 / frac).toLocaleString()}</span>
      </div>
    </div>
  );
}
