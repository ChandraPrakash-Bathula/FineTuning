/**
 * Rental cost. Phase 1 actually ran on an H200; the H100 rate below is the
 * one students are most likely to be quoted, and is shown as an assumption
 * rather than a measurement.
 */
export const GPU_RATES = [
  { id: "h100", label: "H100 80GB", usdPerHour: 2.99, memoryGb: 80 },
  { id: "a100-80", label: "A100 80GB", usdPerHour: 1.79, memoryGb: 80 },
  { id: "a100-40", label: "A100 40GB", usdPerHour: 1.29, memoryGb: 40 },
  { id: "l40s", label: "L40S 48GB", usdPerHour: 1.09, memoryGb: 48 },
] as const;

export type GpuId = (typeof GPU_RATES)[number]["id"];

export const gpu = (id: GpuId) => GPU_RATES.find((g) => g.id === id)!;

export const costUsd = (minutes: number, id: GpuId) =>
  (minutes / 60) * gpu(id).usdPerHour;

/** Does this run physically fit on that card? */
export const fitsOn = (peakMemoryGb: number, id: GpuId) => peakMemoryGb <= gpu(id).memoryGb;

export const money = (usd: number) =>
  usd < 10 ? `$${usd.toFixed(2)}` : `$${usd.toFixed(0)}`;
