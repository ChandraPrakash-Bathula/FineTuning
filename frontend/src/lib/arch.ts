/**
 * Qwen2.5-7B-Instruct architecture, read from the model's own config.json.
 *
 * The LoRA parameter counts derived here reproduce registry.json EXACTLY
 * (r=4 -> 2,523,136 ... r=64 -> 40,370,176), so every matrix drawn in the
 * lesson is the real shape, not an illustration.
 */
export const ARCH = {
  hidden: 3584,
  layers: 28,
  heads: 28,
  kvHeads: 4,
  headDim: 128,
  intermediate: 18944,
  vocab: 152064,
  totalParams: 7_615_616_512,
} as const;

export const KV_DIM = ARCH.kvHeads * ARCH.headDim; // 512

/** The four attention matrices LoRA was applied to, with their real shapes. */
export interface Module {
  name: string;
  /** rows of W == output features */
  out: number;
  /** cols of W == input features */
  in: number;
}

export const TARGET_MODULES: Module[] = [
  { name: "q_proj", out: ARCH.hidden, in: ARCH.hidden },
  { name: "k_proj", out: KV_DIM, in: ARCH.hidden },
  { name: "v_proj", out: KV_DIM, in: ARCH.hidden },
  { name: "o_proj", out: ARCH.hidden, in: ARCH.hidden },
];

/** Parameters in the dense weight matrix W itself. */
export const denseParams = (m: Module) => m.out * m.in;

/** LoRA adds B (out x r) and A (r x in): r*(out + in) per matrix. */
export const loraParams = (m: Module, r: number) => r * (m.out + m.in);

export const perLayerDense = TARGET_MODULES.reduce((s, m) => s + denseParams(m), 0);
export const perLayerLora = (r: number) =>
  TARGET_MODULES.reduce((s, m) => s + loraParams(m, r), 0);

/** Total trainable LoRA parameters across every adapted layer. */
export const totalLora = (r: number) => perLayerLora(r) * ARCH.layers;

/** How many separate weight matrices LoRA touches. */
export const ADAPTED_MATRICES = TARGET_MODULES.length * ARCH.layers; // 112

export const RANKS = [4, 8, 16, 32, 64] as const;
export type Rank = (typeof RANKS)[number];

/** alpha = 2r in every run of this sweep, so the scaling factor is always 2. */
export const alphaFor = (r: number) => 2 * r;
