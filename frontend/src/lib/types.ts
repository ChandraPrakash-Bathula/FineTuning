export type Method = "full_ft" | "lora" | "qlora";

export interface RawVariant {
  variant_id: string;
  method: Method;
  rank: number | null;
  status: string;
  config: Record<string, unknown> & {
    lora_alpha?: number;
    lora_dropout?: number;
    target_modules?: string[];
    precision?: string;
    quantization?: { bits: number; quant_type: string; double_quant: boolean; compute_dtype: string } | null;
    learning_rate?: number;
    epochs?: number;
    effective_batch_size?: number;
    sql_eval_samples?: number;
  };
  parameters: { total: number; trainable: number; trainable_percentage: number };
  resources: {
    peak_memory_allocated_bytes: number;
    peak_memory_reserved_bytes: number;
    training_time_seconds: number;
    checkpoint_size_bytes: number;
  };
  evaluation: {
    eval_loss: number;
    perplexity: number;
    token_accuracy: number;
    num_examples: number;
    num_exact_match: number;
    num_execution_correct: number;
    num_invalid_sql: number;
    sql_exact_match: number;
    sql_execution_accuracy: number;
  };
  environment: Record<string, unknown>;
  training_curve: {
    train: { step: number; loss: number }[];
    eval: { step: number; eval_loss: number; eval_token_accuracy: number }[];
  };
}

export interface Registry {
  base_model: string;
  dataset: string;
  n_variants: number;
  variants: RawVariant[];
  sweep_audit: Record<string, unknown>;
}

/** A variant flattened into the shape every chart and table consumes. */
export interface Variant {
  id: string;
  method: Method;
  rank: number | null;
  label: string;
  alpha: number | null;
  scaling: number | null;
  targetModules: string[] | null;
  precision: string;
  quantization: string | null;
  learningRate: number | null;
  epochs: number | null;
  trainable: number;
  total: number;
  trainablePct: number;
  peakMemoryGb: number;
  checkpointMb: number;
  trainingMinutes: number;
  execAccuracy: number;
  exactMatch: number;
  evalLoss: number;
  tokenAccuracy: number;
  invalidSql: number;
  evalN: number;
  trainCurve: { step: number; loss: number }[];
  evalCurve: { step: number; loss: number }[];
}
