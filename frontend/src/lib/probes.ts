/**
 * The recorded probe runs: 1,000 held-out WikiSQL questions answered by every
 * variant, generated in Phase 1 and exported to web/public/probes.
 *
 * These are RECORDED OUTPUTS, not live inference. The UI must always say which
 * of the two it is showing.
 *
 * Files are split so the browser fetches one 293K question list plus 64K per
 * variant actually selected, instead of a single multi-megabyte blob.
 */

export interface Question {
  id: number;
  question: string;
  /** "col0: Player", "col1: No.", ... */
  schema: string[];
  gold_sql: string;
}

/** Per-variant arrays, index-aligned with the question list. */
export interface VariantProbes {
  variant_id: string;
  /** "4bit" for QLoRA (as the registry evaluated it), "bf16" otherwise. */
  serving_base: "bf16" | "4bit";
  metrics: {
    num_examples: number;
    num_exact_match: number;
    num_execution_correct: number;
    num_invalid_sql: number;
    sql_exact_match: number;
    sql_execution_accuracy: number;
  };
  predicted_sql: string[];
  exact_match: number[];
  execution_correct: number[];
  invalid_sql: number[];
}

export interface FeaturedCase {
  id: number;
  kind: "equivalent" | "split" | "all_wrong";
  n_distinct: number;
  n_correct: number;
  balance: number;
}

export interface Disagreement {
  n_questions: number;
  variants: string[];
  counts: {
    unanimous: number;
    disagree: number;
    equivalent: number;
    split: number;
    all_wrong: number;
  };
  featured: Record<FeaturedCase["kind"], FeaturedCase[]>;
  disagreeing_ids: number[];
}

const base = () => `${import.meta.env.BASE_URL}probes/`;

async function getJson<T>(file: string): Promise<T> {
  const res = await fetch(base() + file);
  if (!res.ok) throw new Error(`probes/${file}: HTTP ${res.status}`);
  return (await res.json()) as T;
}

let questionsPromise: Promise<Question[]> | null = null;
export const loadQuestions = () =>
  (questionsPromise ??= getJson<Question[]>("questions.json"));

let disagreementPromise: Promise<Disagreement> | null = null;
export const loadDisagreement = () =>
  (disagreementPromise ??= getJson<Disagreement>("disagreement.json"));

const variantCache = new Map<string, Promise<VariantProbes>>();

/** Lazily fetch one variant's recorded predictions; cached per session. */
export function loadVariantProbes(variantId: string): Promise<VariantProbes> {
  let p = variantCache.get(variantId);
  if (!p) {
    p = getJson<VariantProbes>(`${variantId}.json`);
    variantCache.set(variantId, p);
  }
  return p;
}

/**
 * QLoRA adapters re-run against a bf16 base instead of the 4-bit base they were
 * trained on. Kept because the difference is a real measured result.
 */
export const loadServingBf16 = (variantId: string) =>
  loadVariantProbes(`${variantId}__bf16`);

/** Group variants by the SQL they produced for one question. */
export interface AnswerGroup {
  sql: string;
  variantIds: string[];
  executionCorrect: boolean;
  exactMatch: boolean;
  invalidSql: boolean;
}

export function groupAnswers(
  questionId: number,
  probes: { id: string; data: VariantProbes }[],
): AnswerGroup[] {
  const map = new Map<string, AnswerGroup>();
  for (const { id, data } of probes) {
    const sql = data.predicted_sql[questionId]?.trim() ?? "";
    const g = map.get(sql);
    if (g) {
      g.variantIds.push(id);
    } else {
      map.set(sql, {
        sql,
        variantIds: [id],
        executionCorrect: !!data.execution_correct[questionId],
        exactMatch: !!data.exact_match[questionId],
        invalidSql: !!data.invalid_sql[questionId],
      });
    }
  }
  return [...map.values()].sort((a, b) => b.variantIds.length - a.variantIds.length);
}
