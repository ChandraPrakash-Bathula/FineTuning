/**
 * Token-level SQL comparison for the side-by-side views.
 *
 * The one rule that matters: SQL keywords and identifiers are case-insensitive,
 * but the contents of a string literal are NOT. In this dataset that distinction
 * is the single most common failure mode -- Q213 has all eleven variants missing
 * the same question purely on literal casing and wording.
 */

/** Quoted spans stay whole so "table" and 'Butler CC (KS)' are single tokens. */
export function tokenize(sql: string): string[] {
  return sql.match(/'(?:[^']|'')*'|"(?:[^"]|"")*"|[A-Za-z_]\w*|\d+(?:\.\d+)?|\S/g) ?? [];
}

export function sameToken(a: string, b: string): boolean {
  const quoted = (t: string) => t.startsWith("'");
  return quoted(a) || quoted(b) ? a === b : a.toUpperCase() === b.toUpperCase();
}

/** True for each token of `pred` that has no counterpart in `gold`. */
export function diffMask(pred: string, gold: string): boolean[] {
  const p = tokenize(pred);
  const g = tokenize(gold);
  const n = p.length;
  const m = g.length;
  // Longest common subsequence; tokens outside it are the differences.
  const dp: number[][] = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i][j] = sameToken(p[i], g[j])
        ? dp[i + 1][j + 1] + 1
        : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  const mask = new Array<boolean>(n).fill(true);
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (sameToken(p[i], g[j])) {
      mask[i] = false;
      i++;
      j++;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) i++;
    else j++;
  }
  return mask;
}

/** No space after "(" or before ")" / "," -- else MIN(col1) reads MIN ( col1 ). */
export function spacer(prev: string, next: string): string {
  if (prev === "(" || next === ")" || next === "," || next === "(") return "";
  return " ";
}


/** Columns a query reads, in order of first appearance: ["col2", "col0"]. */
export function columnsUsed(sql: string): string[] {
  const seen = new Set<string>();
  for (const t of sql.match(/\bcol\d+\b/g) ?? []) seen.add(t);
  return [...seen];
}

/**
 * Which columns the prediction reads that gold does not, and vice versa.
 * Picking the wrong column is the dominant failure mode in this dataset, and
 * "wrong rows" on its own does not tell a student that.
 */
export function columnDelta(pred: string, gold: string) {
  const p = columnsUsed(pred);
  const g = columnsUsed(gold);
  return {
    extra: p.filter((c) => !g.includes(c)),
    missing: g.filter((c) => !p.includes(c)),
  };
}


const AGGREGATES = ["MAX", "MIN", "COUNT", "SUM", "AVG"];

/** The SELECT target of a WikiSQL-shaped query: MAX(col6) -> {agg,col}. */
export function selectTarget(sql: string): { agg: string | null; col: string | null } {
  const m = sql.match(/SELECT\s+(?:(MAX|MIN|COUNT|SUM|AVG)\s*\(\s*)?(col\d+)/i);
  if (!m) return { agg: null, col: null };
  return { agg: m[1] ? m[1].toUpperCase() : null, col: m[2].toLowerCase() };
}

/**
 * Why two textually different queries can return identical rows.
 *
 * These are the two patterns this dataset actually produces; anything else gets
 * the generic explanation rather than a guess.
 */
export function equivalenceReason(sqls: string[]): string | null {
  if (sqls.length < 2) return null;
  const targets = sqls.map(selectTarget);
  if (targets.some((t) => !t.col)) return null;

  const cols = new Set(targets.map((t) => t.col));
  const aggs = new Set(targets.map((t) => t.agg));

  // COUNT over different columns: counting non-null values of any column of the
  // same filtered rows gives the same number.
  if (aggs.size === 1 && [...aggs][0] === "COUNT" && cols.size > 1) {
    return "Every version is a COUNT over the same filtered rows, just counting a "
      + "different column. COUNT returns how many rows there are, so the column "
      + "chosen does not change the answer.";
  }

  // aggregate vs bare column, same column: identical when the filter selects one row.
  if (cols.size === 1 && aggs.size > 1 && aggs.has(null)) {
    const agg = [...aggs].find((a) => a && AGGREGATES.includes(a));
    return `One version wraps ${[...cols][0]} in ${agg}() and the other reads it `
      + `directly. The WHERE clause here matches a single row, and ${agg}() of one `
      + "value is that value — so both return the same cell.";
  }

  return null;
}
