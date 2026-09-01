"""WikiSQL label rendering + sandboxed execution.

Deliberately identical to Phase 1's logic so the app's verdicts match the
scores in registry.json. Do not "improve" the normalisation here without
re-scoring Phase 1 — divergence makes the lab lie to students.
"""
import sqlite3

AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
COND_OPS = ["=", ">", "<", "OP"]


def render_sql(sql):
    """WikiSQL's structured label -> canonical SQL text."""
    sel, agg = f"col{sql['sel']}", AGG_OPS[sql["agg"]]
    out = f'SELECT {f"{agg}({sel})" if agg else sel} FROM "table"'
    conds = sql.get("conds") or {}
    if isinstance(conds, dict):
        triples = list(zip(conds.get("column_index", []),
                           conds.get("operator_index", []),
                           conds.get("condition", [])))
    else:
        triples = [tuple(c) for c in conds]
    if triples:
        out += " WHERE " + " AND ".join(
            f"col{c} {COND_OPS[o]} '" + str(v).strip().replace("'", "''") + "'"
            for c, o, v in triples)
    return out


def execute(table, query):
    """Build the table in memory, run the query. Never raises."""
    try:
        types = table.get("types") or ["text"] * len(table["header"])
        cols = ", ".join(f'col{i} {"REAL" if t == "real" else "TEXT"}'
                         for i, t in enumerate(types))
        conn = sqlite3.connect(":memory:")
        cur = conn.cursor()
        cur.execute(f'CREATE TABLE "table" ({cols})')
        rows = []
        for row in table.get("rows", []):
            vals = []
            for v, t in zip(row, types):
                if t == "real":
                    try:
                        vals.append(float(str(v).replace(",", "")))
                    except (ValueError, TypeError):
                        vals.append(None)
                else:
                    vals.append(str(v))
            rows.append(vals)
        cur.executemany(
            f'INSERT INTO "table" VALUES ({",".join("?" * len(types))})', rows)
        cur.execute(query)
        result = cur.fetchall()
        conn.close()
        return {"rows": result, "error": None}
    except Exception as exc:
        return {"rows": None, "error": f"{type(exc).__name__}: {exc}"}


def compare(table, predicted_sql, gold_sql):
    """Score one prediction the way Phase 1 did: exact match + execution match."""
    pred = execute(table, predicted_sql)
    gold = execute(table, gold_sql) if gold_sql else {"rows": None, "error": None}
    invalid = pred["error"] is not None
    return {
        "predicted_sql": predicted_sql,
        "gold_sql": gold_sql,
        "result_rows": pred["rows"],
        "gold_rows": gold["rows"],
        "exact_match": bool(gold_sql) and predicted_sql.strip() == gold_sql.strip(),
        "execution_correct": (not invalid and gold_sql is not None
                              and pred["rows"] == gold["rows"]),
        "invalid_sql": invalid,
        "error": pred["error"],
    }
