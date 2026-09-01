"""Where do the eleven variants actually behave differently?

Runs over the full 1,000-question probe set and writes the findings the
Playground's featured cases come from. Nothing here is invented: every case is
read out of the recorded probe files.

The most instructive category is `equivalent`: variants emit DIFFERENT SQL that
is nonetheless all execution-correct. Since execution_correct means "returns the
gold rows", two different correct queries necessarily return the same rows -- so
differing SQL does not imply differing behaviour. That is a real lesson and it
is derived, not assumed.

  python3 scripts/check_disagreement.py
"""
import collections
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBES = os.path.join(ROOT, "frontend", "public", "probes")

LORA_RANKS = (4, 8, 16, 32, 64)
VARIANTS = ["full_ft"] + [f"lora_r{r}" for r in LORA_RANKS] + [f"qlora_r{r}" for r in LORA_RANKS]


def main():
    questions = json.load(open(os.path.join(PROBES, "questions.json")))
    data = {}
    for v in VARIANTS:
        p = os.path.join(PROBES, f"{v}.json")
        if not os.path.exists(p):
            raise SystemExit(f"MISSING ARTIFACT\nRequired: probe file for {v}\n"
                             f"Expected location: {p}")
        data[v] = json.load(open(p))

    n = len(questions)
    cases = []
    counts = collections.Counter()

    for i in range(n):
        sqls = {v: data[v]["predicted_sql"][i].strip() for v in VARIANTS}
        correct = {v: bool(data[v]["execution_correct"][i]) for v in VARIANTS}
        distinct = set(sqls.values())
        n_correct = sum(correct.values())

        if len(distinct) == 1:
            counts["unanimous"] += 1
            continue
        counts["disagree"] += 1

        # different SQL, yet every variant returns the gold rows
        if n_correct == len(VARIANTS):
            kind = "equivalent"
        elif n_correct == 0:
            kind = "all_wrong"
        else:
            kind = "split"
        counts[kind] += 1

        groups = collections.Counter(sqls.values())
        cases.append({
            "id": i,
            "kind": kind,
            "n_distinct": len(distinct),
            "n_correct": n_correct,
            # how evenly the variants split -- larger is a better teaching case
            "balance": min(groups.values()) / max(groups.values()),
        })

    print(f"{n} questions, {len(VARIANTS)} variants\n")
    print(f"  identical SQL from all 11 : {counts['unanimous']:>4}  "
          f"({100 * counts['unanimous'] / n:.1f}%)")
    print(f"  any disagreement          : {counts['disagree']:>4}  "
          f"({100 * counts['disagree'] / n:.1f}%)")
    print(f"      different SQL, all correct : {counts['equivalent']:>4}")
    print(f"      genuinely split            : {counts['split']:>4}")
    print(f"      different SQL, all wrong   : {counts['all_wrong']:>4}")

    def best(kind, k=6):
        pool = [c for c in cases if c["kind"] == kind]
        pool.sort(key=lambda c: (-c["n_distinct"], -c["balance"]))
        return pool[:k]

    featured = {
        "equivalent": best("equivalent"),
        "split": best("split"),
        "all_wrong": best("all_wrong"),
    }

    print("\nfeatured teaching cases")
    for kind, items in featured.items():
        print(f"  {kind}:")
        for c in items[:3]:
            q = questions[c['id']]['question']
            print(f"    Q{c['id']:<4} {c['n_distinct']} distinct SQL, "
                  f"{c['n_correct']}/11 correct — {q[:64]}")

    out = {
        "n_questions": n,
        "variants": VARIANTS,
        "counts": dict(counts),
        "featured": featured,
        "disagreeing_ids": [c["id"] for c in cases],
    }
    dest = os.path.join(PROBES, "disagreement.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"\nwrote {os.path.relpath(dest, ROOT)} "
          f"({os.path.getsize(dest) / 1024:.0f}K)")


if __name__ == "__main__":
    main()
