"""Write the frontend copy of the registry, with SQL metrics taken from the probes.

The registry's SQL metrics come from one generation pass run in-process right
after training. The probe runs in frontend/public/probes are a second pass from
the checkpoints reloaded from disk, and they are the only source with a
right/wrong result per question -- which the paired tests need. Showing
headline numbers from one pass and p-values from the other leaves a visible
seam of up to 0.4 points, so the app uses the probes for both.

For each variant, the six SQL generation metrics in `evaluation` are replaced
by the probe manifest's. Training-side metrics the probes do not have
(eval_loss, perplexity, token_accuracy) are kept. The original block is kept
whole as `evaluation_at_training`, so the provenance stays inspectable.

data/registry.json is the raw artifact (scripts/fetch_artifacts.py re-downloads
it) and is never modified. The 25-question generation_probes blocks are also
dropped here: the 1,000-question probe files supersede them.

  python3 scripts/registry_from_probes.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "registry.json")
MANIFEST = os.path.join(ROOT, "frontend", "public", "probes", "manifest.json")
DEST = os.path.join(ROOT, "frontend", "public", "registry.json")

SQL_METRICS = ("num_examples", "num_exact_match", "num_execution_correct",
               "num_invalid_sql", "sql_exact_match", "sql_execution_accuracy")


def build():
    registry = json.load(open(SRC))
    probes = json.load(open(MANIFEST))["variants"]

    web = {k: v for k, v in registry.items() if k != "variants"}
    web["evaluation_source"] = (
        "SQL metrics in `evaluation` are from the probe runs in probes/ (checkpoints "
        "reloaded from disk; per-question results). `evaluation_at_training` holds the "
        "original in-process evaluation.")
    web["variants"] = []
    print(f"{'variant':<12}{'at training':>12}{'probes':>9}{'delta':>8}")
    for var in registry["variants"]:
        vid = var["variant_id"]
        out = {k: v for k, v in var.items() if k != "generation_probes"}
        if vid in probes:
            m = probes[vid]["metrics"]
            out["evaluation_at_training"] = var["evaluation"]
            out["evaluation"] = {**var["evaluation"], **{k: m[k] for k in SQL_METRICS}}
            a = var["evaluation"]["sql_execution_accuracy"]
            b = m["sql_execution_accuracy"]
            print(f"{vid:<12}{a:>12.3f}{b:>9.3f}{b - a:>+8.3f}")
        else:
            print(f"{vid:<12}  no probe run -- registry evaluation kept")
        web["variants"].append(out)

    with open(DEST, "w") as fh:
        json.dump(web, fh, separators=(",", ":"))
    print(f"wrote {DEST}")


if __name__ == "__main__":
    build()
