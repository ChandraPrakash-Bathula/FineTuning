"""Turn the probe exports into the compact form the web app loads.

Two recorded probe runs exist, and they are NOT interchangeable:

  regen.log    all 11 variants generated against a bf16 base
  regen_q.log  the 5 QLoRA variants re-generated against a 4-bit NF4 base

The registry's own Phase 1 evaluation used the 4-bit base for QLoRA -- that
serving mode reproduces it 7.5x more closely than bf16 does. So the canonical
set is: full_ft + lora_* from the standard export, qlora_* from the _4bit one.

The bf16-served QLoRA run is kept as `serving_bf16`, because the gap between
the two is a real measured result worth teaching, not a discrepancy to hide.

Every prediction file repeats the same question/schema/gold_sql, so those are
split into one shared questions file and the per-variant files keep only what
actually varies.

  python3 scripts/build_probes.py <dir-with-standard-export> <dir-with-4bit-export>
"""
import json
import os
import sys

import registry_from_probes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "frontend", "public", "probes")

LORA_RANKS = (4, 8, 16, 32, 64)
CANONICAL = ["full_ft"] + [f"lora_r{r}" for r in LORA_RANKS] + [f"qlora_r{r}" for r in LORA_RANKS]


def load(path):
    with open(path) as fh:
        return json.load(fh)


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: build_probes.py <export-dir> [<4bit-export-dir>]")
    std_dir = sys.argv[1]
    q_dir = sys.argv[2] if len(sys.argv) > 2 else std_dir

    registry = load(os.path.join(ROOT, "data", "registry.json"))
    reg_acc = {v["variant_id"]: v["evaluation"]["sql_execution_accuracy"]
               for v in registry["variants"]}

    # ---- pick the source file for each canonical variant -------------------
    sources = {}
    for vid in CANONICAL:
        if vid.startswith("qlora"):
            cand = os.path.join(q_dir, f"{vid}_4bit.json")
            if os.path.exists(cand):
                sources[vid] = (cand, "4bit")
                continue
        cand = os.path.join(std_dir, f"{vid}.json")
        if not os.path.exists(cand):
            sys.exit(f"MISSING ARTIFACT\nThe application requires probe predictions for:\n"
                     f"  {vid}\nExpected location:\n  {cand}\n"
                     f"Do not substitute an invented value.")
        sources[vid] = (cand, "bf16")

    os.makedirs(OUT, exist_ok=True)

    # ---- shared questions --------------------------------------------------
    first = load(sources["full_ft"][0])["predictions"]
    questions = [{"id": i, "question": p["question"], "schema": p["schema"],
                  "gold_sql": p["gold_sql"]} for i, p in enumerate(first)]

    manifest = {"n_questions": len(questions), "variants": {}, "serving_bf16": {}}

    def emit(vid, path, key, base):
        data = load(path)
        preds = data["predictions"]
        if len(preds) != len(questions):
            sys.exit(f"{vid}: {len(preds)} predictions but {len(questions)} questions")
        for i, p in enumerate(preds):
            if p["question"] != questions[i]["question"]:
                sys.exit(f"{vid}: question {i} does not match the shared list")
        out = {
            "variant_id": vid,
            "serving_base": base,
            "metrics": data["metrics"],
            # index-aligned with questions.json
            "predicted_sql": [p["predicted_sql"] for p in preds],
            "exact_match": [int(p["exact_match"]) for p in preds],
            "execution_correct": [int(p["execution_correct"]) for p in preds],
            "invalid_sql": [int(p["invalid_sql"]) for p in preds],
        }
        fn = f"{key}.json"
        with open(os.path.join(OUT, fn), "w") as fh:
            json.dump(out, fh, separators=(",", ":"))
        return out, os.path.getsize(os.path.join(OUT, fn))

    print(f"{'variant':<12}{'base':>7}{'probe':>9}{'registry':>10}{'delta':>9}{'size':>10}")
    for vid in CANONICAL:
        path, base = sources[vid]
        out, nbytes = emit(vid, path, vid, base)
        probe = out["metrics"]["sql_execution_accuracy"]
        reg = reg_acc[vid]
        manifest["variants"][vid] = {"serving_base": base, "metrics": out["metrics"]}
        print(f"{vid:<12}{base:>7}{probe:>9.3f}{reg:>10.3f}{probe - reg:>+9.3f}{nbytes / 1024:>9.0f}K")

    # ---- the bf16-served QLoRA comparison run ------------------------------
    print("\nQLoRA served on a bf16 base (kept for the serving-mode comparison):")
    for r in LORA_RANKS:
        vid = f"qlora_r{r}"
        path = os.path.join(std_dir, f"{vid}.json")
        if not os.path.exists(path):
            print(f"  {vid}: not present, skipped")
            continue
        out, nbytes = emit(vid, path, f"{vid}__bf16", "bf16")
        manifest["serving_bf16"][vid] = {"metrics": out["metrics"]}
        print(f"  {vid:<12}{out['metrics']['sql_execution_accuracy']:.3f}  ({nbytes / 1024:.0f}K)")

    with open(os.path.join(OUT, "questions.json"), "w") as fh:
        json.dump(questions, fh, separators=(",", ":"))
    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    qsize = os.path.getsize(os.path.join(OUT, "questions.json")) / 1024
    print(f"\nquestions.json {qsize:.0f}K ({len(questions)} questions)")
    print(f"wrote {OUT}")

    # ---- frontend copy of the registry ------------------------------------
    # Headline SQL metrics are rewritten from the manifest just written, so
    # the charts and the paired tests read the same generation pass.
    print()
    registry_from_probes.build()


if __name__ == "__main__":
    main()
