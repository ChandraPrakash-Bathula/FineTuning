"""Paired significance tests between every pair of variants.

Every variant answered the SAME 1,000 held-out questions, so two variants'
accuracies are not independent samples: a question that is easy for one is
usually easy for the other. The right comparison only looks at the questions
where the two variants DISAGREE:

  b = questions A got right and B got wrong
  c = questions B got right and A got wrong

Questions both got right (or both got wrong) say nothing about which is better.

Two paired methods, reported side by side:

  McNemar's exact test   under "no difference", each disagreement is a fair
                         coin flip, so b ~ Binomial(b + c, 0.5). Two-sided p.
  Paired bootstrap       resample the 1,000 questions WITH their pairing kept,
                         recompute acc(A) - acc(B) each time, and read off the
                         95% percentile interval of the difference.

Because 55 pairs are tested at once, a Holm-Bonferroni correction is also
reported: with 55 tests at alpha = 0.05, about 2.75 "significant" results are
expected from noise alone.

The unpaired two-proportion z-test is computed too, only so the lesson can
show what the paired analysis changes.

Inputs are the recorded probe runs in frontend/public/probes (per-question
execution_correct). Those are re-generated runs and differ from the registry's
own evaluation by a few tenths of a point; the registry has only totals, so a
paired test cannot be run on it.

  python3 scripts/paired_stats.py
writes frontend/public/paired_stats.json
"""
import itertools
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBES = os.path.join(ROOT, "frontend", "public", "probes")
OUT = os.path.join(ROOT, "frontend", "public", "paired_stats.json")

RANKS = (4, 8, 16, 32, 64)
VARIANTS = ["full_ft"] + [f"lora_r{r}" for r in RANKS] + [f"qlora_r{r}" for r in RANKS]
N_BOOT = 10_000
SEED = 42


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value: binomial test of b against Bin(b+c, 1/2)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def two_proportion_z(x1, x2, n):
    """The unpaired test the first version used. Kept for contrast only."""
    p1, p2 = x1 / n, x2 / n
    pooled = (x1 + x2) / (2 * n)
    se = math.sqrt(pooled * (1 - pooled) * (2 / n))
    z = (p1 - p2) / se if se else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))
    return z, p


def paired_bootstrap(a, b, rng):
    """95% percentile CI of acc(a) - acc(b), resampling questions with pairing kept."""
    diff = np.asarray(a) - np.asarray(b)  # each in {-1, 0, 1}
    idx = rng.integers(0, len(diff), size=(N_BOOT, len(diff)))
    lo, hi = np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def grouped(label_a, label_b, group_a, group_b, rng):
    """Group A's mean accuracy vs group B's, paired by question.

    d_i = mean over A's models of correct_i - mean over B's models of correct_i.
    p-value: sign-flip permutation test (under "no difference", each question's
    d_i is equally likely to be +d_i or -d_i). Interval: paired bootstrap of mean d.
    """
    a = np.mean(np.asarray(group_a), axis=0)
    b = np.mean(np.asarray(group_b), axis=0)
    d = a - b
    observed = d.mean()
    signs = rng.choice([-1.0, 1.0], size=(N_BOOT, len(d)))
    null = (signs * d).mean(axis=1)
    p = (np.sum(np.abs(null) >= abs(observed) - 1e-12) + 1) / (N_BOOT + 1)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    return {"a": label_a, "b": label_b, "diff": float(observed), "p": float(p),
            "bootstrap_ci95": [float(lo), float(hi)], "test": "sign-flip permutation",
            "resamples": N_BOOT}


def chi2_sf(x, df):
    """Upper tail of chi-squared, via the series for the lower regularised gamma."""
    a, h = df / 2, x / 2
    if h <= 0:
        return 1.0
    term = total = 1.0 / a
    k = 0
    while term > 1e-15 * total:
        k += 1
        term *= h / (a + k)
        total += term
    lower = math.exp(a * math.log(h) - h - math.lgamma(a)) * total
    return max(0.0, 1.0 - lower)


def cochran_q(label, columns):
    """Omnibus test: do any of these k models differ, on the same questions?

    One test instead of k(k-1)/2 pairs, so no multiple-comparison correction is
    needed. Q ~ chi-squared with k-1 df under "all k have the same accuracy".
    """
    x = np.asarray(columns).T  # questions x models
    k = x.shape[1]
    col, row, total = x.sum(axis=0), x.sum(axis=1), x.sum()
    q = (k - 1) * (k * (col ** 2).sum() - total ** 2) / (k * total - (row ** 2).sum())
    return {"models": label, "k": k, "q": float(q), "df": k - 1,
            "p": chi2_sf(float(q), k - 1), "test": "Cochran's Q"}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, same order as input."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def main():
    correct = {}
    for v in VARIANTS:
        p = os.path.join(PROBES, f"{v}.json")
        if not os.path.exists(p):
            raise SystemExit(f"MISSING ARTIFACT\nRequired: probe file for {v}\n"
                             f"Expected location: {p}")
        correct[v] = [int(bool(x)) for x in json.load(open(p))["execution_correct"]]
    n = len(correct[VARIANTS[0]])

    rng = np.random.default_rng(SEED)

    def compare(a, b, ca, cb):

        both = sum(1 for x, y in zip(ca, cb) if x and y)
        neither = sum(1 for x, y in zip(ca, cb) if not x and not y)
        a_only = sum(1 for x, y in zip(ca, cb) if x and not y)
        b_only = sum(1 for x, y in zip(ca, cb) if y and not x)
        z, pz = two_proportion_z(sum(ca), sum(cb), n)
        lo, hi = paired_bootstrap(ca, cb, rng)
        return {
            "a": a, "b": b,
            "acc_a": sum(ca) / n, "acc_b": sum(cb) / n,
            "diff": (sum(ca) - sum(cb)) / n,
            "both_correct": both, "neither_correct": neither,
            "a_only": a_only, "b_only": b_only,
            "mcnemar_p": mcnemar_exact(a_only, b_only),
            "bootstrap_ci95": [lo, hi],
            "unpaired_z": z, "unpaired_p": pz,
        }

    pairs = [compare(a, b, correct[a], correct[b])
             for a, b in itertools.combinations(VARIANTS, 2)]

    for p, adj in zip(pairs, holm([p["mcnemar_p"] for p in pairs])):
        p["mcnemar_p_holm"] = adj

    # Same adapter served on the 4-bit base it was trained on vs a bf16 base.
    # A separate, pre-specified family of five tests, so it gets its own Holm.
    serving = []
    for r in RANKS:
        v = f"qlora_r{r}"
        bf = [int(bool(x)) for x in
              json.load(open(os.path.join(PROBES, f"{v}__bf16.json")))["execution_correct"]]
        serving.append(compare(v, f"{v}__bf16", correct[v], bf))
    for p, adj in zip(serving, holm([p["mcnemar_p"] for p in serving])):
        p["mcnemar_p_holm"] = adj
    bf16 = {r: [int(bool(x)) for x in json.load(open(os.path.join(
        PROBES, f"qlora_r{r}__bf16.json")))["execution_correct"]] for r in RANKS}

    # Two questions asked across a whole group of models rather than one pair.
    # The question is the unit: average each group's right/wrong per question,
    # then test the per-question difference, so no question is counted twice.
    adapters = [v for v in VARIANTS if v != "full_ft"]
    groups = {
        "adapters_vs_full_ft": grouped(
            "mean of the 10 adapters", "full_ft",
            [correct[v] for v in adapters], [correct["full_ft"]], rng),
        "serving_4bit_vs_bf16": grouped(
            "QLoRA adapters on the 4-bit base", "same adapters on a bf16 base",
            [correct[f"qlora_r{r}"] for r in RANKS], [bf16[r] for r in RANKS], rng),
    }

    lora = [f"lora_r{r}" for r in RANKS]
    qlora = [f"qlora_r{r}" for r in RANKS]
    omnibus = {
        "all_eleven": cochran_q("all 11 variants", [correct[v] for v in VARIANTS]),
        "lora_ranks": cochran_q("LoRA r=4..64", [correct[v] for v in lora]),
        "qlora_ranks": cochran_q("QLoRA r=4..64", [correct[v] for v in qlora]),
    }

    out = {
        "source": "frontend/public/probes (recorded probe runs, per-question execution_correct)",
        "n_questions": n,
        "n_pairs": len(pairs),
        "bootstrap_resamples": N_BOOT,
        "bootstrap_seed": SEED,
        "alpha": 0.05,
        "accuracy": {v: sum(correct[v]) / n for v in VARIANTS},
        "pairs": pairs,
        "serving": serving,
        "groups": groups,
        "omnibus": omnibus,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)

    sig = [p for p in pairs if p["mcnemar_p"] < 0.05]
    sig_holm = [p for p in pairs if p["mcnemar_p_holm"] < 0.05]
    print(f"wrote {OUT}")
    print(f"{len(pairs)} pairs; McNemar p<0.05: {len(sig)}; after Holm: {len(sig_holm)}")
    for p in sorted(pairs, key=lambda p: p["mcnemar_p"])[:12]:
        print(f"  {p['a']:>9} vs {p['b']:<9} diff {p['diff']*100:+5.1f}  "
              f"b={p['a_only']:3d} c={p['b_only']:3d}  McNemar p={p['mcnemar_p']:.4f}  "
              f"Holm p={p['mcnemar_p_holm']:.3f}  "
              f"CI [{p['bootstrap_ci95'][0]*100:+.1f}, {p['bootstrap_ci95'][1]*100:+.1f}]  "
              f"unpaired z={p['unpaired_z']:+.2f} p={p['unpaired_p']:.3f}")
    print("serving base (4-bit vs bf16):")
    for p in serving:
        print(f"  {p['a']:>9} diff {p['diff']*100:+5.1f}  b={p['a_only']} c={p['b_only']}  "
              f"McNemar p={p['mcnemar_p']:.4f}  Holm(5) p={p['mcnemar_p_holm']:.3f}")
    for k, g in groups.items():
        print(f"{k}: diff {g['diff']*100:+.2f} pts  sign-flip p={g['p']:.4f}  "
              f"CI [{g['bootstrap_ci95'][0]*100:+.2f}, {g['bootstrap_ci95'][1]*100:+.2f}]")
    for k, o in omnibus.items():
        print(f"{k}: Cochran's Q = {o['q']:.2f}, df {o['df']}, p = {o['p']:.4f}")


if __name__ == "__main__":
    main()
