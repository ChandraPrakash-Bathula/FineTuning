"""Execution accuracy for all eleven variants, annotated with the paired test
against full fine-tuning. Replaces the earlier "noise band" figure, which
implied an unpaired comparison.

  python3 scripts/paired_stats.py      # first
  python3 scripts/paired_figure.py [out.png]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "figures", "paired_accuracy.png")

RANKS = (4, 8, 16, 32, 64)
VARIANTS = ["full_ft"] + [f"lora_r{r}" for r in RANKS] + [f"qlora_r{r}" for r in RANKS]
COLOR = {"full_ft": "#b14434", "lora": "#728f74", "qlora": "#bd803f"}


def main():
    # Bar heights and p-values both come from the probe runs, so they agree.
    stats = json.load(open(os.path.join(ROOT, "frontend", "public", "paired_stats.json")))
    reg = stats["accuracy"]
    pairs = {frozenset((p["a"], p["b"])): p for p in stats["pairs"]}
    group = stats["groups"]["adapters_vs_full_ft"]

    acc = [reg[v] * 100 for v in VARIANTS]
    colors = [COLOR["full_ft"] if v == "full_ft" else COLOR[v.split("_")[0]] for v in VARIANTS]

    fig, ax = plt.subplots(figsize=(23.79 / 1.2, 9.8 / 1.2), dpi=120)
    bars = ax.bar(range(len(VARIANTS)), acc, width=0.62, color=colors)
    ax.axhline(acc[0], color=COLOR["full_ft"], linestyle="--", linewidth=1.4, alpha=0.8)

    for i, (b, v) in enumerate(zip(bars, VARIANTS)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.12, f"{acc[i]:.1f}",
                ha="center", va="bottom", fontsize=15)
        if v == "full_ft":
            ax.text(b.get_x() + b.get_width() / 2, 70.35, "baseline", ha="center",
                    fontsize=13, color="white", fontweight="bold")
            continue
        p = pairs[frozenset((v, "full_ft"))]["mcnemar_p"]
        ax.text(b.get_x() + b.get_width() / 2, 70.35, f"p={p:.3f}" if p < 0.01 else f"p={p:.2f}",
                ha="center", fontsize=13, color="white",
                fontweight="bold" if p < 0.05 else "normal")

    ax.set_xticks(range(len(VARIANTS)))
    ax.set_xticklabels(VARIANTS, rotation=35, ha="right", fontsize=14)
    ax.set_ylim(70, 80.5)
    ax.set_ylabel("SQL execution accuracy (%)", fontsize=16)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Every adapter scored above full fine-tuning on the same 1,000 questions",
                 loc="left", fontsize=21, fontweight="bold")
    ax.text(0.995, 0.97,
            f"p on each bar: McNemar exact test vs full_ft (paired)\n"
            f"mean adapter − full_ft: +{group['diff']*100:.1f} pts, paired p = {group['p']:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=14,
            color="#5a5a7a", style="italic")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
