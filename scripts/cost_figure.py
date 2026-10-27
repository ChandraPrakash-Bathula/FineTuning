"""Execution accuracy against checkpoint size per task, for all eleven variants.

Accuracy is the probe value (paired_stats.json), the same pass the paired tests
use; checkpoint size is the registry's. Replaces the earlier figure, whose
shaded "noise band" implied an unpaired comparison.

  python3 scripts/paired_stats.py      # first
  python3 scripts/cost_figure.py [out.png]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "figures", "accuracy_vs_checkpoint.png")

COLOR = {"full_ft": "#b14434", "lora": "#728f74", "qlora": "#bd803f"}


def main():
    stats = json.load(open(os.path.join(ROOT, "frontend", "public", "paired_stats.json")))
    acc = stats["accuracy"]
    group = stats["groups"]["adapters_vs_full_ft"]
    size = {v["variant_id"]: v["resources"]["checkpoint_size_bytes"] / 1e9
            for v in json.load(open(os.path.join(ROOT, "data", "registry.json")))["variants"]}

    fig, ax = plt.subplots(figsize=(21.79 / 1.2, 9.8 / 1.2), dpi=120)
    for v in acc:
        c = COLOR["full_ft"] if v == "full_ft" else COLOR[v.split("_")[0]]
        ax.scatter(size[v], acc[v] * 100, s=260, color=c, edgecolor="white", linewidth=1.5, zorder=3)

    ft, r4 = "full_ft", "lora_r4"
    ax.annotate(f"full fine-tuning\n{size[ft]:.2f} GB per task, {acc[ft] * 100:.1f}%",
                xy=(size[ft], acc[ft] * 100), xytext=(3.0, 74.0), fontsize=15,
                color=COLOR["full_ft"], arrowprops=dict(arrowstyle="->", color=COLOR["full_ft"], lw=1.8))
    ax.annotate(f"LoRA / QLoRA r=4\n{size[r4]:.3f} GB per task",
                xy=(size[r4], acc[r4] * 100), xytext=(0.025, 80.3), fontsize=15,
                color=COLOR["lora"], arrowprops=dict(arrowstyle="->", color=COLOR["lora"], lw=1.8))
    ax.scatter([], [], s=120, color=COLOR["lora"], label="LoRA")
    ax.scatter([], [], s=120, color=COLOR["qlora"], label="QLoRA")
    ax.legend(loc="upper right", fontsize=14, frameon=False)

    ax.set_xscale("log")
    ax.set_ylim(72, 82)
    ax.set_xlabel("checkpoint size per task, GB (log scale)", fontsize=16)
    ax.set_ylabel("SQL execution accuracy (%)", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.grid(color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ratio = size[ft] / size[r4]
    ax.set_title(f"{ratio:.0f}× more storage, {group['diff'] * 100:.1f} points lower accuracy "
                 f"(paired p = {group['p']:.3f})", loc="left", fontsize=21, fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
