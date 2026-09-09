"""
Generates Figure 3 for the paper: the real benchmark results.

Reads ../benchmark/results/summary_metrics.csv (produced by score_results.py)
and produces a two-panel figure:
  (a) F1 by dataset and prompt style, grouped by model
  (b) Predicted-positive rate by dataset and prompt style, with the
      degenerate-threshold bands shaded -- this is the calibration
      finding, which is the paper's most distinctive contribution and
      deserves its own visual.

Run from the paper/ directory:
    python3 generate_results_figure.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from pathlib import Path

CSV = "../benchmark/results/summary_metrics.csv"
OUT = "figures/results_charts.png"

STYLE_ORDER = ["zero_shot", "few_shot", "rag"]
STYLE_LABEL = {"zero_shot": "Zero-shot", "few_shot": "Few-shot", "rag": "RAG"}
STYLE_COLOR = {"zero_shot": "#9e9e9e", "few_shot": "#42a5f5", "rag": "#43a047"}


def main():
    df = pd.read_csv(CSV)
    datasets = sorted(df["dataset"].unique())
    models = sorted(df["model"].unique())

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # ---- Panel (a): F1 ----
    ax = axes[0]
    x_positions, x_labels = [], []
    pos = 0
    bar_width = 0.26

    for ds in datasets:
        for model in models:
            for i, style in enumerate(STYLE_ORDER):
                row = df[(df["dataset"] == ds) & (df["model"] == model)
                         & (df["prompt_style"] == style)]
                if row.empty:
                    continue
                f1 = row["f1"].values[0]
                lo = row["f1_ci_low"].values[0]
                hi = row["f1_ci_high"].values[0]
                xp = pos + i * bar_width
                ax.bar(xp, f1, bar_width, color=STYLE_COLOR[style],
                       edgecolor="black", linewidth=0.5)
                ax.errorbar(xp, f1, yerr=[[f1 - lo], [hi - f1]],
                            fmt="none", ecolor="black", capsize=2, linewidth=0.8)
            x_positions.append(pos + bar_width)
            short_model = "Qwen" if "qwen" in model else "Mistral"
            x_labels.append(f"{ds}\n{short_model}")
            pos += 1.0
        pos += 0.3   # gap between datasets

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.set_ylabel("F1 (with bootstrap 95% CI)")
    ax.set_title("(a) Anomaly detection F1 by dataset, model, and prompt style")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    handles = [mpatches.Patch(color=STYLE_COLOR[s], label=STYLE_LABEL[s])
               for s in STYLE_ORDER]
    ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=3)

    # ---- Panel (b): calibration (predicted-positive rate) ----
    ax = axes[1]
    pos = 0
    x_positions, x_labels = [], []

    # Shade the degenerate zones
    ax.axhspan(0.85, 1.0, color="#ffcdd2", alpha=0.5, zorder=0)
    ax.axhspan(0.0, 0.15, color="#ffcdd2", alpha=0.5, zorder=0)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, zorder=1)

    for ds in datasets:
        for model in models:
            for i, style in enumerate(STYLE_ORDER):
                row = df[(df["dataset"] == ds) & (df["model"] == model)
                         & (df["prompt_style"] == style)]
                if row.empty:
                    continue
                ppr = row["predicted_positive_rate"].values[0]
                xp = pos + i * bar_width
                ax.bar(xp, ppr, bar_width, color=STYLE_COLOR[style],
                       edgecolor="black", linewidth=0.5, zorder=2)
            x_positions.append(pos + bar_width)
            short_model = "Qwen" if "qwen" in model else "Mistral"
            x_labels.append(f"{ds}\n{short_model}")
            pos += 1.0
        pos += 0.3

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.set_ylabel("Predicted-positive rate")
    ax.set_title("(b) Calibration: fraction of incidents labelled anomalous "
                 "(dashed = true 50% base rate; red = degenerate)")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3, linestyle=":")

    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    plt.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
