"""
Supplementary chart for Claude Haiku 4.5 (Section VI-B), matching the
Llama-3.1-70B chart's style for visual parity. Per-dataset numbers from
the actual scored CSVs (results/cloud_baseline_claude_{style}.csv),
correcting the earlier aggregate-only presentation that masked
per-dataset degeneracy.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

datasets = ["bgl", "hdfs", "openstack", "thunderbird"]
f1 = {
    "bgl":         {"zero_shot": 0.704, "rag": 0.938},
    "hdfs":        {"zero_shot": 0.261, "rag": 0.342},
    "openstack":   {"zero_shot": 0.216, "rag": 0.337},
    "thunderbird": {"zero_shot": 0.735, "rag": 0.920},
}
ppr = {
    "bgl":         {"zero_shot": 0.920, "rag": 0.567},
    "hdfs":        {"zero_shot": 0.113, "rag": 0.240},
    "openstack":   {"zero_shot": 0.180, "rag": 0.173},
    "thunderbird": {"zero_shot": 0.860, "rag": 0.587},
}

styles = ["zero_shot", "rag"]
style_labels = {"zero_shot": "Zero-shot", "rag": "RAG"}
colors = {"zero_shot": "#9e9e9e", "rag": "#43a047"}

fig, axes = plt.subplots(2, 1, figsize=(8, 6.4))

ax = axes[0]
x = np.arange(len(datasets))
width = 0.35
for i, style in enumerate(styles):
    vals = [f1[d][style] for d in datasets]
    ax.bar(x + (i - 0.5) * width, vals, width, label=style_labels[style], color=colors[style],
           edgecolor="black", linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(datasets)
ax.set_ylabel("F1"); ax.set_ylim(0, 1.0)
ax.set_title("(a) Claude Haiku 4.5: F1 by dataset and prompt style (single run)")
ax.grid(axis="y", alpha=0.3, linestyle=":")
ax.legend(loc="upper right", fontsize=9)

ax = axes[1]
ax.axhspan(0.85, 1.0, color="#ffcdd2", alpha=0.5, zorder=0)
ax.axhspan(0.0, 0.15, color="#ffcdd2", alpha=0.5, zorder=0)
ax.axhline(0.5, color="black", linestyle="--", linewidth=1, zorder=1)
for i, style in enumerate(styles):
    vals = [ppr[d][style] for d in datasets]
    ax.bar(x + (i - 0.5) * width, vals, width, color=colors[style],
           edgecolor="black", linewidth=0.5, zorder=2)
ax.set_xticks(x); ax.set_xticklabels(datasets)
ax.set_ylabel("Predicted-positive rate"); ax.set_ylim(0, 1.0)
ax.set_title("(b) Predicted-positive rate (dashed = true 50% base rate; red = degenerate)")
ax.grid(axis="y", alpha=0.3, linestyle=":")

plt.tight_layout()
plt.savefig("figures/claude_chart.png", dpi=200, bbox_inches="tight")
print("Saved: figures/claude_chart.png")
