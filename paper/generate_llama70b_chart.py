"""
Supplementary chart for Llama-3.1-70B (Section V-E), matching the visual
style of Fig. 3 (results_charts.png) so the generalization check gets the
same visual treatment as the primary Qwen/Mistral results -- two panels:
F1 by dataset/style, and predicted-positive rate with the degenerate
threshold shading.

Data is the single-run, per-dataset numbers already reported and scored
in this session (results/cloud_baseline_ollama_{zero_shot,rag}.csv via
score_calibration-style aggregation) -- not re-fabricated here.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

datasets = ["bgl", "hdfs", "openstack", "thunderbird"]
# zero-shot, rag  (F1 and PPR), per dataset -- from the actual scored run
f1 = {
    "bgl":         {"zero_shot": 0.708, "rag": 0.909},
    "hdfs":        {"zero_shot": 0.463, "rag": 0.568},
    "openstack":   {"zero_shot": 0.222, "rag": 0.618},
    "thunderbird": {"zero_shot": 0.758, "rag": 0.847},
}
ppr = {
    "bgl":         {"zero_shot": 0.913, "rag": 0.600},
    "hdfs":        {"zero_shot": 0.393, "rag": 0.533},
    "openstack":   {"zero_shot": 0.160, "rag": 0.407},
    "thunderbird": {"zero_shot": 0.820, "rag": 0.680},
}

styles = ["zero_shot", "rag"]
style_labels = {"zero_shot": "Zero-shot", "rag": "RAG"}
colors = {"zero_shot": "#9e9e9e", "rag": "#43a047"}

fig, axes = plt.subplots(2, 1, figsize=(8, 6.4))

# Panel (a): F1
ax = axes[0]
x = np.arange(len(datasets))
width = 0.35
for i, style in enumerate(styles):
    vals = [f1[d][style] for d in datasets]
    ax.bar(x + (i - 0.5) * width, vals, width, label=style_labels[style], color=colors[style],
           edgecolor="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylabel("F1")
ax.set_ylim(0, 1.0)
ax.set_title("(a) Llama-3.1-70B: F1 by dataset and prompt style (single run)")
ax.grid(axis="y", alpha=0.3, linestyle=":")
ax.legend(loc="upper right", fontsize=9)

# Panel (b): PPR with degenerate shading
ax = axes[1]
ax.axhspan(0.85, 1.0, color="#ffcdd2", alpha=0.5, zorder=0)
ax.axhspan(0.0, 0.15, color="#ffcdd2", alpha=0.5, zorder=0)
ax.axhline(0.5, color="black", linestyle="--", linewidth=1, zorder=1)
for i, style in enumerate(styles):
    vals = [ppr[d][style] for d in datasets]
    ax.bar(x + (i - 0.5) * width, vals, width, color=colors[style],
           edgecolor="black", linewidth=0.5, zorder=2)
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylabel("Predicted-positive rate")
ax.set_ylim(0, 1.0)
ax.set_title("(b) Predicted-positive rate (dashed = true 50% base rate; red = degenerate)")
ax.grid(axis="y", alpha=0.3, linestyle=":")

plt.tight_layout()
plt.savefig("figures/llama70b_chart.png", dpi=200, bbox_inches="tight")
print("Saved: figures/llama70b_chart.png")
