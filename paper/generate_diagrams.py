"""
Generates two figures for the paper:
  Figure 1: end-to-end pipeline architecture
  Figure 2: benchmark scope (4 datasets x 4 local models x 2 baselines)
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------------------
# FIGURE 1: Pipeline architecture
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12.5, 3))
ax.set_xlim(0, 12)
ax.set_ylim(0, 3)
ax.axis("off")

stages = [
    "4 Real Log\nDatasets\n(BGL/HDFS/\nThunderbird/\nOpenStack)",
    "Unified\nIncident\nWindowing",
    "Local LLM\nvia vLLM\n(RTX PRO 6000)",
    "Structured\nRCA Output\n(JSON)",
    "Bootstrap CI\nScoring vs.\nBaselines",
]
x_positions = [0.3, 2.7, 5.0, 7.5, 9.6]
box_width = 1.9
box_height = 1.9
y_center = 1.5

for x, label in zip(x_positions, stages):
    box = FancyBboxPatch((x, y_center - box_height/2), box_width, box_height,
                          boxstyle="round,pad=0.05,rounding_size=0.08",
                          linewidth=1.5, edgecolor="black", facecolor="#e8f0fe")
    ax.add_patch(box)
    ax.text(x + box_width/2, y_center, label, ha="center", va="center", fontsize=8.5)

for i in range(len(x_positions) - 1):
    arrow = FancyArrowPatch((x_positions[i] + box_width, y_center),
                             (x_positions[i+1], y_center),
                             arrowstyle="-|>", mutation_scale=15, linewidth=1.5, color="black")
    ax.add_patch(arrow)

plt.tight_layout()
plt.savefig("figures/pipeline_architecture.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: figures/pipeline_architecture.png")


# ---------------------------------------------------------------------------
# FIGURE 2: Benchmark scope matrix (datasets x models/baselines)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
ax.set_xlim(0, 9)
ax.set_ylim(0, 5)
ax.axis("off")

datasets = ["BGL", "HDFS", "Thunderbird", "OpenStack"]
systems = ["Llama-3.1-8B", "Qwen2.5-14B", "Mistral-Small-22B",
           "Llama-3.3-70B-AWQ", "Cloud API\n(GPT-4o-mini)", "DeepLog\n(LSTM baseline)"]

col_w = 8.4 / len(datasets)
row_h = 3.6 / len(systems)
x0, y0 = 0.3, 4.3

# header row
ax.text(x0 - 0.2, y0 + 0.4, "Local LLMs", fontsize=10, fontweight="bold")
for j, ds in enumerate(datasets):
    ax.text(x0 + j*col_w + col_w/2, y0 + 0.4, ds, ha="center", fontsize=9, fontweight="bold")

for i, sysname in enumerate(systems):
    y = y0 - i*row_h
    is_baseline = "Cloud" in sysname or "DeepLog" in sysname
    color = "#fce8e6" if is_baseline else "#e8f0fe"
    ax.text(x0 - 0.25, y, sysname, ha="right", va="center", fontsize=8)
    for j in range(len(datasets)):
        box = FancyBboxPatch((x0 + j*col_w, y - row_h*0.35), col_w*0.9, row_h*0.7,
                              boxstyle="round,pad=0.02,rounding_size=0.04",
                              linewidth=1, edgecolor="black", facecolor=color)
        ax.add_patch(box)
        ax.text(x0 + j*col_w + col_w*0.45, y, "✓", ha="center", va="center", fontsize=10)

ax.text(4.5, 0.1, "Blue = local open-weight models (this work)   |   Red = baselines",
        ha="center", fontsize=8, style="italic")

plt.tight_layout()
plt.savefig("figures/benchmark_scope.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: figures/benchmark_scope.png")
