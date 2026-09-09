"""
Redesign of the architecture figure as a swimlane diagram: each pipeline
stage sits in its own labeled lane with a consistent muted background,
one accent color per lane (not per box), uniform box styling, and a
single dark arrow color throughout. This avoids a "rainbow boxes" look
and reads closer to a systems-paper figure.

Run from paper/:
    python3 generate_full_architecture.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams["font.family"] = "DejaVu Sans"

ARROW_COLOR = "#37474f"
TEXT_COLOR = "#212121"
LANE_BG = "#eef2f6"
INFER_BG = "#e8eef1"

fig, ax = plt.subplots(figsize=(14.5, 8))
ax.set_xlim(0, 14.5)
ax.set_ylim(0, 8)
ax.axis("off")

def lane(x, y, w, h, color, label):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="none", zorder=0))
    ax.text(x + 0.15, y + h - 0.18, label, fontsize=10, weight="bold",
             color="#455a64", ha="left", va="top", zorder=1)

def box(x, y, w, h, text, border, fill="#ffffff", fontsize=8.3, weight="normal", zorder=3):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.035,rounding_size=0.06",
                        linewidth=1.3, edgecolor=border, facecolor=fill, zorder=zorder)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize,
             color=TEXT_COLOR, weight=weight, zorder=zorder+1)

def arrow(p1, p2, connectionstyle=None, lw=1.3, zorder=2):
    kwargs = dict(arrowstyle="-|>", mutation_scale=11, linewidth=lw,
                   color=ARROW_COLOR, zorder=zorder)
    if connectionstyle:
        kwargs["connectionstyle"] = connectionstyle
    ax.add_patch(FancyArrowPatch(p1, p2, **kwargs))

# Lane backgrounds
lane(0.0, 6.4, 14.5, 1.6, LANE_BG, "1 · DATA")
lane(0.0, 3.6, 14.5, 2.8, LANE_BG, "2 · PROMPTING STRATEGY")
lane(0.0, 1.5, 14.5, 2.1, INFER_BG, "3 · ON-PREMISE INFERENCE")
lane(0.0, 0.0, 14.5, 1.4, LANE_BG, "4 · EVALUATION")

BORDER = "#37474f"

# ---------------------------------------------------------------------
# LANE 1: Data
# ---------------------------------------------------------------------
ds_names = ["BGL", "HDFS", "Thunderbird", "OpenStack"]
ds_w, ds_h = 1.7, 0.5
ds_y = 7.2
ds_gap = 2.05
ds_x0 = 0.6
ds_centers = []
for i, name in enumerate(ds_names):
    bx = ds_x0 + i * ds_gap
    box(bx, ds_y, ds_w, ds_h, name, BORDER, fontsize=8.5)
    ds_centers.append(bx + ds_w/2)

loader_x, loader_y, loader_w, loader_h = 3.5, 6.5, 4.6, 0.55
box(loader_x, loader_y, loader_w, loader_h,
    "Unified loader  ->  600 incidents (150/dataset, balanced 50/50)",
    BORDER, fontsize=8.3)
for cx in ds_centers:
    arrow((cx, ds_y), (loader_x + loader_w/2, loader_y + loader_h))

# ---------------------------------------------------------------------
# LANE 2: Prompting strategies
# ---------------------------------------------------------------------
incident_pt = (loader_x + loader_w/2, loader_y)

strat_y, strat_h, strat_w = 4.55, 0.55, 2.4
strat_gap = 3.2
strat_x0 = 1.4
labels = ["Zero-shot\ninstruction only",
          "Few-shot\n+2 fixed examples",
          "RAG\n+3 retrieved incidents"]
strat_centers = []
for i, text in enumerate(labels):
    bx = strat_x0 + i * strat_gap
    box(bx, strat_y, strat_w, strat_h, text, BORDER, fontsize=8.3)
    strat_centers.append((bx, bx + strat_w))
    arrow(incident_pt, (bx + strat_w/2, strat_y + strat_h))

# RAG retrieval sub-flow, tucked below the RAG box within the same lane
rag_left, rag_right = strat_centers[2]
r1_x, r2_x = rag_left + 0.1, rag_left + 1.35
retr_y = 3.85
box(r1_x, retr_y, 1.15, 0.45, "FAISS index\n(MiniLM-L6)", "#5d4037", fontsize=6.8)
box(r2_x, retr_y, 1.15, 0.45, "Top-3 similar\n(excl. self)", "#5d4037", fontsize=6.8)
arrow((rag_left + 0.5, strat_y), (r1_x + 0.55, retr_y + 0.45), connectionstyle="arc3,rad=-0.1")
arrow((r1_x + 1.15, retr_y + 0.22), (r2_x, retr_y + 0.22))
arrow((r2_x + 0.55, retr_y + 0.45), (rag_left + 1.7, strat_y), connectionstyle="arc3,rad=-0.25")

# ---------------------------------------------------------------------
# LANE 3: Inference
# ---------------------------------------------------------------------
gpu_x, gpu_y, gpu_w, gpu_h = 4.6, 1.85, 5.3, 0.95
box(gpu_x, gpu_y, gpu_w, gpu_h,
    "vLLM inference engine  -  Qwen2.5-14B  |  Mistral-Small\n"
    "single NVIDIA RTX PRO 6000 (96GB VRAM)",
    "#1a5276", fill="#ffffff", fontsize=8.6, weight="bold")

targets = [gpu_x + gpu_w*0.2, gpu_x + gpu_w*0.5, gpu_x + gpu_w*0.8]
for (left, right), tx in zip(strat_centers, targets):
    arrow(((left+right)/2, strat_y), (tx, gpu_y + gpu_h))

out_x, out_y, out_w, out_h = 10.4, 1.85, 3.6, 0.95
box(out_x, out_y, out_w, out_h,
    "Structured output (JSON)\nis_anomaly - severity\nroot_cause - remediation",
    BORDER, fontsize=7.6)
arrow((gpu_x + gpu_w, gpu_y + gpu_h/2), (out_x, out_y + out_h/2))

# ---------------------------------------------------------------------
# LANE 4: Evaluation
# ---------------------------------------------------------------------
deeplog_x, deeplog_y, deeplog_w, deeplog_h = 0.6, 0.35, 3.4, 0.6
box(deeplog_x, deeplog_y, deeplog_w, deeplog_h,
    "DeepLog (LSTM) baseline\ntrain/test split, no leakage",
    "#795548", fontsize=7.6)

score_x, score_y, score_w, score_h = 9.6, 0.25, 4.4, 0.8
box(score_x, score_y, score_w, score_h,
    "Bootstrap 95% CI  -  F1 / Precision / Recall\n"
    "+ calibration diagnostic (3 seeds x 3 styles)",
    BORDER, fontsize=7.8)

arrow((out_x + out_w/2, out_y), (score_x + score_w/2, score_y + score_h))
arrow((deeplog_x + deeplog_w, deeplog_y + deeplog_h/2), (score_x, score_y + 0.2),
      connectionstyle="arc3,rad=-0.08")

plt.tight_layout()
plt.savefig("figures/full_architecture.png", dpi=230, bbox_inches="tight", facecolor="white")
print("Saved: figures/full_architecture.png")
