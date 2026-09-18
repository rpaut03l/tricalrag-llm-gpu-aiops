"""
TriCalRAG end-to-end architecture, v2. Extends the original 4-lane
swimlane diagram with the v2 additions:
  - Lane 3 now shows THREE inference paths in parallel: the primary
    vLLM engine (Qwen2.5-14B | Mistral-Small, 3-seed protocol) plus two
    single-run supplementary checks -- Llama-3.1-70B via Ollama, and
    Claude Haiku 4.5 via cloud API -- matching the green/orange
    "primary protocol" vs "supplementary single-run" color convention
    already used in Fig. 2 (benchmark scope) and Figs. 4-5 of the paper.
  - Lane 4 adds the Calibration Correction Baselines (CC/BC) as a
    second, log-probability-based scoring path alongside the main
    bootstrap-CI scoring.

Same overall size/aspect ratio and visual language as the original
(matplotlib patches, swimlane bands) so it drops into the paper as a
like-for-like replacement.

Run:
    python3 generate_architecture_v2.py
Output:
    figures/architecture_v2.png
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
PRIMARY_BORDER = "#1a5276"      # matches vLLM box border in original
SUPP_BORDER = "#e65100"          # orange = supplementary single-run
SUPP_FILL = "#fff3e0"

fig, ax = plt.subplots(figsize=(14.5, 9.4))
ax.set_xlim(0, 14.5)
ax.set_ylim(0, 9.4)
ax.axis("off")

def lane(x, y, w, h, color, label):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="none", zorder=0))
    ax.text(x + 0.15, y + h - 0.18, label, fontsize=10, weight="bold",
             color="#455a64", ha="left", va="top", zorder=1)

def box(x, y, w, h, text, border, fill="#ffffff", fontsize=8.3, weight="normal",
        zorder=3, dashed=False, lw=1.3):
    style = "round,pad=0.035,rounding_size=0.06"
    ls = (0, (4, 2)) if dashed else "solid"
    b = FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=lw,
                        edgecolor=border, facecolor=fill, zorder=zorder, linestyle=ls)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize,
             color=TEXT_COLOR, weight=weight, zorder=zorder+1)

def arrow(p1, p2, connectionstyle=None, lw=1.3, zorder=2, color=ARROW_COLOR, dashed=False):
    kwargs = dict(arrowstyle="-|>", mutation_scale=11, linewidth=lw, color=color, zorder=zorder)
    if dashed:
        kwargs["linestyle"] = (0, (4, 2))
    if connectionstyle:
        kwargs["connectionstyle"] = connectionstyle
    ax.add_patch(FancyArrowPatch(p1, p2, **kwargs))

BORDER = "#37474f"

# ---------------------------------------------------------------------
# Lane backgrounds (Lane 3 grows to fit 3 parallel inference paths)
# ---------------------------------------------------------------------
lane(0.0, 7.9, 14.5, 1.5, LANE_BG, "1 \u00b7 DATA")
lane(0.0, 5.0, 14.5, 2.9, LANE_BG, "2 \u00b7 PROMPTING STRATEGY")
lane(0.0, 1.9, 14.5, 3.1, INFER_BG, "3 \u00b7 ON-PREMISE / CLOUD INFERENCE")
lane(0.0, 0.0, 14.5, 1.9, LANE_BG, "4 \u00b7 EVALUATION")

# ---------------------------------------------------------------------
# LANE 1: Data
# ---------------------------------------------------------------------
ds_names = ["BGL", "HDFS", "Thunderbird", "OpenStack"]
ds_w, ds_h = 1.7, 0.5
ds_y = 8.55
ds_gap = 2.05
ds_x0 = 0.6
ds_centers = []
for i, name in enumerate(ds_names):
    bx = ds_x0 + i * ds_gap
    box(bx, ds_y, ds_w, ds_h, name, BORDER, fontsize=8.5)
    ds_centers.append(bx + ds_w/2)

loader_x, loader_y, loader_w, loader_h = 3.5, 7.95, 4.6, 0.5
box(loader_x, loader_y, loader_w, loader_h,
    "Unified loader  ->  600 incidents (150/dataset, balanced 50/50)",
    BORDER, fontsize=8.3)
for cx in ds_centers:
    arrow((cx, ds_y), (loader_x + loader_w/2, loader_y + loader_h))

# ---------------------------------------------------------------------
# LANE 2: Prompting strategies
# ---------------------------------------------------------------------
incident_pt = (loader_x + loader_w/2, loader_y)

strat_y, strat_h, strat_w = 6.85, 0.55, 2.4
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

rag_left, rag_right = strat_centers[2]
r1_x, r2_x = rag_left + 0.1, rag_left + 1.35
retr_y = 6.15
box(r1_x, retr_y, 1.15, 0.45, "FAISS index\n(MiniLM-L6)", "#5d4037", fontsize=6.8)
box(r2_x, retr_y, 1.15, 0.45, "Top-3 similar\n(excl. self)", "#5d4037", fontsize=6.8)
arrow((rag_left + 0.5, strat_y), (r1_x + 0.55, retr_y + 0.45), connectionstyle="arc3,rad=-0.1")
arrow((r1_x + 1.15, retr_y + 0.22), (r2_x, retr_y + 0.22))
arrow((r2_x + 0.55, retr_y + 0.45), (rag_left + 1.7, strat_y), connectionstyle="arc3,rad=-0.25")

# ---------------------------------------------------------------------
# LANE 3: Inference -- primary vLLM engine + two supplementary paths
# ---------------------------------------------------------------------
vllm_x, vllm_y, vllm_w, vllm_h = 3.5, 4.0, 5.3, 0.95
box(vllm_x, vllm_y, vllm_w, vllm_h,
    "vLLM inference engine  -  Qwen2.5-14B  |  Mistral-Small\n"
    "single NVIDIA RTX PRO 6000 (96GB VRAM)  \u00b7  PRIMARY (3-seed)",
    PRIMARY_BORDER, fill="#ffffff", fontsize=8.4, weight="bold")

targets = [vllm_x + vllm_w*0.2, vllm_x + vllm_w*0.5, vllm_x + vllm_w*0.8]
for (left, right), tx in zip(strat_centers, targets):
    arrow(((left+right)/2, strat_y), (tx, vllm_y + vllm_h))

# Supplementary inference paths -- same incidents/prompts, single run
supp_y, supp_h = 2.2, 0.85
ollama_x, ollama_w = 0.6, 3.7
cloud_x, cloud_w = 4.6, 3.7
box(ollama_x, supp_y, ollama_w, supp_h,
    "Ollama (local)  \u00b7  Llama-3.1-70B (4-bit)\n"
    "SUPPLEMENTARY \u2014 single run",
    SUPP_BORDER, fill=SUPP_FILL, fontsize=7.8, dashed=True)
box(cloud_x, supp_y, cloud_w, supp_h,
    "Cloud API  \u00b7  Claude Haiku 4.5 (Anthropic)\n"
    "SUPPLEMENTARY \u2014 single run",
    SUPP_BORDER, fill=SUPP_FILL, fontsize=7.8, dashed=True)

arrow((vllm_x + vllm_w*0.25, vllm_y), (ollama_x + ollama_w/2, supp_y + supp_h),
      dashed=True, color=SUPP_BORDER, connectionstyle="arc3,rad=0.15")
arrow((vllm_x + vllm_w*0.75, vllm_y), (cloud_x + cloud_w/2, supp_y + supp_h),
      dashed=True, color=SUPP_BORDER, connectionstyle="arc3,rad=-0.15")
ax.text(vllm_x + vllm_w/2, 3.55, "same 600 incidents, same prompt templates",
        fontsize=6.6, style="italic", ha="center", color="#616161")

out_x, out_y, out_w, out_h = 9.4, 2.6, 4.5, 0.95
box(out_x, out_y, out_w, out_h,
    "Structured output (JSON)\nis_anomaly - severity\nroot_cause - remediation",
    BORDER, fontsize=7.8)
arrow((vllm_x + vllm_w, vllm_y + vllm_h*0.5), (out_x, out_y + out_h*0.75))
arrow((ollama_x + ollama_w/2, supp_y + supp_h), (out_x, out_y + out_h*0.85),
      dashed=True, color=SUPP_BORDER, connectionstyle="arc3,rad=0.12")
arrow((cloud_x + cloud_w/2, supp_y + supp_h), (out_x, out_y + out_h*0.15),
      dashed=True, color=SUPP_BORDER, connectionstyle="arc3,rad=-0.12")

# ---------------------------------------------------------------------
# LANE 4: Evaluation
# ---------------------------------------------------------------------
deeplog_x, deeplog_y, deeplog_w, deeplog_h = 0.5, 0.65, 3.0, 0.6
box(deeplog_x, deeplog_y, deeplog_w, deeplog_h,
    "DeepLog (LSTM) baseline\ntrain/test split, no leakage",
    "#795548", fontsize=7.3)

score_x, score_y, score_w, score_h = 3.8, 0.55, 4.0, 0.8
box(score_x, score_y, score_w, score_h,
    "Bootstrap 95% CI  -  F1 / Precision / Recall\n"
    "+ calibration diagnostic (3 seeds x 3 styles)",
    BORDER, fontsize=7.4)

calib_x, calib_w = 8.2, 4.3
box(calib_x, score_y, calib_w, score_h,
    "Calibration baselines: CC / BC\n"
    "(log-prob scoring, Sec. V-D) \u00b7 supplementary",
    SUPP_BORDER, fill=SUPP_FILL, fontsize=7.2, dashed=True)

arrow((out_x + out_w*0.15, out_y), (score_x + score_w*0.75, score_y + score_h))
arrow((out_x + out_w*0.55, out_y), (calib_x + calib_w*0.45, score_y + score_h),
      dashed=True, color=SUPP_BORDER)
arrow((deeplog_x + deeplog_w, deeplog_y + deeplog_h/2), (score_x, score_y + 0.2),
      connectionstyle="arc3,rad=-0.08")

# ---------------------------------------------------------------------
# Legend (placed inside the visible plot range, not below it)
# ---------------------------------------------------------------------
ax.add_patch(FancyBboxPatch((0.5, 0.12), 0.35, 0.15, boxstyle="round,pad=0.01",
                              linewidth=1.3, edgecolor=PRIMARY_BORDER, facecolor="white"))
ax.text(1.0, 0.195, "Primary protocol (3 seeds, bootstrap CI)", fontsize=7.5, va="center")
ax.add_patch(FancyBboxPatch((6.2, 0.12), 0.35, 0.15, boxstyle="round,pad=0.01",
                              linewidth=1.3, edgecolor=SUPP_BORDER, facecolor=SUPP_FILL,
                              linestyle=(0, (4, 2))))
ax.text(6.7, 0.195, "Supplementary check (single run, v2)", fontsize=7.5, va="center")

plt.tight_layout()
plt.savefig("figures/architecture_v2.png", dpi=230, bbox_inches="tight", facecolor="white")
print("Saved: figures/architecture_v2.png")
