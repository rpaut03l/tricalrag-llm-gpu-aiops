# LogSentinel-RAG — End-to-End: Setup → Benchmark → Paper → arXiv

**Status log:** this file is the living record of what's actually been done vs. what's next. Update it as each phase completes — don't let it drift out of sync with reality (an earlier version of this file predated the repo's actual structure; this rewrite fixes that).

## What we're building
A **benchmark**, not a one-off experiment: 4 real log datasets (BGL, HDFS, Thunderbird, OpenStack) × 4 local open-weight LLMs × 2 baselines (cloud API + classical DeepLog) × 3 prompt styles (zero-shot, few-shot, RAG), scored with statistical rigor (3 seeds, bootstrap 95% CI), plus ablations (batch scaling, quantization). The core benchmark makes a narrow, fully defensible claim using only real data and real model runs. Preliminary synthetic-data explorations (hardware cross-layer, full-stack trust/provisioning) live separately in `extensions/` and are **not** part of the core validated results — see `extensions/README.md`.

---

## PHASE 0 — Machine Facts (recorded, don't re-verify unless something changes)

- **GPU**: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 96GB VRAM (97,887 MiB), idle at baseline
- **Driver**: 595.58.03, **CUDA**: 13.2 (driver-reported)
- **Confirmed working stack** (as of actual install on this machine): `torch==2.13.0+cu130`, `vllm==0.28.0`, `transformers==5.16.1` — the **stable** vLLM release (0.28.0) has native Blackwell/CUDA 13.0 support. The nightly-build fallback in earlier notes was unnecessary — stable install worked directly via `pip install -r requirements.txt`.
- **GitHub auth on this machine**: a pre-existing SSH config had an old `Host github.com` block pointing to a repo-scoped deploy key (`~/.ssh/github_rtx6000`), which silently overrode any new key added afterward (SSH configs are first-match-wins). Fixed by rewriting `~/.ssh/config` to a single `Host github.com` block pointing to a new personal key (`~/.ssh/id_ed25519_personal`) with `IdentitiesOnly yes`. If cloning any new repo fails with "Repository not found" despite `ssh -T git@github.com` succeeding, check `cat ~/.ssh/config` for duplicate `Host github.com` entries first.
- **CLI note**: `huggingface-cli` is deprecated on this environment's `huggingface_hub` version — use `hf auth login` (not `huggingface-cli login`, not `hf login`).

```bash
nvidia-smi
nvidia-smi --query-gpu=compute_cap --format=csv
```

## PHASE 1 — Clone + Environment Setup

**Status: ✅ done**

```bash
cd ~
git clone git@github.com:rpaut03l/logsentinel-rag-llm-gpu-aiops.git
cd logsentinel-rag-llm-gpu-aiops
pwd && ls -la
```

```bash
python3 -m venv logsentinel-env
source logsentinel-env/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
cd benchmark
pip install -r requirements.txt
```

**Step 1.5 — Blackwell/vLLM note**: NOT needed on this machine. The stable `pip install -r requirements.txt` resolved to `vllm==0.28.0` + `torch==2.13.0+cu130`, both with native Blackwell support out of the box. Skip the nightly-build workaround unless a future dependency resolution regresses this.

**Sanity check — confirmed passing:**
```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Output: 2.13.0+cu130 True NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
```

**Hugging Face auth** (needed before downloading gated models like Llama):
```bash
hf auth login
```
(Not `huggingface-cli login` — deprecated on this `huggingface_hub` version. Not `hf login` — that command doesn't exist; it's `hf auth login`.) Paste a token from https://huggingface.co/settings/tokens when prompted.

Verify:
```bash
hf auth whoami
```

## PHASE 2 — Get the 4 Datasets

**Status: 🔄 next up**

```bash
cd ~/logsentinel-rag-llm-gpu-aiops
git clone https://github.com/logpai/loghub.git
```
Follow each dataset's `README.md` inside `loghub/` (BGL, HDFS, Thunderbird, OpenStack) — LogHub gates the actual `.log` files behind Zenodo links for size/licensing reasons. Place them at:
```
loghub/BGL/BGL.log
loghub/HDFS/HDFS.log
loghub/HDFS/anomaly_label.csv     (ships alongside HDFS.log)
loghub/Thunderbird/Thunderbird.log
loghub/OpenStack/OpenStack.log
```

```bash
cd benchmark
python loaders/multi_dataset_loader.py --seed 42 --out data/incidents.jsonl
```
**Why:** normalizes all 4 datasets into one schema — 150 balanced incidents each, 600 total. Check the printed per-dataset breakdown before moving on.

## PHASE 3 — Run the Main Benchmark (3 seeds × 3 prompt styles)

**Status: ⬜ not yet run**

```bash
for seed in 1 2 3; do
  python benchmark.py --seed $seed --prompt-style zero_shot
  python benchmark.py --seed $seed --prompt-style few_shot
  python benchmark.py --seed $seed --prompt-style rag
done
```
Monitor VRAM in a second terminal: `watch -n 1 nvidia-smi`

If the 70B model hits VRAM limits, comment it out of `MODELS` in `benchmark.py` and run it alone afterward with `gpu_memory_utilization=0.95`.

**Recommended first run**: test with just `llama3.1-8b` (comment out the other 3 models) and one seed/style combo to confirm the pipeline works end-to-end before committing GPU time to the full 3×3×4 sweep.

## PHASE 4 — Score with Bootstrap CI

**Status: ⬜ not yet run**

```bash
python score_results.py
```
Produces `results/summary_metrics.csv` (per-dataset, per-model, with 95% CI) and `results/macro_summary.csv` (macro-averaged headline table).

## PHASE 5 — Run Ablations

**Status: ⬜ not yet run**

```bash
python ablation.py --mode batch_sweep
python ablation.py --mode quantization
```

## PHASE 6 — Run the DeepLog Baseline

**Status: ⬜ not yet run**

```bash
cd ../baselines
python deeplog_baseline.py
```
Output lands in `../benchmark/results/deeplog_baseline.json`.

## PHASE 7 — (Optional) Cloud API Baseline

**Status: ⬜ not yet built**

Add a script calling GPT-4o-mini or Claude Haiku on the same `data/incidents.jsonl` with the same prompt template, log latency + cost per call, score identically. Needed for the "local vs. cloud" comparison claim in the abstract.

## PHASE 8 — Generate Figures + Write the Paper

**Status: ⬜ not yet run**

```bash
cd ../paper
python3 generate_diagrams.py
python3 make_table.py   # reads macro_summary.csv, prints a LaTeX table to paste into main.tex
```
Fill in `main.tex`: abstract's top-line finding, Introduction, Related Work (add recent LLM-for-AIOps papers), Results tables, Discussion.

```bash
sudo apt install -y texlive-full   # if not already installed
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
Proofread `main.pdf` fully before moving on.

## PHASE 9 — GitHub

**Status: ✅ done** — repo live at `https://github.com/rpaut03l/logsentinel-rag-llm-gpu-aiops`, restructured (core benchmark vs. `extensions/`) on `main` as of the `restructure/core-vs-extensions` PR merge.

Once real results exist, commit them:
```bash
cd ~/logsentinel-rag-llm-gpu-aiops
git checkout -b add-benchmark-results
git add benchmark/results/ benchmark/data/incidents.jsonl paper/main.tex paper/figures/
git commit -m "Add real benchmark results and finalized paper draft"
git push -u origin add-benchmark-results
gh pr create --title "Add real benchmark results" --base main
```

## PHASE 10 — arXiv Submission

**Status: ⬜ not started**

1. Register: https://arxiv.org/user/register
2. Check endorsement need: https://arxiv.org/auth/endorse (start early — can take days)
3. Primary category: `cs.DC`; cross-list `cs.AI`, `cs.LG`, `cs.PF`
4. Submit at https://arxiv.org/submit — upload `main.tex`, `references.bib`, `figures/`
5. Review the compiled PDF preview carefully, then finalize
6. Wait for moderation (1–2 business days) → get arXiv ID

## PHASE 11 — Hugging Face Papers + Papers with Code

**Status: ⬜ not started**

Once you have an arXiv ID:
1. Submit to https://huggingface.co/papers/submit
2. Submit to https://paperswithcode.com/submit, linking the GitHub repo
3. Consider uploading `data/incidents.jsonl` splits to Hugging Face Datasets (respecting each source dataset's original license terms)

## PHASE 12 — AI-SPC 2026 Workshop Submission (parallel, optional)

**Status: ⬜ not started**

- Trim to 4 pages + 1 reference page (IEEE format)
- Confirm with chairs that arXiv preprints are acceptable given the "unpublished manuscript" clause
- Submission opens Sept 1, 2026; deadline Oct 15, 2026

---

## Realistic expectation-setting
This benchmark is built to be reused and cited over time — real datasets, real models, real statistical rigor. That's the right target. "Millions of citations" isn't a realistic goal for any single paper; steady, compounding citations from a genuinely reusable benchmark is.
