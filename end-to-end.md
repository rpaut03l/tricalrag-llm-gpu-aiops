# TriCalRAG - End-to-End: Setup → Benchmark → Paper → arXiv

**Status log:** this file is the living record of what's actually been done vs. what's next. Update it as each phase completes — don't let it drift out of sync with reality (an earlier version of this file predated the repo's actual structure; this rewrite fixes that).

## What we're building
A **benchmark**, not a one-off experiment: 4 real log datasets (BGL, HDFS, Thunderbird, OpenStack) × 4 local open-weight LLMs × 2 baselines (cloud API + classical DeepLog) × 3 prompt styles (zero-shot, few-shot, RAG), scored with statistical rigor (3 seeds, bootstrap 95% CI), plus ablations (batch scaling, quantization). The core benchmark makes a narrow, fully defensible claim using only real data and real model runs. Preliminary synthetic-data explorations (hardware cross-layer, full-stack trust/provisioning) live separately in `extensions/` and are **not** part of the core validated results — see `extensions/README.md`.

---

## PHASE 0 - Machine Facts (recorded, don't re-verify unless something changes)

- **GPU**: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 96GB VRAM (97,887 MiB), idle at baseline
- **Driver**: 595.58.03, **CUDA**: 13.2 (driver-reported)
- **Confirmed working stack** (as of actual install on this machine): `torch==2.13.0+cu130`, `vllm==0.28.0`, `transformers==5.16.1` — the **stable** vLLM release (0.28.0) has native Blackwell/CUDA 13.0 support. The nightly-build fallback in earlier notes was unnecessary — stable install worked directly via `pip install -r requirements.txt`.
- **GitHub auth on this machine**: a pre-existing SSH config had an old `Host github.com` block pointing to a repo-scoped deploy key (`~/.ssh/github_rtx6000`), which silently overrode any new key added afterward (SSH configs are first-match-wins). Fixed by rewriting `~/.ssh/config` to a single `Host github.com` block pointing to a new personal key (`~/.ssh/id_ed25519_personal`) with `IdentitiesOnly yes`. If cloning any new repo fails with "Repository not found" despite `ssh -T git@github.com` succeeding, check `cat ~/.ssh/config` for duplicate `Host github.com` entries first.
- **CLI note**: `huggingface-cli` is deprecated on this environment's `huggingface_hub` version — use `hf auth login` (not `huggingface-cli login`, not `hf login`).
- **Llama models are gated**: `meta-llama/*` models return `403 GatedRepoError` even when authenticated — you must separately visit the model page (e.g. https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and accept Meta's license form. Approval isn't instant. **Workaround used**: swapped the first validation run to `Qwen/Qwen2.5-14B-Instruct` (fully open, no gating) to keep moving while Llama access is pending — swap back once approved.
- **FlashInfer sampler JIT-compiles on first use and needs the CUDA toolkit (`nvcc`), not just the driver.** This machine has driver 13.2 but no toolkit at `/usr/local/cuda`, causing: `RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist` — this crashes the engine at the *first inference call*, after model load/compile/graph-capture all succeed (so it looks like it's almost done, then fails). **Fix, no toolkit install needed:**
  ```bash
  export VLLM_USE_FLASHINFER_SAMPLER=0
  echo 'export VLLM_USE_FLASHINFER_SAMPLER=0' >> ~/.bashrc   # make permanent
  ```
  With this set, vLLM falls back to a non-JIT sampling path with no measurable slowdown observed (600 incidents processed in 44.3s either way).

```bash
nvidia-smi
nvidia-smi --query-gpu=compute_cap --format=csv
```

## PHASE 1 - Clone + Environment Setup

**Status: done**

```bash
cd ~
git clone git@github.com:rpaut03l/tricalrag-llm-gpu-aiops.git
cd tricalrag-llm-gpu-aiops
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

**Step 1.5 - Blackwell/vLLM note**: NOT needed on this machine. The stable `pip install -r requirements.txt` resolved to `vllm==0.28.0` + `torch==2.13.0+cu130`, both with native Blackwell support out of the box. Skip the nightly-build workaround unless a future dependency resolution regresses this.

**Sanity check - confirmed passing:**
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

## PHASE 2 - Get the 4 Datasets

**Status: done - real datasets downloaded and verified on the GPU machine**

The public `git clone https://github.com/logpai/loghub.git` only ships 2,000-line **demo samples**, not the full datasets — each dataset's README circularly points back at the GitHub repo itself. The real full datasets live on **Zenodo (record 8196385)**, freely downloadable with no login or approval process:

```bash
cd ~/tricalrag-llm-gpu-aiops
git clone https://github.com/logpai/loghub.git   # gives 2k demo samples only, kept for reference/quick pipeline tests
mkdir -p loghub/full_datasets && cd loghub/full_datasets

wget "https://zenodo.org/records/8196385/files/BGL.zip?download=1" -O BGL.zip
wget "https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1" -O HDFS_v1.zip
wget "https://zenodo.org/records/8196385/files/OpenStack.tar.gz?download=1" -O OpenStack.tar.gz
wget "https://zenodo.org/records/8196385/files/Thunderbird.tar.gz?download=1" -O Thunderbird.tar.gz

unzip BGL.zip -d BGL/
unzip HDFS_v1.zip -d HDFS/
tar -xzf OpenStack.tar.gz -C .
tar -xzf Thunderbird.tar.gz -C .
```

**Real extracted structure (verified on this machine — do not assume LogHub's naming is consistent across datasets):**
```
loghub/full_datasets/BGL/BGL.log                              (743 MB, single file, matches expected space-separated format)
loghub/full_datasets/HDFS/HDFS.log                             (1.6 GB)
loghub/full_datasets/HDFS/preprocessed/anomaly_label.csv       (NOTE: nested under preprocessed/, not flat)
loghub/full_datasets/openstack_normal1.log                     (extracted FLAT, not into an OpenStack/ subfolder)
loghub/full_datasets/openstack_normal2.log
loghub/full_datasets/openstack_abnormal.log
loghub/full_datasets/anomaly_labels.txt                        (lists VM instance UUIDs with injected anomalies, referenced by openstack_abnormal.log)
loghub/full_datasets/Thunderbird.log                           (31.7 GB — DO NOT load directly, see below)
```

**Thunderbird is 31.7GB - do not load the full file.** Our loader uses `readlines()`, which would try to pull the entire file into RAM. Create a bounded subset instead:
```bash
head -n 2000000 Thunderbird.log > Thunderbird_subset.log
```
This gives ~257MB (2M lines) - manageable, and still large enough for a representative anomaly/normal mix.

**OpenStack is NOT one combined log with inline anomaly markers** (our original loader assumed this — it was wrong). The real format is three separate files: `openstack_normal1.log` and `openstack_normal2.log` (no anomalies), and `openstack_abnormal.log` (contains injected anomalies tied to 4 specific VM instance UUIDs listed in `anomaly_labels.txt`). Fixed in `multi_dataset_loader.py`: `parse_openstack()` now takes three file paths and labels all abnormal-file lines as anomalous, all normal-file lines as normal.

**`multi_dataset_loader.py` has been updated** with a corrected `parse_openstack()` function and real verified paths for all four datasets (see the `config` dict in `if __name__ == "__main__"`). Confirm your extracted paths match before running — if `full_datasets/` ends up structured differently on a re-download, update the config dict accordingly rather than assuming this layout is guaranteed by LogHub.

```bash
cd ~/tricalrag-llm-gpu-aiops/benchmark
python loaders/multi_dataset_loader.py --seed 42 --out data/incidents.jsonl
```

**Confirmed working on the GPU machine** - output:
```
Wrote 600 incidents to data/incidents.jsonl
Per-dataset breakdown: {'bgl': 150, 'hdfs': 150, 'thunderbird': 150, 'openstack': 150}
```
Clean 150/dataset balanced split, no errors - BGL's real format matched the parser's column assumptions without needing adjustment.
**Why:** normalizes all 4 datasets into one schema — 150 balanced incidents each, 600 total. Check the printed per-dataset breakdown before moving on.

## PHASE 3 - Run the Main Benchmark (3 seeds × 3 prompt styles)

**Status: done (2-model sweep) - all 9 seed×style combos complete for Qwen2.5-14B + Mistral-Small (1201 lines each, 2 models × 600 incidents + header). Llama-3.1-8B and the 70B model still pending (see below).**

**Bug hit and fixed during this run**: RAG prompts (which inject 3 retrieved past incidents as context) overflowed the original `max_model_len=4096` — retrieved context pushed some prompts to 4097+ tokens. All 3 RAG runs (seeds 1/2/3) failed identically with `VLLMValidationError: maximum context length is 4096 tokens`, while zero-shot and few-shot (no retrieval context) completed fine. **Fix**: bumped `max_model_len=4096` → `max_model_len=8192` in `benchmark.py`. Re-ran only the 3 broken RAG files (deleted the header-only stubs first so the smart-resume loop didn't skip them) — all completed successfully after the fix.

**Confirmed timing** (Mistral-Small, seed 3, RAG, post-fix): 600 incidents in 273.7s (131.7 tok/s) — notably slower than zero-shot/few-shot (~45s) because RAG prompts are much longer (retrieved context adds significant token count per request). **RAG runs take ~4-5 min per model**, not ~90s like the other two styles — factor this into future timing estimates.

**Smart-resume pattern used** (safe to reuse for the remaining Llama/70B runs later):
```bash
for seed in 1 2 3; do
  for style in zero_shot few_shot rag; do
    outfile="results/raw_results_seed${seed}_${style}.csv"
    if [ -f "$outfile" ]; then
      echo "SKIP: $outfile already exists"
    else
      python benchmark.py --seed $seed --prompt-style $style
    fi
  done
done
```
Note: this only checks file *existence*, not completeness — always verify row counts (`wc -l results/*.csv`, expect 1201 per model added) after any resume, since a crashed run can leave a header-only stub that looks "done" to a naive existence check.

**Validation run confirmed** (Qwen2.5-14B, seed 1, zero-shot, all 600 incidents):
```
600 incidents in 44.3s (741.5 tok/s), VRAM ~85289MB
Done -> results/raw_results_seed1_zero_shot.csv
```
Model load + torch.compile + CUDA graph capture: ~40s (with weights cache warm). Full inference pass: 44.3s. **Total wall time per model per seed/style combo: under 90 seconds** once the FlashInfer fix (see Phase 0) is applied and weights are cached locally.

**Llama-3.1-8B access status: still pending Meta's manual gated-repo approval** (confirmed via direct `hf_hub_download` test, not just `model_info` — the latter only reads public metadata and does NOT confirm actual file-download access even on gated repos). Submitted the license form; approval can take hours to days. **Do not block the sweep waiting on this** — run with the 2 confirmed-accessible models now, add Llama back into `MODELS` and re-run once approved.

**Current `MODELS` list in `benchmark.py`:**
```python
MODELS = [
#    {"name": "llama3.1-8b",   "hf_id": "meta-llama/Llama-3.1-8B-Instruct"},   # PENDING Meta approval
     {"name": "qwen2.5-14b",   "hf_id": "Qwen/Qwen2.5-14B-Instruct"},          # confirmed working
     {"name": "mistral-small", "hf_id": "mistralai/Mistral-Small-Instruct-2409"},  # no gating, added this run
#    {"name": "llama3.3-70b-awq", "hf_id": "hugging-quants/Meta-Llama-3.3-70B-Instruct-AWQ-INT4"},  # add after 2-model sweep confirmed clean
]
```

### Running the full sweep in `screen` (survives SSH disconnects)

```bash
which screen || sudo apt install -y screen
screen -S logsentinel-sweep
```

Inside the new screen session:
```bash
cd ~/tricalrag-llm-gpu-aiops/benchmark
source ../logsentinel-env/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0

for seed in 1 2 3; do
  python benchmark.py --seed $seed --prompt-style zero_shot
  python benchmark.py --seed $seed --prompt-style few_shot
  python benchmark.py --seed $seed --prompt-style rag
done 2>&1 | tee sweep_log_$(date +%Y%m%d_%H%M).txt
```

**Detach** (leaves it running): `Ctrl+A`, then `D`

**Reattach later to check progress:**
```bash
screen -r logsentinel-sweep
```

**Check progress without reattaching:**
```bash
tail -20 ~/tricalrag-llm-gpu-aiops/benchmark/sweep_log_*.txt
```

**Confirm the session is still alive:**
```bash
screen -ls
```

**Watch VRAM in a separate terminal/screen window:** `watch -n 1 nvidia-smi`

If the 70B model (once enabled) hits VRAM limits, comment it out of `MODELS` and run it alone afterward with `gpu_memory_utilization=0.95`.

**Timing estimate**: 2 models × 3 seeds × 3 styles = 18 runs. At ~90s per run (confirmed Qwen timing; Mistral may run somewhat slower) plus a one-time Mistral-Small download (~44GB, not yet cached — could take 10-20+ min depending on connection speed), expect roughly **30–60 minutes** for this 2-model sweep. Once Llama access clears and/or the 70B is added, re-run with the full 4-model list — that full sweep will take longer (~1.5–2.5 hours), primarily driven by the 70B's larger size.

## PHASE 4 - Score with Bootstrap CI

**Status: not started**

```bash
python score_results.py
```
Produces `results/summary_metrics.csv` (per-dataset, per-model, with 95% CI) and `results/macro_summary.csv` (macro-averaged headline table).

## PHASE 5 - Run Ablations

**Status: not started**

```bash
python ablation.py --mode batch_sweep
python ablation.py --mode quantization
```

## PHASE 6 - Run the DeepLog Baseline

**Status: not started**

```bash
cd ../baselines
python deeplog_baseline.py
```
Output lands in `../benchmark/results/deeplog_baseline.json`.

## PHASE 7 - (Optional) Cloud API Baseline

**Status: not started**

Add a script calling GPT-4o-mini or Claude Haiku on the same `data/incidents.jsonl` with the same prompt template, log latency + cost per call, score identically. Needed for the "local vs. cloud" comparison claim in the abstract.

## PHASE 8 - Generate Figures + Write the Paper

**Status: not started**

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

## PHASE 9 - GitHub

**Status: done** — repo live at `https://github.com/rpaut03l/tricalrag-llm-gpu-aiops`, restructured (core benchmark vs. `extensions/`) on `main` as of the `restructure/core-vs-extensions` PR merge.

Once real results exist, commit them:
```bash
cd ~/tricalrag-llm-gpu-aiops
git checkout -b add-benchmark-results
git add benchmark/results/ benchmark/data/incidents.jsonl paper/main.tex paper/figures/
git commit -m "Add real benchmark results and finalized paper draft"
git push -u origin add-benchmark-results
gh pr create --title "Add real benchmark results" --base main
```

## PHASE 10 - arXiv Submission

**Status: not started**

1. Register: https://arxiv.org/user/register
2. Check endorsement need: https://arxiv.org/auth/endorse (start early — can take days)
3. Primary category: `cs.DC`; cross-list `cs.AI`, `cs.LG`, `cs.PF`
4. Submit at https://arxiv.org/submit — upload `main.tex`, `references.bib`, `figures/`
5. Review the compiled PDF preview carefully, then finalize
6. Wait for moderation (1–2 business days) → get arXiv ID

## PHASE 11 - Hugging Face Papers + Papers with Code

**Status: not started**

Once you have an arXiv ID:
1. Submit to https://huggingface.co/papers/submit
2. Submit to https://paperswithcode.com/submit, linking the GitHub repo
3. Consider uploading `data/incidents.jsonl` splits to Hugging Face Datasets (respecting each source dataset's original license terms)

## PHASE 12 - AI-SPC 2026 Workshop Submission (parallel, optional)

**Status: not started**

- Trim to 4 pages + 1 reference page (IEEE format)
- Confirm with chairs that arXiv preprints are acceptable given the "unpublished manuscript" clause
- Submission opens Sept 1, 2026; deadline Oct 15, 2026

---

## Realistic expectation-setting
This benchmark is built to be reused and cited over time — real datasets, real models, real statistical rigor. That's the right target. "Millions of citations" isn't a realistic goal for any single paper; steady, compounding citations from a genuinely reusable benchmark is.
