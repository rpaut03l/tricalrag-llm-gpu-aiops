# LogSentinel — Full Pipeline, Setup to Public Release

## What we're building (recap)
A **benchmark**, not just a one-off experiment: 4 real log datasets x 4 local
open-weight LLMs x 2 baselines (cloud API + classical DeepLog), scored with
statistical rigor (3 seeds, bootstrap 95% CI), plus ablations (prompt style,
batch scaling, quantization). Released as open code + data splits so other
researchers can reuse it — this reuse is what actually compounds citations
over time, unlike a single closed experiment.

---

## PHASE 1 — GPU Machine Setup
```bash
nvidia-smi
sudo apt update && sudo apt install -y python3-pip python3-venv git texlive-full
python3 -m venv logsentinel-env
source logsentinel-env/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
cd benchmark
pip install vllm pandas huggingface_hub tqdm scikit-learn matplotlib scipy
```

## PHASE 2 — Get the 4 Datasets
```bash
cd ..
git clone https://github.com/logpai/loghub.git
```
Follow each dataset's README.md inside `loghub/` (BGL, HDFS, Thunderbird,
OpenStack) — LogHub gates the actual `.log` files behind Zenodo links for
licensing/size reasons. Download and place them at:
```
loghub/BGL/BGL.log
loghub/HDFS/HDFS.log
loghub/HDFS/anomaly_label.csv     (ships alongside HDFS.log)
loghub/Thunderbird/Thunderbird.log
loghub/OpenStack/OpenStack.log
```

```bash
cd benchmark
huggingface-cli login
python loaders/multi_dataset_loader.py --seed 42 --out data/incidents.jsonl
```
**Why:** normalizes all 4 datasets into one schema — 150 balanced incidents
each, 600 total. Check the printed per-dataset breakdown matches expectations.

## PHASE 3 — Run the Main Benchmark (3 seeds x 2 prompt styles)
```bash
for seed in 1 2 3; do
  python benchmark.py --seed $seed --prompt-style zero_shot
  python benchmark.py --seed $seed --prompt-style few_shot
done
```
**Why 3 seeds x 2 styles = 6 runs per model:** gives you both the
statistical rigor (mean ± CI) and the prompt-style ablation from one sweep.
Monitor VRAM in a second terminal: `watch -n 1 nvidia-smi`

If you hit VRAM limits on the 70B model, comment it out of `MODELS` in
`benchmark.py` and run it alone afterward with `gpu_memory_utilization=0.95`.

## PHASE 4 — Score with Bootstrap CI
```bash
python score_results.py
```
**Why:** produces `results/summary_metrics.csv` (per-dataset, per-model, with
95% CI) and `results/macro_summary.csv` (macro-averaged headline table).

## PHASE 5 — Run Ablations
```bash
python ablation.py --mode batch_sweep
python ablation.py --mode quantization
```
**Why:** batch sweep shows throughput scaling on your single 96GB card
(novel data point — most papers assume clusters). Quantization comparison
shows whether 4-bit AWQ hurts structured RCA accuracy enough to matter.

## PHASE 6 — Run the DeepLog Baseline
```bash
cd ../baselines
pip install torch scikit-learn
python deeplog_baseline.py
```
**Why:** gives you the classical-ML comparison point every AIOps reviewer
will expect. Output lands in `../benchmark/results/deeplog_baseline.json`.

## PHASE 7 — (Optional but recommended) Cloud API Baseline
Add a small script calling GPT-4o-mini or Claude Haiku on the same
`data/incidents.jsonl` with the same prompt template, log latency + cost
per call, and score identically. This is the direct "local vs. cloud" claim
your abstract needs.

## PHASE 8 — Generate Figures + Paper
```bash
cd ../paper
python3 generate_diagrams.py       # Figure 1 (pipeline) + Figure 2 (scope matrix)
pip install pandas --break-system-packages
python3 make_table.py              # (adapt from earlier version to read macro_summary.csv)
```
Fill in `main.tex`:
- Abstract's `[FILL IN YOUR TOP-LINE FINDING]`
- Introduction, Related Work (add 3-5 recent arXiv papers on LLM-for-AIOps you find via search)
- Results tables from `macro_summary.csv` and `summary_metrics.csv`
- Discussion: cost-per-1000-incidents math, when LLMs beat/lose to DeepLog

Compile:
```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
Proofread `main.pdf` fully before moving on.

## PHASE 9 — Publish Code Publicly on GitHub
```bash
cd ..
git init
cat > .gitignore << 'EOF'
logsentinel-env/
*.pyc
__pycache__/
data/*.jsonl
loghub/
EOF
git add .
git commit -m "Initial release: LogSentinel benchmark for on-prem LLM-based AIOps RCA"
gh repo create logsentinel-benchmark --public --source=. --push
```
**Why now, before arXiv:** you want the GitHub URL live so you can put it in
the paper's abstract/footnote before submission — reviewers and future
citers click through to it immediately.

Add a proper `README.md` to the repo root documenting how to reproduce every
result (you already have the phases above — reuse them there).

## PHASE 10 — arXiv Submission
1. Register: https://arxiv.org/user/register
2. Check endorsement need: https://arxiv.org/auth/endorse (start early, can take days)
3. Primary category: `cs.DC`; cross-list `cs.AI`, `cs.LG`, `cs.PF`
4. Submit at https://arxiv.org/submit — upload `main.tex`, `references.bib`, `figures/`
5. Review the compiled PDF preview carefully, then finalize
6. Wait for moderation (1–2 business days) → get your arXiv ID

## PHASE 11 — Hugging Face Papers + Papers with Code
Once you have an arXiv ID:
1. **Hugging Face Papers**: go to https://huggingface.co/papers/submit and submit your arXiv link — this surfaces it to the HF community and often drives early citations/discussion.
2. **Papers with Code**: go to https://paperswithcode.com/submit, link your arXiv paper + your GitHub repo — this is specifically what makes benchmark papers discoverable to people looking for "SOTA on X" style leaderboards, which is a major long-term citation driver for benchmark-style papers.
3. Also consider uploading your dataset splits (the `data/incidents.jsonl` files, properly licensed per each source dataset's terms) to **Hugging Face Datasets** so people can `load_dataset("your-username/logsentinel")` directly.

## PHASE 12 — Also Submit to AI-SPC Workshop (parallel, optional)
- Trim to 4 pages + 1 reference page (IEEE format) — keep only the macro
  results table and one ablation, move the rest to an appendix or "extended
  version on arXiv" footnote.
- Confirm with chairs that arXiv preprints are acceptable given the
  "unpublished manuscript" clause: <subhasis.banerjee@shell.com>,
  <aniruddha.panda@shell.com>
- Submission opens Sept 1, 2026; deadline Oct 15, 2026.

---

## Realistic expectation-setting
A well-executed benchmark paper with public code in a genuinely useful niche
(on-prem AIOps is a real, growing need) can realistically accumulate
citations over 1-3 years as people build on it or cite it as related work —
but "millions of citations" is not something any single paper achieves;
even top-cited-of-all-time papers (BERT, ResNet, Adam) sit in the tens of
thousands after a decade. Building it this rigorously is what maximizes
your realistic odds of steady, compounding citations — that's the honest
target.
