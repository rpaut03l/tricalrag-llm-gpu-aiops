"""
TriCalRAG v2 -- calibration baselines.

WHY this exists: v1 found that zero-shot prompting drives models toward
predicting "anomaly" on almost every incident. That phenomenon is a known
label bias in in-context learning, and there are two standard published
fixes. Reviewers will ask why we did not compare against them. This script
runs both, on the same incidents and prompt styles as v1, so the paper can
answer RQ2 directly: does retrieval (RAG) fix calibration better than the
standard calibration methods?

  CC  Contextual Calibration -- Zhao et al., "Calibrate Before Use", ICML 2021.
      Estimate the model's label bias from content-free inputs ("N/A"),
      then divide it out.
  BC  Batch Calibration -- Zhou et al., "Batch Calibration", ICLR 2024.
      Estimate the bias as the mean prediction over the test batch,
      then subtract it. Needs no labels and no extra inputs.

HOW the decision is scored: instead of parsing JSON text, we read the
model's log-probability of the next token after  "is_anomaly":  being
"true" vs "false". This gives a real probability P(true | x) that the two
calibration methods can operate on. v1's text-parsing path stays as-is
in benchmark.py; this is an additional measurement, not a replacement.

Usage (same layout as benchmark.py):
    python benchmark_calibrated.py --seed 1 --prompt-style zero_shot
    python benchmark_calibrated.py --seed 1 --prompt-style rag
    python benchmark_calibrated.py --models Qwen/Qwen2.5-14B-Instruct google/gemma-2-9b-it

Output: results/calib_seed{seed}_{style}.csv, one row per (model, incident)
with raw / CC / BC probabilities and predictions. Score with
score_calibration.py.
"""

import argparse
import csv
import gc
import json
import math
from pathlib import Path

import numpy as np
import torch
from vllm import LLM, SamplingParams

from prompts import build_prompt
from retrieval import IncidentRetriever, build_retrieval_context

DEFAULT_MODELS = [
    ("qwen2.5-14b",   "Qwen/Qwen2.5-14B-Instruct"),
    ("mistral-small", "mistralai/Mistral-Small-Instruct-2409"),
]

# The model is asked for a JSON object; we hand it the opening of that
# object and score the very next token. This keeps the prompt identical to
# v1 up to the point of decision.
ANSWER_PREFIX = '\n{\n  "is_anomaly": '
LABELS = {"true": "true", "false": "false"}

# Content-free inputs for Contextual Calibration (Zhao et al. use "N/A",
# "[MASK]", and the empty string; we average over all three).
CONTENT_FREE_INPUTS = ["N/A", "[MASK]", ""]


def load_incidents(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def label_token_count(tokenizer, prefix, label):
    """How many tokens the label adds to the prefix under this tokenizer.
    Using the difference in lengths absorbs any merging at the boundary."""
    n_prefix = len(tokenizer.encode(prefix, add_special_tokens=False))
    n_full = len(tokenizer.encode(prefix + label, add_special_tokens=False))
    return max(1, n_full - n_prefix)


def score_label_logprobs(llm, tokenizer, prefixes):
    """
    For each prefix, return (logp_true, logp_false): the summed log-prob
    of the label tokens given the prefix. Uses vLLM prompt_logprobs by
    scoring prefix+label as a prompt and reading the last k token logprobs.
    """
    prompts, meta = [], []
    for i, prefix in enumerate(prefixes):
        for name, label in LABELS.items():
            k = label_token_count(tokenizer, prefix, label)
            prompts.append(prefix + label)
            meta.append((i, name, k))

    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    outputs = llm.generate(prompts, sp)

    result = [[None, None] for _ in prefixes]
    for (i, name, k), out in zip(meta, outputs):
        ids = out.prompt_token_ids
        plps = out.prompt_logprobs  # list aligned with ids; entry 0 is None
        total = 0.0
        for j in range(len(ids) - k, len(ids)):
            entry = plps[j]
            tok_id = ids[j]
            if entry is None or tok_id not in entry:
                total += -30.0  # missing -> treat as very unlikely
            else:
                total += entry[tok_id].logprob
        result[i][0 if name == "true" else 1] = total
    return result


def normalize(lp_true, lp_false):
    """Binary softmax over the two label log-probs -> P(true)."""
    m = max(lp_true, lp_false)
    pt = math.exp(lp_true - m)
    pf = math.exp(lp_false - m)
    return pt / (pt + pf)


def contextual_calibration(p_true, p_true_cf):
    """
    Zhao et al. 2021: divide each class probability by its content-free
    probability, then renormalize. For the binary case this is
        s_true  = p_true / p_true_cf
        s_false = (1 - p_true) / (1 - p_true_cf)
        P_cc(true) = s_true / (s_true + s_false)
    """
    eps = 1e-9
    s_true = p_true / max(p_true_cf, eps)
    s_false = (1.0 - p_true) / max(1.0 - p_true_cf, eps)
    return s_true / (s_true + s_false)


def batch_calibration(p_true_all):
    """
    Zhou et al. 2024: subtract the batch-mean log-probability from each
    class, i.e. shift the decision boundary so the batch-average logit is
    zero. Returns calibrated P(true) for every item in the batch.
    """
    eps = 1e-9
    logit = np.log(np.clip(p_true_all, eps, 1 - eps)) - np.log(np.clip(1 - p_true_all, eps, 1 - eps))
    logit_cal = logit - logit.mean()
    return 1.0 / (1.0 + np.exp(-logit_cal))


def run_model(name, hf_id, incidents, prompt_style, seed, writer, retriever=None):
    print(f"\n=== {name} | style={prompt_style} | seed={seed} ===")
    llm = LLM(model=hf_id, dtype="auto", gpu_memory_utilization=0.85, max_model_len=8192)
    tokenizer = llm.get_tokenizer()

    # --- prompts for the real incidents (identical to v1 up to ANSWER_PREFIX) ---
    prefixes = []
    for inc in incidents:
        if prompt_style == "rag":
            retrieved = retriever.retrieve(inc["log_window"], exclude_id=inc["id"], top_k=3)
            ctx = build_retrieval_context(retrieved)
            p = build_prompt(inc["log_window"], style="rag", retrieved_context=ctx)
        else:
            p = build_prompt(inc["log_window"], style=prompt_style)
        prefixes.append(p + ANSWER_PREFIX)

    # --- content-free prompts for CC (same template, empty content) ---
    cf_prefixes = []
    for cf in CONTENT_FREE_INPUTS:
        if prompt_style == "rag":
            p = build_prompt(cf, style="rag", retrieved_context="None found.")
        else:
            p = build_prompt(cf, style=prompt_style)
        cf_prefixes.append(p + ANSWER_PREFIX)

    lp = score_label_logprobs(llm, tokenizer, prefixes)
    lp_cf = score_label_logprobs(llm, tokenizer, cf_prefixes)

    p_true = np.array([normalize(a, b) for a, b in lp])
    p_true_cf = float(np.mean([normalize(a, b) for a, b in lp_cf]))
    p_cc = np.array([contextual_calibration(p, p_true_cf) for p in p_true])
    p_bc = batch_calibration(p_true)

    print(f"  content-free P(true) = {p_true_cf:.3f}   "
          f"raw PPR={np.mean(p_true > 0.5):.3f}  CC PPR={np.mean(p_cc > 0.5):.3f}  "
          f"BC PPR={np.mean(p_bc > 0.5):.3f}")

    for inc, pr, pc, pb in zip(incidents, p_true, p_cc, p_bc):
        writer.writerow({
            "model": name, "dataset": inc["dataset"], "seed": seed,
            "prompt_style": prompt_style, "incident_id": inc["id"],
            "ground_truth_is_anomaly": inc["is_anomaly"],
            "p_true_raw": round(float(pr), 6),
            "p_true_cf": round(p_true_cf, 6),
            "p_true_cc": round(float(pc), 6),
            "p_true_bc": round(float(pb), 6),
            "pred_raw": bool(pr > 0.5),
            "pred_cc": bool(pc > 0.5),
            "pred_bc": bool(pb > 0.5),
        })

    del llm
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--prompt-style", choices=["zero_shot", "few_shot", "rag"], default="zero_shot")
    ap.add_argument("--data", default="data/incidents.jsonl")
    ap.add_argument("--models", nargs="*", default=None,
                    help="HF ids. Default: Qwen2.5-14B-Instruct and Mistral-Small-Instruct-2409")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    models = DEFAULT_MODELS if not args.models else [(m.split("/")[-1].lower(), m) for m in args.models]
    incidents = load_incidents(args.data)
    print(f"Loaded {len(incidents)} incidents")

    retriever = IncidentRetriever(incidents) if args.prompt_style == "rag" else None

    out_path = args.out or f"results/calib_seed{args.seed}_{args.prompt_style}.csv"
    Path("results").mkdir(exist_ok=True)
    fieldnames = ["model", "dataset", "seed", "prompt_style", "incident_id",
                  "ground_truth_is_anomaly", "p_true_raw", "p_true_cf",
                  "p_true_cc", "p_true_bc", "pred_raw", "pred_cc", "pred_bc"]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, hf_id in models:
            run_model(name, hf_id, incidents, args.prompt_style, args.seed, writer, retriever)
            f.flush()
    print(f"\nDone -> {out_path}")


if __name__ == "__main__":
    main()
