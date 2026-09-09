"""
Two ablations that make the paper significantly stronger:

1. BATCH SIZE SCALING — shows how throughput scales on a single 96GB card.
   WHY: most published LLM-serving benchmarks assume multi-GPU clusters.
   Showing the scaling curve on ONE workstation card is a genuinely novel,
   citable data point for anyone considering single-node AIOps deployment.

2. QUANTIZATION IMPACT — FP16 vs 4-bit AWQ on the same model: does
   quantization hurt RCA accuracy enough to matter, in exchange for fitting
   larger models in memory?
   WHY: this is the exact tradeoff a practitioner deploying on a single
   card needs answered, and it's rarely tested for RCA-style structured
   generation tasks specifically (most quantization papers test perplexity
   or general QA, not structured JSON extraction).
"""

import json
import time
import gc
import csv
import argparse
from pathlib import Path

import torch
from vllm import LLM, SamplingParams

from prompts import build_prompt

def load_incidents(path, n=100):
    with open(path) as f:
        incidents = [json.loads(line) for line in f]
    return incidents[:n]

def run_batch_sweep(model_hf_id, incidents, batch_sizes, out_path):
    llm = LLM(model=model_hf_id, dtype="auto", gpu_memory_utilization=0.85, max_model_len=8192)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=200)

    rows = []
    for bs in batch_sizes:
        subset = incidents[:bs]
        prompts = [build_prompt(inc["log_window"]) for inc in subset]
        start = time.time()
        outputs = llm.generate(prompts, sampling_params)
        elapsed = time.time() - start
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0
        print(f"batch_size={bs}: {elapsed:.2f}s, {tok_per_sec:.1f} tok/s")
        rows.append({"batch_size": bs, "elapsed_s": round(elapsed, 2),
                      "tokens_per_sec": round(tok_per_sec, 1)})

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["batch_size", "elapsed_s", "tokens_per_sec"])
        writer.writeheader()
        writer.writerows(rows)

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Saved batch sweep -> {out_path}")

def run_quantization_comparison(fp16_hf_id, awq_hf_id, incidents, out_path):
    from score_results import bootstrap_ci
    from sklearn.metrics import f1_score
    import pandas as pd

    results = {}
    for label, hf_id in [("fp16", fp16_hf_id), ("awq_4bit", awq_hf_id)]:
        llm = LLM(model=hf_id, dtype="auto", gpu_memory_utilization=0.85, max_model_len=8192)
        sampling_params = SamplingParams(temperature=0.0, max_tokens=200)
        prompts = [build_prompt(inc["log_window"]) for inc in incidents]

        start = time.time()
        outputs = llm.generate(prompts, sampling_params)
        elapsed = time.time() - start

        preds = []
        for out in outputs:
            text = out.outputs[0].text.strip().strip("`")
            try:
                parsed = json.loads(text)
                preds.append(parsed.get("is_anomaly"))
            except Exception:
                preds.append(None)

        results[label] = {"preds": preds, "elapsed": elapsed}
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    gt = [inc["is_anomaly"] for inc in incidents]
    rows = []
    for label, r in results.items():
        valid_pairs = [(g, p) for g, p in zip(gt, r["preds"]) if p is not None]
        yt = [g for g, p in valid_pairs]
        yp = [p for g, p in valid_pairs]
        f1 = f1_score(yt, yp, zero_division=0) if yt else 0
        rows.append({"variant": label, "f1": round(f1, 3), "latency_s": round(r["elapsed"], 2)})

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved quantization comparison -> {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["batch_sweep", "quantization"], required=True)
    ap.add_argument("--data", default="data/incidents.jsonl")
    args = ap.parse_args()

    Path("results").mkdir(exist_ok=True)
    incidents = load_incidents(args.data, n=128)

    if args.mode == "batch_sweep":
        run_batch_sweep(
            model_hf_id="Qwen/Qwen2.5-14B-Instruct",
            incidents=incidents,
            batch_sizes=[1, 8, 32, 64, 128],
            out_path="results/ablation_batch_sweep.csv",
        )
    elif args.mode == "quantization":
        run_quantization_comparison(
            fp16_hf_id="Qwen/Qwen2.5-14B-Instruct",
            awq_hf_id="Qwen/Qwen2.5-14B-Instruct-AWQ",
            incidents=incidents,
            out_path="results/ablation_quantization.csv",
        )

if __name__ == "__main__":
    main()
