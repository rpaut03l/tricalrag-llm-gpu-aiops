"""
Runs the full benchmark: for each model, for each incident, generate an
RCA prediction via vLLM, record latency/throughput/VRAM, save per-incident
results tagged with dataset name, seed, and prompt style.

WHY seeds: running the same benchmark 3x with different data samples lets
us report mean +/- std dev instead of a single noisy number — this is what
separates a workshop-quality single-run result from a benchmark-paper-quality
statistically grounded one.
"""

import json
import time
import subprocess
import csv
import gc
import argparse
from pathlib import Path

import torch
from vllm import LLM, SamplingParams

from prompts import build_prompt
from retrieval import IncidentRetriever, build_retrieval_context

MODELS = [
#    {"name": "llama3.1-8b",   "hf_id": "meta-llama/Llama-3.1-8B-Instruct"},
     {"name": "qwen2.5-14b",   "hf_id": "Qwen/Qwen2.5-14B-Instruct"},
     {"name": "mistral-small", "hf_id": "mistralai/Mistral-Small-Instruct-2409"},
#    {"name": "llama3.3-70b-awq", "hf_id": "hugging-quants/Meta-Llama-3.3-70B-Instruct-AWQ-INT4"},
]

MAX_TOKENS = 200

def get_gpu_mem_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        )
        return max(int(x) for x in out.decode().strip().split("\n"))
    except Exception:
        return -1

def try_parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").replace("json", "", 1).strip()
    try:
        return json.loads(text)
    except Exception:
        return None

def load_incidents(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def run_model(model_cfg, incidents, writer, prompt_style, seed, retriever=None):
    print(f"\n=== {model_cfg['name']} | prompt={prompt_style} | seed={seed} ===")
    llm = LLM(model=model_cfg["hf_id"], dtype="auto",
              gpu_memory_utilization=0.85, max_model_len=8192)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    if prompt_style == "rag":
        prompts = []
        for inc in incidents:
            retrieved = retriever.retrieve(inc["log_window"], exclude_id=inc["id"], top_k=3)
            context = build_retrieval_context(retrieved)
            prompts.append(build_prompt(inc["log_window"], style="rag", retrieved_context=context))
    else:
        prompts = [build_prompt(inc["log_window"], style=prompt_style) for inc in incidents]

    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start
    peak_vram = get_gpu_mem_mb()

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0
    print(f"  {len(incidents)} incidents in {elapsed:.1f}s ({tok_per_sec:.1f} tok/s), VRAM ~{peak_vram}MB")

    for inc, out in zip(incidents, outputs):
        raw_text = out.outputs[0].text
        parsed = try_parse_json(raw_text)
        writer.writerow({
            "model": model_cfg["name"], "dataset": inc["dataset"], "seed": seed,
            "prompt_style": prompt_style, "incident_id": inc["id"],
            "ground_truth_is_anomaly": inc["is_anomaly"],
            "ground_truth_category": inc["ground_truth_category"],
            "pred_is_anomaly": parsed.get("is_anomaly") if parsed else None,
            "pred_severity": parsed.get("severity") if parsed else None,
            "pred_root_cause": parsed.get("root_cause") if parsed else None,
            "raw_output": raw_text.replace("\n", " ")[:400],
            "latency_total_s": round(elapsed, 2),
            "tokens_per_sec": round(tok_per_sec, 2),
            "peak_vram_mb": peak_vram,
        })

    del llm
    gc.collect()
    torch.cuda.empty_cache()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt-style", choices=["zero_shot", "few_shot", "rag"], default="zero_shot")
    ap.add_argument("--data", default="data/incidents.jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or f"results/raw_results_seed{args.seed}_{args.prompt_style}.csv"
    incidents = load_incidents(args.data)
    print(f"Loaded {len(incidents)} incidents across datasets: "
          f"{set(i['dataset'] for i in incidents)}")

    retriever = None
    if args.prompt_style == "rag":
        retriever = IncidentRetriever(incidents)

    Path("results").mkdir(exist_ok=True)
    fieldnames = ["model", "dataset", "seed", "prompt_style", "incident_id",
                  "ground_truth_is_anomaly", "ground_truth_category",
                  "pred_is_anomaly", "pred_severity", "pred_root_cause",
                  "raw_output", "latency_total_s", "tokens_per_sec", "peak_vram_mb"]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_cfg in MODELS:
            run_model(model_cfg, incidents, writer, args.prompt_style, args.seed, retriever=retriever)
            f.flush()

    print(f"\nDone -> {out_path}")

if __name__ == "__main__":
    main()
