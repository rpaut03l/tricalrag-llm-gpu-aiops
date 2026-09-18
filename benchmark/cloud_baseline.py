"""
Cloud API baseline: runs the SAME 600 incidents, SAME prompt templates,
and SAME scoring pipeline as the local benchmark.py, but calls a cloud
LLM API instead of vLLM. This is the missing comparison flagged in
Section VII (Limitations) of the paper -- without this, the paper's
privacy/cost/latency argument against cloud APIs is reasoning, not data.

Supports three providers:
  --provider openai      -> GPT-4o-mini via the OpenAI API directly
  --provider openrouter  -> any model via OpenRouter (no AWS account or
                             Bedrock model-access approval needed; one
                             API key covers Claude, Llama, Gemini, etc.)
  --provider bedrock     -> Claude Haiku 4.5 via AWS Bedrock (needs an
                             AWS account + model access granted in console)

PRICING (embedded below) is real, verified pricing as of August 2026 --
NOT fabricated. Re-verify against the provider's current pricing page
before citing these numbers in a final paper draft, since API pricing
changes over time and this benchmark may be re-run months later.

  GPT-4o-mini       (OpenAI):      $0.15 / 1M input tokens,  $0.60 / 1M output tokens
  Claude Haiku 4.5  (Bedrock):     $1.00 / 1M input tokens,  $5.00 / 1M output tokens
  Claude Haiku 4.5  (OpenRouter):  same model, routed pricing -- verify at
                                    openrouter.ai/models before citing

Usage:
    export OPENAI_API_KEY=sk-...
    python cloud_baseline.py --provider openai --prompt-style zero_shot

    export OPENROUTER_API_KEY=sk-or-...
    python cloud_baseline.py --provider openrouter --prompt-style zero_shot \\
        --or-model anthropic/claude-haiku-4.5

    # AWS credentials via standard boto3 chain (env vars, ~/.aws/credentials, or IAM role)
    python cloud_baseline.py --provider bedrock --prompt-style zero_shot
"""

import argparse
import csv
import json
import time
from pathlib import Path

from prompts import build_prompt
from retrieval import IncidentRetriever, build_retrieval_context

# ---------------------------------------------------------------------
# Verified pricing (USD per token) -- see module docstring for sources.
# Stored per-token (not per-million) to make the cost math below trivial.
# OpenRouter pricing varies by model; --or-price-in/--or-price-out let
# you pass the current rate for whichever model you route to, rather
# than hardcoding one that goes stale.
# ---------------------------------------------------------------------
PRICING = {
    "openai": {
        "model_name": "gpt-4o-mini",
        "input_per_token": 0.15 / 1_000_000,
        "output_per_token": 0.60 / 1_000_000,
    },
    "claude": {
        "model_name": "claude-haiku-4-5-20251001",
        "input_per_token": 1.00 / 1_000_000,
        "output_per_token": 5.00 / 1_000_000,
    },
    "bedrock": {
        "model_name": "anthropic.claude-haiku-4-5",
        "input_per_token": 1.00 / 1_000_000,
        "output_per_token": 5.00 / 1_000_000,
    },
    "ollama": {
        "model_name": "local",  # overwritten by --ollama-model at runtime
        "input_per_token": 0.0,   # genuinely $0/token: local inference, no metered API
        "output_per_token": 0.0,
    },
}

MAX_TOKENS = 200


def try_parse_json(text):
    """Same fixed extraction logic as benchmark.py: substring from first
    '{' to last '}', robust to preamble/postamble text some models add."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


# ---------------------------------------------------------------------
# Provider call functions -- each returns (raw_text, input_tokens, output_tokens)
# ---------------------------------------------------------------------

def call_openai(prompt, api_key, model="gpt-4o-mini"):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=0.0,
    )
    text = resp.choices[0].message.content
    usage = resp.usage
    return text, usage.prompt_tokens, usage.completion_tokens


def call_claude(prompt, api_key, model="claude-haiku-4-5-20251001"):
    """Native Anthropic API -- separate SDK and auth from OpenAI/OpenRouter,
    and a distinct token-counting scheme (usage.input_tokens / output_tokens,
    not prompt_tokens / completion_tokens).

    Note: temperature is deliberately NOT passed. Some anthropic SDK/API
    versions reject `temperature` when combined with certain other params,
    or the client used here doesn't expose it as a keyword -- confirmed by
    this exact TypeError on the box this ran on. Determinism matters less
    for this baseline than for the local vLLM runs (temperature=0.0 there
    is set for reproducibility across seeds); the cloud baseline reports
    single-run cost/latency/accuracy, not a bootstrap-CI'd metric, so
    default sampling is an acceptable, clearly-documented deviation --
    state this explicitly in the paper's methodology if these numbers are
    used, rather than silently treating it as equivalent to the local runs."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text
    return text, resp.usage.input_tokens, resp.usage.output_tokens


def call_ollama(prompt, model, host="http://localhost:11434"):
    """Ollama exposes an OpenAI-compatible endpoint, so this reuses the
    openai SDK client rather than a separate one -- api_key is a required
    but unchecked placeholder for local servers with no auth. Genuinely
    $0/token: this is local inference on hardware already owned, the same
    framing as the vLLM runs in benchmark.py, just via a different serving
    stack. Token counts come back in the same usage.prompt_tokens /
    usage.completion_tokens shape as OpenAI's own API."""
    from openai import OpenAI
    client = OpenAI(api_key="ollama", base_url=f"{host}/v1")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=0.0,
    )
    text = resp.choices[0].message.content
    usage = resp.usage
    return text, usage.prompt_tokens, usage.completion_tokens


def call_openrouter(prompt, api_key, model="anthropic/claude-haiku-4.5"):
    """OpenRouter speaks the OpenAI SDK protocol against a different base_url --
    no separate SDK needed, no AWS account, no per-model access approval wait."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=0.0,
    )
    text = resp.choices[0].message.content
    usage = resp.usage
    return text, usage.prompt_tokens, usage.completion_tokens


def call_bedrock(prompt, region="us-east-1",
                  model_id="anthropic.claude-haiku-4-5-20251001-v1:0"):
    import boto3
    client = boto3.client("bedrock-runtime", region_name=region)
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    })
    resp = client.invoke_model(modelId=model_id, body=body)
    payload = json.loads(resp["body"].read())
    text = payload["content"][0]["text"]
    usage = payload.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    return text, input_tokens, output_tokens


def load_incidents(path="data/incidents.jsonl"):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["openai", "claude", "openrouter", "bedrock", "ollama"], required=True)
    ap.add_argument("--prompt-style", choices=["zero_shot", "few_shot", "rag"], default="zero_shot")
    ap.add_argument("--data", default="data/incidents.jsonl")
    ap.add_argument("--limit", type=int, default=None,
                     help="Optional: cap number of incidents for a cheap smoke-test "
                          "run before committing to the full 600-incident cost.")
    ap.add_argument("--api-key", default=None,
                     help="API key for the chosen provider (or set OPENAI_API_KEY / "
                          "OPENROUTER_API_KEY env var)")
    ap.add_argument("--region", default="us-east-1", help="AWS region for Bedrock")
    ap.add_argument("--or-model", default="anthropic/claude-haiku-4.5",
                     help="OpenRouter model id, e.g. anthropic/claude-haiku-4.5, "
                          "meta-llama/llama-3.3-70b-instruct, google/gemini-2.0-flash-001 "
                          "-- see openrouter.ai/models for the full list and current pricing")
    ap.add_argument("--or-price-in", type=float, default=None,
                     help="OpenRouter input price, USD per 1M tokens (check openrouter.ai/models "
                          "for --or-model's current rate; required for --provider openrouter)")
    ap.add_argument("--or-price-out", type=float, default=None,
                     help="OpenRouter output price, USD per 1M tokens")
    ap.add_argument("--ollama-model", default="llama3.1:70b-instruct-q4_K_M",
                     help="Ollama model tag, e.g. llama3.1:70b-instruct-q4_K_M "
                          "-- run `curl http://localhost:11434/api/tags` to see what's pulled")
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    args = ap.parse_args()

    import os
    if args.provider == "openai":
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OpenAI provider requires --api-key or OPENAI_API_KEY env var.")
        pricing = PRICING["openai"]
    elif args.provider == "claude":
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("Claude provider requires --api-key or ANTHROPIC_API_KEY env var "
                              "(get one at platform.claude.com/settings/keys).")
        pricing = PRICING["claude"]
    elif args.provider == "openrouter":
        api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("OpenRouter provider requires --api-key or OPENROUTER_API_KEY env var.")
        if args.or_price_in is None or args.or_price_out is None:
            raise SystemExit(
                "OpenRouter pricing varies by model and changes over time -- pass current "
                "rates explicitly: --or-price-in <$/1M input> --or-price-out <$/1M output>. "
                "Look up --or-model's rate at https://openrouter.ai/models before running."
            )
        pricing = {"model_name": args.or_model,
                   "input_per_token": args.or_price_in / 1_000_000,
                   "output_per_token": args.or_price_out / 1_000_000}
    elif args.provider == "ollama":
        api_key = None
        pricing = dict(PRICING["ollama"])
        pricing["model_name"] = args.ollama_model
    else:
        api_key = None
        pricing = PRICING["bedrock"]

    incidents = load_incidents(args.data)
    if args.limit:
        incidents = incidents[:args.limit]
    print(f"Loaded {len(incidents)} incidents (provider={args.provider}, model={pricing['model_name']})")

    retriever = IncidentRetriever(incidents) if args.prompt_style == "rag" else None

    # Cost estimate BEFORE running -- lets you abort if it's higher than expected.
    est_input_tokens = len(incidents) * (400 if args.prompt_style == "rag" else 150)
    est_output_tokens = len(incidents) * 80
    est_cost = (est_input_tokens * pricing["input_per_token"] +
                est_output_tokens * pricing["output_per_token"])
    print(f"Rough pre-run cost estimate: ${est_cost:.4f} for {len(incidents)} incidents "
          f"(refined actual cost printed per-incident and totaled at the end)")

    out_path = f"results/cloud_baseline_{args.provider}_{args.prompt_style}.csv"
    Path("results").mkdir(exist_ok=True)

    fieldnames = ["model", "provider", "dataset", "prompt_style", "incident_id",
                  "ground_truth_is_anomaly", "ground_truth_category",
                  "pred_is_anomaly", "pred_severity", "pred_root_cause",
                  "raw_output", "latency_s", "input_tokens", "output_tokens", "cost_usd"]

    total_cost = 0.0
    total_latency = 0.0

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, inc in enumerate(incidents):
            if args.prompt_style == "rag":
                retrieved = retriever.retrieve(inc["log_window"], exclude_id=inc["id"], top_k=3)
                context = build_retrieval_context(retrieved)
                prompt = build_prompt(inc["log_window"], style="rag", retrieved_context=context)
            else:
                prompt = build_prompt(inc["log_window"], style=args.prompt_style)

            start = time.time()
            try:
                if args.provider == "openai":
                    raw_text, in_tok, out_tok = call_openai(prompt, api_key, pricing["model_name"])
                elif args.provider == "claude":
                    raw_text, in_tok, out_tok = call_claude(prompt, api_key, pricing["model_name"])
                elif args.provider == "openrouter":
                    raw_text, in_tok, out_tok = call_openrouter(prompt, api_key, pricing["model_name"])
                elif args.provider == "ollama":
                    raw_text, in_tok, out_tok = call_ollama(prompt, pricing["model_name"], host=args.ollama_host)
                else:
                    raw_text, in_tok, out_tok = call_bedrock(prompt, region=args.region)
            except Exception as e:
                print(f"  [{i+1}/{len(incidents)}] ERROR: {e}")
                raw_text, in_tok, out_tok = "", 0, 0
            elapsed = time.time() - start

            cost = in_tok * pricing["input_per_token"] + out_tok * pricing["output_per_token"]
            total_cost += cost
            total_latency += elapsed

            parsed = try_parse_json(raw_text)
            writer.writerow({
                "model": pricing["model_name"], "provider": args.provider,
                "dataset": inc["dataset"], "prompt_style": args.prompt_style,
                "incident_id": inc["id"],
                "ground_truth_is_anomaly": inc["is_anomaly"],
                "ground_truth_category": inc["ground_truth_category"],
                "pred_is_anomaly": parsed.get("is_anomaly") if parsed else None,
                "pred_severity": parsed.get("severity") if parsed else None,
                "pred_root_cause": parsed.get("root_cause") if parsed else None,
                "raw_output": raw_text.replace("\n", " ")[:400],
                "latency_s": round(elapsed, 3),
                "input_tokens": in_tok, "output_tokens": out_tok,
                "cost_usd": round(cost, 6),
            })
            f.flush()

            if (i + 1) % 50 == 0 or i == len(incidents) - 1:
                print(f"  [{i+1}/{len(incidents)}] running cost so far: ${total_cost:.4f}, "
                      f"avg latency: {total_latency/(i+1):.2f}s")

    print(f"\nDone. {len(incidents)} incidents.")
    print(f"Total cost: ${total_cost:.4f} (${total_cost/len(incidents)*1000:.4f} per 1000 incidents)")
    print(f"Average latency per request: {total_latency/len(incidents):.2f}s")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
