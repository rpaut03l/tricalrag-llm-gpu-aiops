"""
Aggregates results across all seed runs, computes per-model per-dataset
metrics with bootstrap 95% confidence intervals.

WHY bootstrap CI: a single F1 number says nothing about how stable it is.
Reporting "F1 = 0.81 [0.76, 0.85]" (95% CI via bootstrap resampling) is
the standard practitioners expect in a benchmark paper and immediately
signals rigor to reviewers.
"""

import glob
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

N_BOOTSTRAP = 1000

def bootstrap_ci(y_true, y_pred, metric_fn, n=N_BOOTSTRAP, seed=0):
    rng = np.random.default_rng(seed)
    scores = []
    idx = np.arange(len(y_true))
    for _ in range(n):
        sample_idx = rng.choice(idx, size=len(idx), replace=True)
        yt = y_true.iloc[sample_idx] if hasattr(y_true, "iloc") else y_true[sample_idx]
        yp = y_pred.iloc[sample_idx] if hasattr(y_pred, "iloc") else y_pred[sample_idx]
        try:
            scores.append(metric_fn(yt, yp, zero_division=0))
        except Exception:
            continue
    lower, upper = np.percentile(scores, [2.5, 97.5])
    return round(lower, 3), round(upper, 3)

def load_all_results(pattern="results/raw_results_seed*_*.csv"):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No result files matching {pattern}. Run benchmark.py first.")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["pred_is_anomaly"] = df["pred_is_anomaly"].astype(str).str.lower().map({"true": True, "false": False})
    df["ground_truth_is_anomaly"] = df["ground_truth_is_anomaly"].astype(str).str.lower().map({"true": True, "false": False})
    return df

def summarize(df):
    rows = []
    # Group by model + dataset + prompt_style, aggregate across seeds
    for (model, dataset, style), g in df.groupby(["model", "dataset", "prompt_style"]):
        valid = g.dropna(subset=["pred_is_anomaly"])
        if len(valid) == 0:
            continue
        parse_rate = len(valid) / len(g)
        yt, yp = valid["ground_truth_is_anomaly"], valid["pred_is_anomaly"]

        f1 = f1_score(yt, yp, zero_division=0)
        f1_lo, f1_hi = bootstrap_ci(yt, yp, f1_score)
        prec = precision_score(yt, yp, zero_division=0)
        rec = recall_score(yt, yp, zero_division=0)
        acc = accuracy_score(yt, yp)

        rows.append({
            "model": model, "dataset": dataset, "prompt_style": style,
            "n_seeds": g["seed"].nunique(), "json_parse_rate": round(parse_rate, 3),
            "accuracy": round(acc, 3), "f1": round(f1, 3),
            "f1_ci_low": f1_lo, "f1_ci_high": f1_hi,
            "precision": round(prec, 3), "recall": round(rec, 3),
            "avg_tokens_per_sec": round(g["tokens_per_sec"].mean(), 1),
            "std_tokens_per_sec": round(g["tokens_per_sec"].std(), 1),
            "peak_vram_mb": int(g["peak_vram_mb"].max()),
        })
    return pd.DataFrame(rows).sort_values(["dataset", "f1"], ascending=[True, False])

def main():
    df = load_all_results()
    summary = summarize(df)
    summary.to_csv("results/summary_metrics.csv", index=False)
    print(summary.to_string(index=False))

    # Overall (macro-averaged across datasets) per model — the headline table
    macro = summary.groupby("model").agg(
        mean_f1=("f1", "mean"),
        mean_tokens_per_sec=("avg_tokens_per_sec", "mean"),
        max_vram_mb=("peak_vram_mb", "max"),
    ).reset_index().sort_values("mean_f1", ascending=False)
    macro.to_csv("results/macro_summary.csv", index=False)
    print("\n=== Macro-averaged across all 4 datasets ===")
    print(macro.to_string(index=False))

if __name__ == "__main__":
    main()
