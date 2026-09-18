"""
Scores the output of benchmark_calibrated.py.

For each (model, dataset, prompt_style) it reports, for the three decision
rules -- raw logprob threshold, Contextual Calibration (CC), Batch
Calibration (BC) -- the F1 and the predicted-positive rate (PPR), with a
bootstrap 95% CI on F1 pooled over seeds. PPR far from the true 0.50 base
rate is the degenerate behaviour v1 documented.

This is the table that answers RQ2: does RAG fix calibration better than
the standard calibration methods, and does calibration alone (no
retrieval) get you most of the way there?

Usage:
    python score_calibration.py                # reads results/calib_*.csv
    python score_calibration.py --files results/calib_seed1_zero_shot.csv ...
"""

import argparse
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

METHODS = [("raw", "pred_raw"), ("cc", "pred_cc"), ("bc", "pred_bc")]
DEGENERATE_HI, DEGENERATE_LO = 0.85, 0.15


def to_bool(s):
    return s.astype(str).str.lower().map({"true": True, "false": False})


def bootstrap_f1(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        vals.append(f1_score(y[s], p[s], zero_division=0))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--out", default="results/calibration_summary.csv")
    args = ap.parse_args()

    files = args.files or sorted(glob.glob("results/calib_*.csv"))
    if not files:
        raise SystemExit("No results/calib_*.csv found. Run benchmark_calibrated.py first.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["y"] = to_bool(df["ground_truth_is_anomaly"])
    for _, col in METHODS:
        df[col] = to_bool(df[col])

    rows = []
    for (model, dataset, style), g in df.groupby(["model", "dataset", "prompt_style"]):
        row = {"model": model, "dataset": dataset, "prompt_style": style,
               "n": len(g), "seeds": g["seed"].nunique(),
               "p_true_cf": round(g["p_true_cf"].mean(), 3)}
        for name, col in METHODS:
            y, p = g["y"].values, g[col].values
            f1 = f1_score(y, p, zero_division=0)
            lo, hi = bootstrap_f1(y, p)
            ppr = float(np.mean(p))
            flag = "DEGENERATE" if (ppr > DEGENERATE_HI or ppr < DEGENERATE_LO) else "ok"
            row.update({f"f1_{name}": round(f1, 3),
                        f"f1_{name}_ci": f"[{lo:.3f}, {hi:.3f}]",
                        f"ppr_{name}": round(ppr, 3),
                        f"flag_{name}": flag})
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["model", "prompt_style", "dataset"])
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))

    # Macro view: how many configs are degenerate under each rule
    print("\n=== Degenerate configurations (of {}) ===".format(len(out)))
    for name, _ in METHODS:
        print(f"  {name}: {(out[f'flag_{name}'] == 'DEGENERATE').sum()}")

    out.to_csv(args.out, index=False)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
