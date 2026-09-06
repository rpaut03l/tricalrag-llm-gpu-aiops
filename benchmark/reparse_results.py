import glob
import json
import shutil
from pathlib import Path

import pandas as pd


def try_parse_json(text):
    """Extracts JSON object substring, robust to preamble/postamble text
    some models add (e.g. Mistral-Small prepending 'Example: ' before JSON)."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    candidate = text[start:end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return None


def reparse_file(path):
    df = pd.read_csv(path)
    before_parsed = df["pred_is_anomaly"].astype(str).str.lower().isin(["true", "false"]).sum()

    new_is_anomaly, new_severity, new_root_cause = [], [], []
    for raw in df["raw_output"]:
        parsed = try_parse_json(raw)
        new_is_anomaly.append(parsed.get("is_anomaly") if parsed else None)
        new_severity.append(parsed.get("severity") if parsed else None)
        new_root_cause.append(parsed.get("root_cause") if parsed else None)

    df["pred_is_anomaly"] = new_is_anomaly
    df["pred_severity"] = new_severity
    df["pred_root_cause"] = new_root_cause

    after_parsed = df["pred_is_anomaly"].notna().sum()
    return df, before_parsed, after_parsed, len(df)


def main():
    files = sorted(glob.glob("results/raw_results_seed*_*.csv"))
    if not files:
        print("No results files found matching results/raw_results_seed*_*.csv")
        return

    backup_dir = Path("results/backup_before_reparse")
    backup_dir.mkdir(exist_ok=True)

    print(f"{'File':<45} {'Before':>10} {'After':>10} {'Total':>8} {'Improvement':>12}")
    print("-" * 90)

    for f in files:
        shutil.copy(f, backup_dir / Path(f).name)
        df, before, after, total = reparse_file(f)
        df.to_csv(f, index=False)
        improvement = after - before
        print(f"{Path(f).name:<45} {before:>10} {after:>10} {total:>8} {improvement:>+12}")

    print(f"\nOriginals backed up to {backup_dir}/")
    print("Re-run 'python score_results.py' now to get corrected metrics.")


if __name__ == "__main__":
    main()
