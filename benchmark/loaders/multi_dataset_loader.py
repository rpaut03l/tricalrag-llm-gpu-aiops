"""
WHY: A benchmark that only works on one dataset invites the single biggest
review criticism in ML/AIOps papers: "does this generalize?" This module
normalizes 4 different real log corpora into ONE schema so every downstream
script (benchmark.py, score_results.py) doesn't need to know which dataset
it's looking at.

Unified incident schema:
    {
        "id": int,
        "dataset": str,          # "bgl" | "hdfs" | "thunderbird" | "openstack"
        "log_window": str,       # raw log lines joined by \n
        "is_anomaly": bool,
        "ground_truth_category": str,
        "component": str,
    }

Each dataset has a different raw format, so each gets its own parser
function, but they all funnel into the same `build_incidents()` output.
"""

import re
import json
import random
from pathlib import Path

WINDOW_SIZE = 5
N_SAMPLES_PER_DATASET = 150   # keep each dataset's contribution balanced
SEED = 42


# ---------------------------------------------------------------------------
# BGL — space-separated, label is first column ('-' = normal)
# ---------------------------------------------------------------------------
def parse_bgl(raw_path):
    incidents = []
    with open(raw_path, errors="ignore") as f:
        lines = f.readlines()
    parsed = []
    for line in lines:
        parts = line.strip().split(" ", 9)
        if len(parts) < 10:
            continue
        label = parts[0]
        content = parts[9]
        parsed.append({"is_anomaly": label != "-", "label": label,
                        "content": content, "component": parts[7]})
    return _window(parsed, "bgl")


# ---------------------------------------------------------------------------
# HDFS — block-based; anomaly labels come from a separate anomaly_label.csv
# mapping block_id -> Normal/Anomaly (LogHub distributes this alongside HDFS.log)
# ---------------------------------------------------------------------------
def parse_hdfs(raw_path, label_csv_path):
    import csv
    label_map = {}
    with open(label_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_map[row["BlockId"]] = row["Label"]  # "Normal" or "Anomaly"

    block_re = re.compile(r"(blk_-?\d+)")
    parsed = []
    with open(raw_path, errors="ignore") as f:
        for line in f:
            m = block_re.search(line)
            if not m:
                continue
            blk_id = m.group(1)
            label = label_map.get(blk_id, "Normal")
            parsed.append({"is_anomaly": label == "Anomaly", "label": label,
                            "content": line.strip(), "component": "hdfs_block"})
    return _window(parsed, "hdfs")


# ---------------------------------------------------------------------------
# Thunderbird — same space-separated style as BGL
# ---------------------------------------------------------------------------
def parse_thunderbird(raw_path):
    incidents = []
    parsed = []
    with open(raw_path, errors="ignore") as f:
        for line in f:
            parts = line.strip().split(" ", 9)
            if len(parts) < 10:
                continue
            label = parts[0]
            parsed.append({"is_anomaly": label != "-", "label": label,
                            "content": parts[9], "component": parts[7] if len(parts) > 7 else "unknown"})
    return _window(parsed, "thunderbird")


# ---------------------------------------------------------------------------
# OpenStack — REAL FORMAT: three separate files, not one combined log.
#   - openstack_normal1.log, openstack_normal2.log: normal operation, no anomalies
#   - openstack_abnormal.log: contains injected anomalies tied to specific
#     VM instance UUIDs listed in anomaly_labels.txt
# Lines in the abnormal file are labeled anomalous; all normal-file lines
# are labeled normal. This matches how the dataset was originally released
# and used in DeepLog/LogAnomaly-style evaluations.
# ---------------------------------------------------------------------------
def parse_openstack(normal1_path, normal2_path, abnormal_path):
    parsed = []
    for path, is_anom in [(normal1_path, False), (normal2_path, False), (abnormal_path, True)]:
        with open(path, errors="ignore") as f:
            for line in f:
                if not line.strip():
                    continue
                parsed.append({
                    "is_anomaly": is_anom,
                    "label": "ANOMALY" if is_anom else "-",
                    "content": line.strip(),
                    "component": "openstack",
                })
    return _window(parsed, "openstack")


def _window(parsed, dataset_name):
    """Group consecutive parsed lines into fixed-size incident windows."""
    windows = []
    for i in range(0, len(parsed) - WINDOW_SIZE, WINDOW_SIZE):
        chunk = parsed[i:i + WINDOW_SIZE]
        any_anom = any(c["is_anomaly"] for c in chunk)
        gt_category = next((c["label"] for c in chunk if c["is_anomaly"]), "NORMAL")
        windows.append({
            "dataset": dataset_name,
            "log_window": "\n".join(c["content"] for c in chunk),
            "is_anomaly": any_anom,
            "ground_truth_category": gt_category,
            "component": chunk[0]["component"],
        })
    return windows


def sample_balanced(windows, n, seed):
    random.seed(seed)
    anomalies = [w for w in windows if w["is_anomaly"]]
    normals = [w for w in windows if not w["is_anomaly"]]
    n_each = n // 2
    sample = random.sample(anomalies, min(n_each, len(anomalies))) + \
             random.sample(normals, min(n_each, len(normals)))
    random.shuffle(sample)
    return sample


def build_all_datasets(config, seed=SEED):
    """
    config: dict mapping dataset name -> path(s), e.g.
        {
          "bgl": {"path": "loghub/full_datasets/BGL/BGL.log"},
          "hdfs": {"path": "loghub/full_datasets/HDFS/HDFS.log",
                   "label_csv": "loghub/full_datasets/HDFS/preprocessed/anomaly_label.csv"},
          "thunderbird": {"path": "loghub/full_datasets/Thunderbird_subset.log"},
          "openstack": {"normal1_path": "loghub/full_datasets/openstack_normal1.log",
                        "normal2_path": "loghub/full_datasets/openstack_normal2.log",
                        "abnormal_path": "loghub/full_datasets/openstack_abnormal.log"},
        }
    Returns combined, balanced, labeled incident list across all 4 datasets.
    """
    all_incidents = []
    idx = 0

    if "bgl" in config:
        w = parse_bgl(config["bgl"]["path"])
        for inc in sample_balanced(w, N_SAMPLES_PER_DATASET, seed):
            inc["id"] = idx; idx += 1
            all_incidents.append(inc)

    if "hdfs" in config:
        w = parse_hdfs(config["hdfs"]["path"], config["hdfs"]["label_csv"])
        for inc in sample_balanced(w, N_SAMPLES_PER_DATASET, seed):
            inc["id"] = idx; idx += 1
            all_incidents.append(inc)

    if "thunderbird" in config:
        w = parse_thunderbird(config["thunderbird"]["path"])
        for inc in sample_balanced(w, N_SAMPLES_PER_DATASET, seed):
            inc["id"] = idx; idx += 1
            all_incidents.append(inc)

    if "openstack" in config:
        w = parse_openstack(
            config["openstack"]["normal1_path"],
            config["openstack"]["normal2_path"],
            config["openstack"]["abnormal_path"],
        )
        for inc in sample_balanced(w, N_SAMPLES_PER_DATASET, seed):
            inc["id"] = idx; idx += 1
            all_incidents.append(inc)

    return all_incidents


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", default="data/incidents.jsonl")
    args = p.parse_args()

    # Verified real paths on the GPU machine (loghub/full_datasets/, downloaded
    # directly from Zenodo record 8196385 — see END_TO_END.md Phase 2 notes).
    config = {
        "bgl": {"path": "../loghub/full_datasets/BGL/BGL.log"},
        "hdfs": {"path": "../loghub/full_datasets/HDFS/HDFS.log",
                 "label_csv": "../loghub/full_datasets/HDFS/preprocessed/anomaly_label.csv"},
        "thunderbird": {"path": "../loghub/full_datasets/Thunderbird_subset.log"},
        "openstack": {"normal1_path": "../loghub/full_datasets/openstack_normal1.log",
                      "normal2_path": "../loghub/full_datasets/openstack_normal2.log",
                      "abnormal_path": "../loghub/full_datasets/openstack_abnormal.log"},
    }

    incidents = build_all_datasets(config, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for inc in incidents:
            f.write(json.dumps(inc) + "\n")

    from collections import Counter
    counts = Counter(i["dataset"] for i in incidents)
    print(f"Wrote {len(incidents)} incidents to {args.out}")
    print(f"Per-dataset breakdown: {dict(counts)}")
