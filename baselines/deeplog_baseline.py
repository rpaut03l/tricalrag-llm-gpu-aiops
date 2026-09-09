"""
WHY this baseline matters: a benchmark paper that only compares LLMs against
each other invites the question "but is any of this better than a simple
classical detector?" DeepLog (Du et al. 2017) is THE canonical non-LLM log
anomaly baseline everyone in this field cites.

This is a lightweight re-implementation using an LSTM over log-key
sequences, with a proper train/test split to avoid data leakage.
"""

import json
import math
import numpy as np
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score

class LogKeyLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        e = self.embed(x)
        out, _ = self.lstm(e)
        return self.fc(out[:, -1, :])

def tokenize_logs_to_keys(incidents):
    vocab = {}
    sequences = []
    for inc in incidents:
        lines = inc["log_window"].split("\n")
        keys = []
        for line in lines:
            template = "".join(c if not c.isdigit() else "#" for c in line)
            if template not in vocab:
                vocab[template] = len(vocab) + 1
            keys.append(vocab[template])
        sequences.append(keys)
    return sequences, vocab

def train_deeplog(train_sequences, vocab_size, window=4, epochs=30, batch_size=64, lr=1e-3):
    model = LogKeyLSTM(vocab_size + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X, y = [], []
    for seq in train_sequences:
        for i in range(len(seq) - window):
            X.append(seq[i:i + window])
            y.append(seq[i + window])
    if not X:
        raise ValueError("Not enough normal sequences to train DeepLog baseline.")

    X = torch.tensor(X, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)
    n = len(X)
    random_baseline_loss = math.log(vocab_size + 1)
    print(f"Training on {n} pairs, vocab_size={vocab_size}")
    print(f"Random-guessing baseline loss: {random_baseline_loss:.4f}")

    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss, n_batches = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X[idx], y[idx]
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / n_batches
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1}/{epochs} avg_loss={avg_loss:.4f}")

    return model

def predict_anomaly(model, sequence, window=4, top_k=9):
    if len(sequence) <= window:
        return False
    flags = []
    for i in range(len(sequence) - window):
        x = torch.tensor([sequence[i:i + window]], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)
        topk = torch.topk(logits, top_k, dim=-1).indices[0].tolist()
        actual_next = sequence[i + window]
        flags.append(actual_next not in topk)
    return any(flags)

def main():
    with open("../benchmark/data/incidents.jsonl") as f:
        incidents = [json.loads(line) for line in f]

    sequences, vocab = tokenize_logs_to_keys(incidents)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    normal_idx = [i for i, inc in enumerate(incidents) if not inc["is_anomaly"]]
    anomaly_idx = [i for i, inc in enumerate(incidents) if inc["is_anomaly"]]

    rng = np.random.default_rng(42)
    shuffled_normal = rng.permutation(normal_idx)
    split_point = int(len(shuffled_normal) * 0.8)
    train_normal_idx = shuffled_normal[:split_point].tolist()
    test_normal_idx = shuffled_normal[split_point:].tolist()

    print(f"Normal: {len(normal_idx)} total -> {len(train_normal_idx)} train / {len(test_normal_idx)} test")
    print(f"Anomalous (all held out): {len(anomaly_idx)}")

    train_sequences = [sequences[i] for i in train_normal_idx]
    print("Training on TRAIN-SPLIT normal sequences only...")
    model = train_deeplog(train_sequences, vocab_size)

    # Balance the eval set to match the LLM benchmark's 50/50 anomaly/normal
    # split -- otherwise predicted-positive-rate and F1 aren't comparable
    # across the two evaluations (an 83%-anomalous eval set makes even a
    # majority-class guess look artificially well-calibrated).
    eval_rng = np.random.default_rng(123)
    sampled_anomaly_idx = eval_rng.choice(anomaly_idx, size=len(test_normal_idx), replace=False).tolist()
    eval_idx = test_normal_idx + sampled_anomaly_idx
    print(f'Balanced eval set: {len(test_normal_idx)} normal + {len(sampled_anomaly_idx)} anomalous (50/50)')

    preds, gt = [], []
    for i in eval_idx:
        preds.append(predict_anomaly(model, sequences[i]))
        gt.append(incidents[i]["is_anomaly"])

    f1 = f1_score(gt, preds, zero_division=0)
    prec = precision_score(gt, preds, zero_division=0)
    rec = recall_score(gt, preds, zero_division=0)
    predicted_positive_rate = sum(preds) / len(preds)
    is_degenerate = predicted_positive_rate > 0.85 or predicted_positive_rate < 0.15

    print(f"Evaluated on {len(eval_idx)} held-out incidents")
    print(f"DeepLog baseline: F1={f1:.3f} Precision={prec:.3f} Recall={rec:.3f}")
    print(f"Predicted-positive rate: {predicted_positive_rate:.3f} ({'DEGENERATE' if is_degenerate else 'ok'})")

    Path("../benchmark/results").mkdir(parents=True, exist_ok=True)
    with open("../benchmark/results/deeplog_baseline.json", "w") as f:
        json.dump({
            "f1": f1, "precision": prec, "recall": rec,
            "predicted_positive_rate": predicted_positive_rate,
            "calibration_flag": "DEGENERATE" if is_degenerate else "ok",
            "n_train_normal": len(train_normal_idx),
            "n_eval_normal": len(test_normal_idx),
            "n_eval_anomalous": len(anomaly_idx),
        }, f, indent=2)

if __name__ == "__main__":
    main()
