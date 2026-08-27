"""
WHY this baseline matters: a benchmark paper that only compares LLMs against
each other invites the question "but is any of this better than a simple
classical detector?" DeepLog (Du et al. 2017) is THE canonical non-LLM log
anomaly baseline everyone in this field cites. Beating/matching it with an
LLM-based approach is a citable, meaningful claim; losing to it is also an
honest and useful finding.

This is a lightweight re-implementation (not the full DeepLog paper) using
an LSTM over log-key sequences, sufficient as a fair baseline reference point.
"""

import json
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
        return self.fc(out[:, -1, :])   # predict next log key

def tokenize_logs_to_keys(incidents):
    """Map each unique log line (roughly a 'log template') to an integer key."""
    vocab = {}
    sequences = []
    for inc in incidents:
        lines = inc["log_window"].split("\n")
        keys = []
        for line in lines:
            # crude templating: strip numbers to group similar lines
            template = "".join(c if not c.isdigit() else "#" for c in line)
            if template not in vocab:
                vocab[template] = len(vocab) + 1  # 0 reserved for padding
            keys.append(vocab[template])
        sequences.append(keys)
    return sequences, vocab

def train_deeplog(train_sequences, vocab_size, window=4, epochs=5):
    model = LogKeyLSTM(vocab_size + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        print(f"  epoch {epoch+1}/{epochs} loss={loss.item():.4f}")

    return model

def predict_anomaly(model, sequence, window=4, top_k=9):
    """DeepLog's core idea: if the actual next log key isn't in the model's
    top-k predicted keys, flag it as anomalous."""
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
    with open("data/incidents.jsonl") as f:
        incidents = [json.loads(line) for line in f]

    sequences, vocab = tokenize_logs_to_keys(incidents)
    vocab_size = len(vocab)
    print(f"Vocabulary size (unique log templates): {vocab_size}")

    # Train only on NORMAL sequences (DeepLog's core assumption:
    # model learns "normal" patterns, flags deviations)
    normal_idx = [i for i, inc in enumerate(incidents) if not inc["is_anomaly"]]
    train_sequences = [sequences[i] for i in normal_idx]

    print("Training DeepLog-style LSTM baseline on normal sequences only...")
    model = train_deeplog(train_sequences, vocab_size)

    preds, gt = [], []
    for inc, seq in zip(incidents, sequences):
        pred_anomaly = predict_anomaly(model, seq)
        preds.append(pred_anomaly)
        gt.append(inc["is_anomaly"])

    f1 = f1_score(gt, preds, zero_division=0)
    prec = precision_score(gt, preds, zero_division=0)
    rec = recall_score(gt, preds, zero_division=0)
    print(f"\nDeepLog-style baseline: F1={f1:.3f} Precision={prec:.3f} Recall={rec:.3f}")

    Path("../benchmark/results").mkdir(parents=True, exist_ok=True)
    with open("../benchmark/results/deeplog_baseline.json", "w") as f:
        json.dump({"f1": f1, "precision": prec, "recall": rec}, f, indent=2)

if __name__ == "__main__":
    main()
