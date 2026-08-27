"""
WHY RAG for RCA: raw zero-shot/few-shot prompting gives the model only
hand-written examples (prompts.py). Retrieval-Augmented Generation instead
retrieves the top-k most SIMILAR real past incidents (by embedding
similarity) from a labeled incident store, and includes their actual
ground-truth root cause/remediation as in-context precedent. This mirrors
how a real SRE would search a runbook or past-incident database before
diagnosing a new one — and it's a legitimate, citable technique (RAG),
not just an ablation for its own sake.

This module:
1. Embeds all labeled incidents using a lightweight sentence-transformer
2. Builds a FAISS index for fast nearest-neighbor retrieval
3. Given a NEW incident, retrieves top-k similar PAST incidents
   (excluding itself) to inject into the prompt as retrieved context

Note: to keep this a fair test, the retrieval index for evaluating incident
X must never include X itself (no leakage) — enforced via `exclude_id`.
"""

import json
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer
import faiss

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # small, fast, CPU-friendly — doesn't need GPU VRAM


class IncidentRetriever:
    def __init__(self, incidents, embed_model_name=EMBED_MODEL_NAME):
        self.incidents = incidents
        self.model = SentenceTransformer(embed_model_name)
        texts = [inc["log_window"] for inc in incidents]
        print(f"Embedding {len(texts)} incidents for retrieval index...")
        self.embeddings = self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)   # inner product on normalized vectors = cosine similarity
        self.index.add(np.array(self.embeddings, dtype=np.float32))

    def retrieve(self, query_log_window, exclude_id=None, top_k=3):
        query_emb = self.model.encode([query_log_window], normalize_embeddings=True)
        # retrieve extra in case we need to filter out exclude_id
        scores, idxs = self.index.search(np.array(query_emb, dtype=np.float32), top_k + 1)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            inc = self.incidents[idx]
            if exclude_id is not None and inc["id"] == exclude_id:
                continue
            results.append({"incident": inc, "similarity": float(score)})
            if len(results) >= top_k:
                break
        return results


def build_retrieval_context(retrieved):
    """Format retrieved past incidents as in-context precedent for the prompt."""
    if not retrieved:
        return "No similar past incidents found."
    lines = []
    for i, r in enumerate(retrieved, 1):
        inc = r["incident"]
        gt_label = "ANOMALY" if inc["is_anomaly"] else "NORMAL"
        lines.append(
            f"Retrieved Precedent {i} (similarity={r['similarity']:.2f}, dataset={inc['dataset']}):\n"
            f"Log window:\n{inc['log_window']}\n"
            f"Known outcome: {gt_label} — category: {inc['ground_truth_category']}"
        )
    return "\n\n".join(lines)


if __name__ == "__main__":
    # Quick sanity test
    with open("data/incidents.jsonl") as f:
        incidents = [json.loads(line) for line in f]

    retriever = IncidentRetriever(incidents)
    test_query = incidents[0]["log_window"]
    results = retriever.retrieve(test_query, exclude_id=incidents[0]["id"], top_k=3)
    print("\nTop matches for incident 0:")
    for r in results:
        print(f"  sim={r['similarity']:.3f} dataset={r['incident']['dataset']} "
              f"anomaly={r['incident']['is_anomaly']}")
