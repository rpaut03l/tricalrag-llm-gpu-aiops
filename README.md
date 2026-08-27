<div align="center">

# 🛡️ LogSentinel-RAG

### On-Premise, Retrieval-Augmented LLM Benchmark for AIOps Root Cause Analysis

*Can a single high-memory workstation GPU - running open-weight LLMs with retrieval over past incidents  -  match cloud APIs and classical ML for log-based anomaly detection and root cause analysis?*

[![Status](https://img.shields.io/badge/status-work--in--progress-yellow)]()
[![License](https://img.shields.io/badge/license-TBD-lightgrey)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![vLLM](https://img.shields.io/badge/inference-vLLM-orange)]()
[![GPU](https://img.shields.io/badge/GPU-RTX%20PRO%206000%20(96GB)-76B900)]()
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b)]()

[Overview](#-overview) • [Why This Matters](#-why-this-matters) • [Architecture](#-system-architecture) • [Benchmark Scope](#-benchmark-scope) • [Repo Structure](#-repository-structure) • [Reproduce](#-reproducing-results) • [Citation](#-citation)

</div>

---

## 📌 Overview

**LogSentinel-RAG** benchmarks whether open-weight LLMs, served entirely on a single **NVIDIA RTX PRO 6000 (96GB VRAM)** workstation via **vLLM**, can match cloud-API LLMs and classical ML for **root cause analysis (RCA)** on real production log data  -  with an added **retrieval-augmented generation (RAG)** layer that surfaces similar past incidents before the model reasons about a new one.

This is a **benchmark**, not a single experiment: it's built to be reused, extended, and cited by anyone evaluating on-prem LLM deployment for AIOps.

## 🎯 Why This Matters

| Problem | LogSentinel-RAG's Angle |
|---|---|
| Cloud LLM APIs for log analysis risk **data privacy leaks** and don't scale cost-wise with log volume | Everything runs **on-premise**, on hardware most infra teams could actually own |
| Classical anomaly detectors (LSTM-based) flag *that* something broke, not *why* | LLMs generate **natural-language root cause + remediation**, not just a flag |
| Most LLM-serving benchmarks assume **multi-GPU clusters** | This benchmarks what a **single workstation-class GPU** can realistically deliver |
| Naive prompting has no access to institutional incident history | **RAG** retrieves the top-k most similar past incidents as precedent before generating RCA |

## 🧩 What This Builds On

- **Original to this project**: the benchmark design, RAG-for-RCA application, multi-dataset + multi-baseline harness, and single-GPU-workstation framing.
- **[TriShieldRAG](https://arxiv.org/abs/2607.23838)** *(Mohanty, Patel, Yuvaraj, Chaudhary, Singhania, 2026)*  -  our own prior work on defense-in-depth for RAG pipelines; referenced here for paper structure and presentation style. See [Citation](#-citation) below.
- **[DeepLog](https://dl.acm.org/doi/10.1145/3133956.3134015)** *(Du et al., 2017)*  -  the classical LSTM-based log anomaly baseline this benchmark compares against.
- **[LogHub](https://github.com/logpai/loghub)**  -  source of all four real-world log datasets.
- **[vLLM](https://github.com/vllm-project/vllm)**  -  the inference serving engine powering every local model run.

## 🗺️ System Architecture

### 1. End-to-End Pipeline

```mermaid
flowchart LR
    A[("📂 Raw Logs<br/>BGL · HDFS · Thunderbird · OpenStack")] --> B["🔧 Unified Incident<br/>Windowing"]
    B --> C{"Prompt<br/>Strategy"}
    C -->|Zero-shot| D["Local LLM<br/>via vLLM"]
    C -->|Few-shot| D
    C -->|RAG| E["🔍 Retrieve Top-3<br/>Similar Past Incidents"] --> D
    D["🖥️ Local LLM<br/>(RTX PRO 6000)"] --> F["📋 Structured RCA Output<br/>JSON: anomaly · severity · root cause · remediation"]
    F --> G["📊 Bootstrap-CI Scoring"]
    H["☁️ Cloud API Baseline"] --> G
    I["📈 DeepLog Baseline<br/>(LSTM)"] --> G
    G --> J[("📄 Results:<br/>F1, Precision, Recall,<br/>Throughput, VRAM")]

    style A fill:#e8f0fe,stroke:#333
    style D fill:#fff3cd,stroke:#333
    style G fill:#d4edda,stroke:#333
    style J fill:#f8d7da,stroke:#333
```

**What this shows:** every incident flows through one of three prompting strategies before hitting the LLM. The RAG path adds a retrieval step that the zero-shot and few-shot paths skip entirely  -  this is the core experimental variable the benchmark measures.

---

### 2. RAG Retrieval  -  Detailed Sequence

```mermaid
sequenceDiagram
    participant Inc as New Incident
    participant Emb as Sentence-Transformer<br/>(all-MiniLM-L6-v2)
    participant Idx as FAISS Index<br/>(all past incidents)
    participant LLM as Local LLM (vLLM)
    participant Out as Structured RCA

    Inc->>Emb: Encode log_window
    Emb->>Idx: Query embedding (cosine similarity)
    Idx-->>Idx: Exclude self (no leakage)
    Idx->>Inc: Top-3 similar past incidents<br/>+ their known outcomes
    Inc->>LLM: Prompt = current incident<br/>+ retrieved precedent
    LLM->>Out: {is_anomaly, severity,<br/>root_cause, remediation}
```

**What this shows:** the retrieval step happens *before* the LLM ever sees the new incident  -  it's given real historical precedent (with known ground-truth outcomes) as context, similar to how an SRE would check a runbook before diagnosing a new alert. The "exclude self" step is critical for fairness: an incident can never retrieve itself as its own precedent.

---

### 3. Multi-Dataset Normalization

```mermaid
flowchart TD
    A1["BGL.log<br/>(supercomputer)"] --> N["Unified Schema<br/>{id, dataset, log_window,<br/>is_anomaly, category, component}"]
    A2["HDFS.log +<br/>anomaly_label.csv<br/>(distributed FS)"] --> N
    A3["Thunderbird.log<br/>(large cluster)"] --> N
    A4["OpenStack.log<br/>(cloud infra)"] --> N
    N --> B["Balanced Sampling<br/>150 incidents/dataset<br/>(50% anomaly / 50% normal)"]
    B --> C[("data/incidents.jsonl<br/>600 total incidents")]

    style N fill:#fff3cd,stroke:#333
    style C fill:#d4edda,stroke:#333
```

**What this shows:** four structurally different raw log formats get parsed by dataset-specific loaders, then converge into one common schema  -  this is what lets every downstream script (benchmark, scoring, ablations) stay dataset-agnostic.

---

### 4. Benchmark Execution Matrix

```mermaid
flowchart TD
    subgraph Models["Local Models (via vLLM)"]
        M1["Llama-3.1-8B"]
        M2["Qwen2.5-14B"]
        M3["Mistral-Small-22B"]
        M4["Llama-3.3-70B (AWQ 4-bit)"]
    end
    subgraph Baselines
        B1["☁️ Cloud API<br/>GPT-4o-mini"]
        B2["📈 DeepLog<br/>LSTM"]
    end
    subgraph Prompts["Prompt Styles"]
        P1["Zero-shot"]
        P2["Few-shot"]
        P3["RAG"]
    end
    subgraph Datasets
        D1["BGL"]
        D2["HDFS"]
        D3["Thunderbird"]
        D4["OpenStack"]
    end
    subgraph Seeds
        S1["Seed 1"]
        S2["Seed 2"]
        S3["Seed 3"]
    end

    Models --> Prompts --> Datasets --> Seeds --> R[("results/<br/>raw_results_*.csv")]
    Baselines --> Datasets

    style R fill:#f8d7da,stroke:#333
```

**What this shows:** the full combinatorial scope  -  4 local models × 3 prompt styles × 4 datasets × 3 seeds = **144 local-model runs**, plus baseline comparisons on top. This is what "benchmark" means here, not a single experiment.

---

### 5. Scoring & Statistical Pipeline

```mermaid
flowchart LR
    A[("raw_results_*.csv<br/>(all seeds, all prompt styles)")] --> B["Group by<br/>model × dataset × prompt_style"]
    B --> C["Compute F1, Precision,<br/>Recall, Accuracy"]
    C --> D["Bootstrap Resampling<br/>(1000 iterations)"]
    D --> E["95% Confidence<br/>Intervals"]
    E --> F[("summary_metrics.csv<br/>(per dataset)")]
    E --> G[("macro_summary.csv<br/>(averaged across datasets)")]
    F --> H["📄 Paper Tables<br/>& Figures"]
    G --> H

    style D fill:#fff3cd,stroke:#333
    style H fill:#d4edda,stroke:#333
```

**What this shows:** why the results are trustworthy  -  every reported F1 score comes with a bootstrap-derived confidence interval, not a single noisy number from one run.

---

### 6. Research Lineage

```mermaid
flowchart TD
    T["TriShieldRAG (2026)<br/>arXiv:2607.23838<br/>Defense-in-depth for RAG<br/>knowledge corruption"] -.->|"paper structure &<br/>retrieval-scoring principles"| L["LogSentinel-RAG (2026)<br/>this project<br/>On-prem LLM benchmark<br/>for AIOps RCA"]
    DL["DeepLog (2017)<br/>Du et al.<br/>LSTM log anomaly detection"] -.->|"classical baseline"| L
    LH["LogHub<br/>Zhu et al.<br/>Log dataset collection"] -.->|"dataset source"| L
    VL["vLLM (2023)<br/>Kwon et al.<br/>PagedAttention serving"] -.->|"inference engine"| L

    style L fill:#d4edda,stroke:#333,stroke-width:2px
    style T fill:#e8f0fe,stroke:#333
```

**What this shows:** LogSentinel-RAG is original in its combination and application (on-prem AIOps RCA), while drawing on established prior work for structure (TriShieldRAG), baseline comparison (DeepLog), data (LogHub), and infrastructure (vLLM)  -  see [What This Builds On](#-what-this-builds-on) for details.

## 📊 Benchmark Scope

|  | BGL | HDFS | Thunderbird | OpenStack |
|---|:---:|:---:|:---:|:---:|
| 🦙 Llama-3.1-8B | ✅ | ✅ | ✅ | ✅ |
| 🐦 Qwen2.5-14B | ✅ | ✅ | ✅ | ✅ |
| 🌬️ Mistral-Small-22B | ✅ | ✅ | ✅ | ✅ |
| 🦙 Llama-3.3-70B (AWQ 4-bit) | ✅ | ✅ | ✅ | ✅ |
| ☁️ Cloud API (GPT-4o-mini) | ✅ | ✅ | ✅ | ✅ |
| 📈 DeepLog (LSTM baseline) | ✅ | ✅ | ✅ | ✅ |

Each local model is evaluated under **three prompting strategies**  -  zero-shot, few-shot, and RAG (FAISS + sentence-transformer retrieval of top-3 similar past incidents)  -  across **3 random seeds**, with **bootstrap 95% confidence intervals** on every metric.

## 📁 Repository Structure

```
.
├── benchmark/
│   ├── loaders/multi_dataset_loader.py   # normalizes BGL/HDFS/Thunderbird/OpenStack
│   ├── prompts.py                        # zero-shot, few-shot, RAG prompt templates
│   ├── retrieval.py                      # FAISS + sentence-transformer retrieval index
│   ├── benchmark.py                      # main vLLM benchmark runner
│   ├── score_results.py                  # bootstrap CI scoring
│   └── ablation.py                       # batch-size sweep + quantization comparison
├── baselines/
│   └── deeplog_baseline.py               # classical LSTM log anomaly baseline
├── paper/
│   ├── main.tex                          # IEEE-format paper
│   ├── references.bib
│   ├── generate_diagrams.py              # figure generation
│   └── figures/
├── END_TO_END.md                         # full step-by-step run guide
└── README.md
```

## ⚙️ Hardware

- **GPU**: NVIDIA RTX PRO 6000, 96GB VRAM, single-node workstation
- **Inference**: vLLM (local models)  -  retrieval embeddings run on CPU, no VRAM contention

## 📚 Datasets

Sourced from **[LogHub](https://github.com/logpai/loghub)**: BGL, HDFS, Thunderbird, OpenStack. See each dataset's original license/terms before redistribution.

## 🚀 Reproducing Results

```bash
# 1. Environment setup
python3 -m venv logsentinel-env && source logsentinel-env/bin/activate
pip install -r benchmark/requirements.txt

# 2. Prepare dataset (after downloading LogHub datasets)
python benchmark/loaders/multi_dataset_loader.py --seed 42 --out benchmark/data/incidents.jsonl

# 3. Run benchmark (repeat for seeds 1,2,3 and all 3 prompt styles)
python benchmark/benchmark.py --seed 1 --prompt-style rag

# 4. Score results
python benchmark/score_results.py
```

📖 Full step-by-step instructions: [`END_TO_END.md`](./END_TO_END.md)

## 📖 Citation

If you use LogSentinel-RAG in your research, please cite:

```bibtex
@misc{rohitpatellogsentinelrag2026,
  title={LogSentinel-RAG: A Retrieval-Augmented Benchmark for On-Premise LLM-Based Root Cause Analysis in AIOps},
  author={Patel, Rohit},
  year={2026},
  howpublished={\url{https://github.com/rpaut03l/logsentinel-rag-llm-gpu-aiops}}
}
```

*(arXiv citation to be added upon submission.)*

This work also draws on our related prior paper on securing RAG pipelines:

```bibtex
@article{rohitpatel2026trishieldrag,
  title={TriShieldRAG: A Three-Ring Defense-in-Depth Framework Against Knowledge Corruption in Retrieval-Augmented Generation},
  author={Mohanty, Susil Kumar and Patel, Rohit and Yuvaraj, Kosuru and Chaudhary, Jeenal and Singhania, Disha},
  journal={arXiv preprint arXiv:2607.23838},
  year={2026}
}
```

## 📄 License

*(Choose a license  -  MIT or Apache 2.0 are common for benchmark code. Add a `LICENSE` file before making the repo public.)*

## 🙏 Acknowledgments

Built on [vLLM](https://github.com/vllm-project/vllm), [LogHub](https://github.com/logpai/loghub), and [FAISS](https://github.com/facebookresearch/faiss). Baseline methodology inspired by DeepLog (Du et al., 2017). Paper structure and presentation inspired by our own prior work, TriShieldRAG (arXiv:2607.23838).

---

<div align="center">

*Part of ongoing AIOps research toward AI-SPC 2026 (HiPC workshop) submission.*

</div>
