# GastricPath AI — Phase 4.4.1 Pathology Encoder Pilot Report
### Small-Scale Empirical Validation of `owkin/phikon` on GCHTID

**Document ID:** `GP-PILOT-PHASE4-4-1-2026-V1`  
**Execution Phase:** Phase 4.4.1 (Small Pilot Validation)  
**Target Architecture:** Owkin Phikon (`owkin/phikon`, ViT-B/16, $d=768$)  
**Target Repository:** `gastric-cancer-histopathology-dashboard`  
**Git Branch:** `gastricpath-ai-evolution`  
**Base Commit:** `aa0613342fbe53af5bcf264669ec27ffde3d62f9`  
**Status:** Pilot Successful — Full-Dataset Extraction Authorized (Phase 4.4.2)  

---

## 1. Executive Summary & Pilot Mandate

Following the Phase 4.4.0 architectural feasibility confirmation, **Phase 4.4.1** conducted an empirical validation pilot of `owkin/phikon` on a balanced, stratified subset of $N = 64$ images ($8$ images per canonical tissue class: ADI, DEB, LYM, MUC, MUS, NOR, STR, TUM) drawn strictly from the training partition (`manifest_train.csv`).

### Core Findings:
1. **Zero Parameter Leakage & Offline Integrity:**
   - Pretrained model loaded successfully from offline cache (`local_files_only=True`).
   - Total parameter count: $86,389,248$. Trainable parameters: **$0$** ($100\%$ frozen; `requires_grad=False` verified across all tensor weights).
2. **Numerical & Geometric Correctness:**
   - Output CLS token shape: `[64, 768]`, $100\%$ finite (`isfinite() == True`), zero NaNs or Infs.
   - Output Mean Spatial Patch token shape: `[64, 768]`, $100\%$ finite.
   - L2 normalization: Maximum Euclidean norm deviation from $1.0$ is $< 1.2 \times 10^{-7}$ for all 64 vectors.
3. **Runtime & Throughput Benchmarking:**
   - Per-image CPU forward latency: **$218.8\text{ ms/image}$** ($4.57\text{ images/second}$ at batch size 8; $4.48\text{ images/second}$ at batch size 16/32).
   - Peak RSS memory: $852.4\text{ MB}$ (Starting at $416.3\text{ MB}$, model loaded at $750.0\text{ MB}$, peak during tensor forward pass at $852.4\text{ MB}$).
   - Memory headroom: Uses less than $1\text{ GB}$ of RAM, operating comfortably within the host's $15.72\text{ GB}$ system capacity.
4. **Linear Probing Compatibility:**
   - Both CLS and Mean Patch embeddings integrated seamlessly into Scikit-Learn multinomial logistic regression, converging with zero numerical issues.

---

## 2. Quantitative Pilot Metrics & Telemetry

| Metric / Parameter | Value Measured / Observed | Acceptance Threshold | Audit Verdict |
|:---|:---:|:---:|:---:|
| **Model Checkpoint SHA-256** | `d01642bfeb683c6b039622eca273e0a8e4208065a4595b63edba178c1c4cfa68` | Matches Hugging Face Hub | **Verified** |
| **Total Parameters** | $86,389,248$ | $86.4\text{M}$ (ViT-Base) | **Verified** |
| **Trainable Parameters** | **$0$** | Exactly $0$ | **Verified (Strictly Frozen)** |
| **Embedding Dimension ($d$)** | **768** | Exactly 768 | **Verified** |
| **Evaluated Pilot Cohort** | $64$ images ($8 \times 8$ classes) | $32 \le N \le 128$ balanced | **Verified** |
| **Data Partition Source** | `manifest_train.csv` | Train split only (no test data) | **Verified (Zero Test Leakage)** |
| **Model Load Time** | $0.52\text{ seconds}$ | $< 5.0\text{ s}$ | **Verified** |
| **Total Inference Time (64 img)**| $14.00\text{ seconds}$ | $< 30.0\text{ s}$ | **Verified** |
| **Per-Image Latency (CPU)** | **$218.76\text{ ms}$** | $< 400.0\text{ ms}$ | **Verified** |
| **Throughput (CPU)** | **$4.57\text{ images/sec}$** | $> 2.5\text{ img/s}$ | **Verified** |
| **Initial Process RSS RAM** | $416.3\text{ MB}$ | — | **Verified** |
| **Model Loaded RSS RAM** | $750.0\text{ MB}$ | $< 1.5\text{ GB}$ | **Verified** |
| **Peak Forward Pass RAM** | **$852.4\text{ MB}$** | $< 2.0\text{ GB}$ | **Verified (Very Low Footprint)** |
| **CLS Output Shape** | `(64, 768)` | `(N, 768)` | **Verified** |
| **Mean Patch Output Shape** | `(64, 768)` | `(N, 768)` | **Verified** |
| **Numerical Integrity** | $100\%$ Finite (0 NaN, 0 Inf) | $100\%$ Finite | **Verified** |
| **Max L2-Norm Deviation** | $1.19 \times 10^{-7}$ | $< 1.0 \times 10^{-5}$ | **Verified** |
| **Linear Probe Fit Accuracy** | $93.75\%$ (CLS) / $87.50\%$ (Patch) | Loss converges | **Verified** |

---

## 3. Full-Dataset Extraction Projections ($N = 31,096$)

Based on the pilot measurements on this Intel 10-core CPU:

- **Total Dataset Size:** $31,096$ images ($224 \times 224$ RGBA PNGs)
- **Batched Extraction Strategy:** Batch size $16$, periodic memory-mapped array flushing every $50$ batches ($800$ images)
- **Projected Throughput:** $\approx 4.5\text{ images/second}$
- **Projected Total Runtime:**
  $$\text{Runtime} = \frac{31,096\text{ images}}{4.5\text{ images/sec}} \approx 6,910\text{ seconds} \approx \mathbf{115\text{ minutes (1.9 hours)}}$$
- **Projected Storage Footprint:**
  - CLS Embedding Matrix: $(31096, 768)$ `float32` $= 95.53\text{ MB}$
  - Mean Patch Embedding Matrix: $(31096, 768)$ `float32` $= 95.53\text{ MB}$
  - Metadata Manifests (CSV/JSON): $\approx 15\text{ MB}$
  - **Total Disk Footprint:** $\approx \mathbf{206\text{ MB}}$ (Available on disk: $38.80\text{ GB}$)

---

## 4. Pilot Verdict & Authorization

**Pilot Verdict:** **PASS.**
All pre-conditions for Phase 4.4.2 (Full Dataset Frozen Feature Extraction) have been empirically verified:
1. Model weights are cached locally and load offline.
2. RAM consumption stays below $900\text{ MB}$.
3. Embeddings are numerically stable and strictly L2-normalized.
4. Linear probing converges as expected.

**Authorized Next Step:** Proceed to **Phase 4.4.2 (Full Dataset Frozen Feature Extraction)**.

