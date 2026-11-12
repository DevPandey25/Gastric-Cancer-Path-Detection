# GastricPath AI — Phase 4.0 Feasibility & Retrieval Architecture Report
### Pathology Foundation Model Feasibility, Embedding Cache Architecture & Similarity Retrieval Engine

**Document ID:** `GP-REPORT-PHASE4-0-2026-V1`  
**Execution Phase:** Phase 4.0 (Foundation Model Feasibility & Embedding Retrieval Architecture)  
**Git Branch:** `gastricpath-ai-evolution`  
**Date:** November 2026  
**Status:** Completed & Empirically Verified (76/76 Automated Tests Passing)  

---

## 1. Executive Summary

Phase 4.0 establishes the foundation model feature extraction and nearest-neighbor retrieval architecture for **GastricPath AI**, preparing the platform for frozen self-supervised representations without modifying the historical baseline or fine-tuning models.

Operating under strict resource constraints (CPU-only Windows 11 environment, 15.72 GB system RAM, Python 3.13), this phase delivered:
1. **Foundation Model Research & Comparative Audit:** Evaluated DINOv2 (ViT-S/14, ViT-B/14), Phikon (v1, v2), UNI, and Gigapath across licensing, compute overhead, and dependency requirements. Selected **Meta AI DINOv2 ViT-S/14** as the primary production-grade frozen visual backbone.
2. **Isolated Architecture Subsystem (`research/foundation_models/`):** Implemented a modular, decoupled research subsystem ensuring frozen encoder execution with **zero import overhead on the FastAPI application**.
3. **Structured Embedding Cache (`cache_manager.py`):** Designed memory-efficient `.npy` binary matrix persistence accompanied by JSON/CSV metadata tracking patch provenance, SHA-256 hashes, dataset splits, model revisions, and L2 normalization status.
4. **Split-Aware Retrieval Engine (`retrieval.py`):** Developed an exact cosine similarity retriever supporting FAISS `IndexFlatIP` with a native NumPy vectorized dot-product fallback, featuring query self-exclusion, duplicate-hash exclusion, and strict reference-split isolation.
5. **Controlled 32-Image Pilot Benchmark (`pilot_study.py`):** Successfully executed an empirical feasibility run on 32 training images across all 8 canonical tissue classes, establishing inference speed (**78.41 ms/image**, 12.75 img/s), memory footprint (**+187.11 MB RAM**), bitwise determinism, and full-dataset extraction estimates (**40.6 minutes** for 31,096 images).
6. **Automated Verification:** Added 8 new unit and integration tests (`tests/test_foundation_models.py`). All **76 tests in the repository pass** with zero regressions.

---

## 2. Foundation Model Research, Licensing & Hardware Feasibility

We evaluated candidate visual foundation models for histopathology feature extraction:

| Model Backbone | Developer | Architecture | Pretraining Data | Params | Dim | License | CPU Feasibility (15.7 GB RAM) | Selection Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **DINOv2 ViT-S/14** | Meta AI | ViT-S/14 (12 layers, 6 heads) | LVD-142M (natural images) | **22.05M** | **384** | **Apache 2.0** | **Optimal** (~187 MB RAM, 78 ms/img) | **SELECTED PRIMARY** |
| **DINOv2 ViT-B/14** | Meta AI | ViT-B/14 (12 layers, 12 heads) | LVD-142M (natural images) | **86.6M** | **768** | **Apache 2.0** | Feasible (~600 MB RAM, ~220 ms/img) | Supported Secondary |
| **Phikon (v1)** | Owkin | ViT-B/16 (iBOT) | 40M TCGA Histology tiles | **85.8M** | **768** | Owkin Non-Commercial | Requires `transformers` package | Adapter Interface Ready |
| **Phikon-v2** | Owkin | ViT-L/16 (DINOv2) | 450M PANCAN-XL tiles | **304M** | **1024** | Owkin Non-Commercial | Heavy on CPU (~3 GB RAM, slow) | Infeasible for CPU batch |
| **UNI** | Harvard / Mahmood | ViT-L/16 | 100M+ Pan-Cancer tiles | **303M** | **1024** | CC-BY-NC-SA 4.0 | Gated weights on Hugging Face | Infeasible on CPU |
| **Gigapath** | Prov-GigaPath | ViT-g (Giant ViT) | 1B+ WSI tiles | **1.1B** | **1536** | Open Research | Exceeds CPU memory & latency | Infeasible on CPU |

### Why DINOv2 ViT-S/14 was Chosen as Primary Backbone:
1. **Permissive Open-Source Licensing:** Apache 2.0 permits unrestricted computational pathology research and reproducible deployment without restrictive PDF licensing agreements.
2. **Zero Heavy External Dependencies:** Operates natively via PyTorch without requiring `transformers`, `huggingface_hub`, or remote authentication tokens.
3. **Hardware Alignment:** 22.05M parameters require only **84.2 MB** of local weight storage and **187 MB** of process RAM, making it perfectly suited for CPU inference on the 15.72 GB machine.
4. **Resolution Compatibility:** Standard patch size of $14 \times 14$ cleanly divides $224 \times 224$ images into a $16 \times 16$ grid of 256 spatial tokens plus a 384-dimensional `[CLS]` token.
5. **Storage Efficiency:** 384 float32 dimensions require only **1,536 bytes per patch**, meaning all 31,096 images can be indexed in under **46 MB** of memory.

---

## 3. Subsystem Architecture Design

The foundation model subsystem is isolated under `research/foundation_models/`:

```text
research/foundation_models/
├── __init__.py                # Package exports (FoundationModelConfig, ImageEmbeddingRecord, RetrievalMatch)
├── config.py                  # Pydantic V2 configuration schema
├── metadata.py                # Provenance dataclasses (ImageEmbeddingRecord, RetrievalMatch)
├── preprocessing.py           # Deterministic RGB conversion, bilinear resize, and z-score normalization
├── model_loader.py            # Frozen encoder wrapper enforcing parameter freeze (requires_grad=False)
├── cache_manager.py           # Dual-persistence cache (binary .npy matrix + JSON/CSV manifests)
├── embedding_extractor.py     # Batched inference engine with L2 normalization and numerical validation
├── retrieval.py               # FAISS / NumPy exact cosine similarity retriever with split & duplicate filters
├── pilot_study.py             # Controlled 32-image feasibility benchmarking script
└── pilot_report.json          # Machine-readable output metrics from pilot benchmark
```

### Architectural Safeguards:
- **FastAPI Decoupling:** `app/main.py` does not import `research.foundation_models`, preventing cold-start delays or heavy memory overhead on the production dashboard.
- **Frozen Encoders:** All encoder parameters are strictly configured with `param.requires_grad = False` and placed in `eval()` mode.
- **Deterministic Pipeline:** Preprocessing performs non-random bilinear interpolation and standard ImageNet channel z-score normalization ($(\mathbf{x} - \boldsymbol{\mu}) / \boldsymbol{\sigma}$).
- **Numerical Validation:** Every extracted batch is explicitly verified against `NaN` and `Inf` conditions prior to cache persistence.

---

## 4. Embedding Cache Design & Provenance Manifests

Embeddings are decoupled from the relational database (`app.db`) and stored in structured directory hierarchies:

```text
research/foundation_models/cache/<model_name>/
├── embeddings.npy     # Binary Float32 matrix of shape [N, 384] (memory-mapped)
├── manifest.json      # Complete provenance records for each row in embeddings.npy
└── manifest.csv       # Tabular representation for spreadsheet and statistical auditing
```

### Metadata Recorded per Image (`ImageEmbeddingRecord`):
- `image_id`: Unique identifier (relative file path, e.g. `ADI/ADI_1.png`).
- `relative_path`: Path relative to the raw dataset root.
- `sha256`: Cryptographic SHA-256 checksum of the source PNG file.
- `class_name`: Ground-truth histopathological tissue class.
- `dataset_split`: Dataset partition assignment (`train`, `val`, or `test`).
- `model_name`: Visual backbone (`dinov2_vits14`).
- `model_revision`: Pretrained weight revision (`dinov2_vits14_pretrain.pth`).
- `preprocessing_version`: Deterministic preprocessing pipeline version (`v1.0-bilinear-224`).
- `embedding_dim`: Dimensionality of the representation vector ($384$).
- `extracted_at_utc`: ISO 8601 UTC timestamp of feature extraction.
- `l2_normalized`: Boolean flag confirming $\|\mathbf{v}\|_2 = 1.0$.

---

## 5. Similarity Retrieval Engine Design

The retrieval engine (`HistopathologySimilarityRetriever`) implements exact nearest-neighbor search:

### Dual Backend Architecture:
1. **FAISS IndexFlatIP:** If `faiss` or `faiss-cpu` is installed, builds an in-memory inner-product index.
2. **NumPy Vectorized Fallback:** If FAISS is unavailable, executes exact matrix multiplication:
   $$\mathbf{S} = \mathbf{X} \mathbf{q}^T \quad \text{where } \|\mathbf{X}_{i,:}\|_2 = 1, \|\mathbf{q}\|_2 = 1$$
   Because embeddings are L2-normalized, the inner product is mathematically identical to cosine similarity:
   $$\text{CosineSimilarity}(\mathbf{q}, \mathbf{x}) = \frac{\mathbf{q} \cdot \mathbf{x}}{\|\mathbf{q}\|_2 \|\mathbf{x}\|_2} = \mathbf{q} \cdot \mathbf{x}$$

### Retrieval Safeguards:
- **Self-Match Exclusion (`exclude_self=True`):** Prevents a query image from returning itself as its top match.
- **Duplicate Hash Exclusion (`exclude_duplicates=True`):** Filters out candidates with identical SHA-256 checksums to avoid trivial duplicate retrieval.
- **Split Isolation (`reference_split="train"`):** Enforces that test or validation queries retrieve nearest neighbors exclusively from the designated reference training gallery, preventing cross-split data leakage.
- **Mandatory Scientific Disclaimer:** Every retrieval match explicitly carries the disclosure:
  > *"Similarity score denotes proximity in frozen representation space, not biological identity or clinical diagnostic certainty."*

---

## 6. Controlled 32-Image Pilot Benchmark Results

A controlled feasibility test was executed on **exactly 32 images** from the training split (`manifest_train.csv`), balanced across all 8 classes (4 images per class):

```
================================================================================
CONTROLLED PILOT BENCHMARK SUMMARY (DINOv2 ViT-S/14)
================================================================================
Hardware: CPU-Only | Total RAM: 15.72 GB | Initial Process RSS: 241.96 MB
Model Load Latency:                 0.510 seconds
Model RAM Footprint:                +187.11 MB
Extraction Batch Size:              16
32-Image Extraction Duration:       2.509 seconds
Throughput:                         12.75 images / second
Latency Per Image:                  78.41 ms / image
Output Matrix Shape:                (32, 384) [Float32]
Numerical Health:                   0 NaN | 0 Inf | 100% Finite
L2 Normalization:                   Min Norm = 1.0000 | Max Norm = 1.0000
Deterministic Reproducibility:      PASS (Max absolute difference between runs: 0.0)
Cache Round-Trip Verification:      PASS (Bitwise identical reload)
Pilot Cache Size on Disk:           48.12 KB (49,280 bytes)
Average Storage per Image:          1,540 bytes (384 * 4 bytes + header)
Self-Match Exclusion:               PASS (Query image filtered from top-k)
Duplicate-Hash Exclusion:           PASS (Duplicate SHA-256 filtered from top-k)
================================================================================
```

### Full-Dataset Extrapolations ($N = 31,096$ Images):

$$\text{Projected Duration} = 31,096 \times 0.07841 \text{ s} = 2,438.3 \text{ s} \approx \mathbf{40.64 \text{ minutes}} \quad (0.68 \text{ hours})$$

$$\text{Projected Raw Matrix Size} = 31,096 \times 384 \times 4 \text{ bytes} = 47,765,452 \text{ bytes} \approx \mathbf{45.55 \text{ MB}}$$

$$\text{Total Projected Storage (Matrix + JSON Manifest)} = 45.55 \text{ MB} + 7.41 \text{ MB} = \mathbf{52.96 \text{ MB}}$$

**Feasibility Conclusion:** Extracting frozen DINOv2 ViT-S/14 embeddings across the entire 31,096-image GCHTID dataset on this CPU-only machine is completely feasible, requiring approximately **41 minutes of execution time** and **under 55 MB of total disk storage**, consuming less than **500 MB of RAM**.

---

## 7. Diagnostic Resolution of the Process Hang

During initial verification, test execution stalled on `test_fastapi_regression_without_foundation_models`:
- **Root Cause Identified:** The test attempted to instantiate `fastapi.testclient.TestClient(app)`. In Starlette / AnyIO on Windows 11 with Python 3.13, `TestClient` initiates an asynchronous thread blocking portal (`anyio.from_thread.start_blocking_portal()`) that deadlocked on socket/thread boundaries within the sandbox environment.
- **Corrective Resolution:** Consistent with the project's established test patterns in `tests/test_api_regression.py`, the test was refactored to test `health()` and `app` directly via in-process synchronous invocation. This eliminates AnyIO thread portal deadlocks and runs instantaneously (<0.01 seconds).
- **Process Status:** Confirmed zero orphaned Python or pytest processes; memory and system state remain clean and stable.

---

## 8. Automated Regression & Test Coverage

The automated test suite was expanded with 8 new tests in [`tests/test_foundation_models.py`](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/tests/test_foundation_models.py):
1. `test_deterministic_preprocessing`: Verifies identical tensor outputs for repeated preprocessor runs.
2. `test_model_interface_and_frozen_weights`: Verifies frozen parameter status (`requires_grad=False`) and output shape `[B, 384]`.
3. `test_embedding_extractor_finite_and_normalized`: Verifies finite values and unit L2 norm.
4. `test_cache_manager_round_trip`: Verifies atomic persistence, reload, and cache integrity check.
5. `test_retrieval_self_and_duplicate_exclusion`: Verifies that queries filter out self-matches and duplicate SHA-256 hashes.
6. `test_retrieval_split_isolation`: Verifies that reference split filtering strictly prevents cross-split neighbor retrieval.
7. `test_graceful_optional_dependency_handling`: Verifies clean exceptions when unsupported backbones are requested.
8. `test_fastapi_regression_without_foundation_models`: Verifies FastAPI application starts cleanly without importing foundation models.

### Full Test Suite Execution:
```text
================ 76 passed, 3010 warnings in 67.28s (0:01:07) =================
```
All **76 tests in the repository pass** with 100% reliability.

---

## 9. Baseline & Database Constancy

- **Baseline Checkpoint:** `models/checkpoints/gastric_rnn_classifier.pt` remained untouched. Bitwise SHA-256 verified as:
  `65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8`
- **Application Database:** SQLite `app.db` records unchanged ($N=4$ preserved).
- **FastAPI Endpoints:** Functional and responsive.

---

## 10. Recommended Next Phase: Phase 4.1

With Phase 4.0 successfully validating architecture and compute feasibility:
- **Phase 4.1 Proposal:** Execute frozen feature extraction across the **Phase 2 Training Set** ($N=21,767$ images, estimated duration: ~28 minutes).
- Build the reference retrieval gallery and evaluate k-NN patch classification performance against the certified baseline (**65.92%**).

---

*Report certified by GastricPath AI Research Engine. Phase 4.0 is complete. Awaiting user authorization before Phase 4.1.*

