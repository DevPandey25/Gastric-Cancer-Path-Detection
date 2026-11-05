# GastricPath AI — Phase 1 Implementation Report
### Research Infrastructure, Model Registry, Experiment Tracking & Reproducibility Verification

**Document ID:** `GP-REPORT-PHASE1-2026-V1`  
**Execution Phase:** Phase 1 (Research Infrastructure & Reproducibility)  
**Git Branch:** `gastricpath-ai-evolution`  
**Date:** November 2026  
**Status:** Completed & Empirically Verified (21/21 Automated Tests Passing)  

---

## 1. Executive Summary

Phase 1 establishes the foundational scientific and engineering infrastructure for **GastricPath AI**, transforming the original prototype dashboard into an explainable, reproducible, and verifiable computational pathology research platform.

In accordance with the **GastricPath AI Master Project Specification** (`GastricPath_AI_Research_and_Implementation_Blueprint.docx`), Phase 1 introduces:
1. **Centralized Strongly-Typed Configuration** (`research/config.py` + `research/configs/base_research.yaml`) using Pydantic V2.
2. **Multi-Layer Deterministic Reproducibility** (`research/reproducibility.py`) enforcing random seeding across Python, NumPy, and PyTorch, environment provenance logging, and explicit disclosure of nondeterministic operations.
3. **Local Audit-Grade Experiment Tracking** (`research/tracking/experiment_tracker.py`) managing experiment runs, time-series metrics (`metrics.jsonl`), artifact logging, and a queryable master index without cloud dependencies.
4. **Local Model Registry & Medical AI Model Cards** (`research/registry/model_registry.py` + `research/registry/model_metadata.py`) cataloging model provenance, chunked SHA-256 verification, evaluation metrics, and clinical limitations.
5. **Comprehensive Automated Test Suite** (`tests/`) containing 21 tests covering configuration, reproducibility, tracking, registry, image preprocessing, model tensor math, database integrity, and FastAPI ASGI regression.
6. **Zero-Regression Preservation & Performance Optimization**: The existing FastAPI dashboard, SQLite database (`app.db`), and trained PyTorch checkpoint (`gastric_rnn_classifier.pt`) remain 100% operational. A major performance bottleneck in `app/ml/inference.py` was resolved by transitioning prototype loading to an on-demand lazy pattern, slashing application import and startup time from **51.6 seconds down to 0.46 seconds** (a ~112x latency reduction).

---

## 2. Implemented Architecture & Component Specification

```
gastric-cancer-histopathology-dashboard/
├── research/
│   ├── __init__.py
│   ├── config.py                     # Strongly-typed Pydantic V2 configuration loader
│   ├── configs/
│   │   └── base_research.yaml        # Master research YAML configuration
│   ├── reproducibility.py            # Deterministic seeds, provenance extraction & disclosures
│   ├── tracking/
│   │   ├── __init__.py
│   │   └── experiment_tracker.py     # Local JSON/JSONL experiment tracker
│   └── registry/
│       ├── __init__.py
│       ├── model_metadata.py         # Medical AI Model Card schema (Pydantic V2)
│       ├── model_registry.py         # Local model catalog & chunked SHA-256 hashing
│       └── models.json               # Model registry database (registered baseline)
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # Pytest fixtures, test isolation & synthetic image generator
│   ├── test_config.py                # Configuration loading & override tests
│   ├── test_reproducibility.py       # Seed determinism & provenance tests
│   ├── test_tracker.py               # Experiment run lifecycle & metrics logging tests
│   ├── test_registry.py              # SHA-256 computation, CRUD & baseline registration tests
│   ├── test_ml_pipeline.py           # Preprocessing tensor shapes & RNN forward pass tests
│   ├── test_api_regression.py        # FastAPI ASGI routes regression tests
│   └── test_database.py              # SQLite PredictionRecord CRUD & stats tests
├── pytest.ini                        # Test runner configuration (excludes raw image directory)
└── docs/
    ├── REPOSITORY_AUDIT.md           # Phase 0 audit document
    ├── IMPLEMENTATION_STATUS.md      # Live capability tracking matrix
    └── PHASE1_IMPLEMENTATION_REPORT.md # This document
```

### 2.1 Centralized Configuration (`research/config.py` & `research/configs/base_research.yaml`)
- **Structure:** Defined by clean, typed sub-configurations:
  - `ProjectConfig`: Project name, version, authors, random seed (default `42`), and device (`auto`, `cpu`, `cuda`).
  - `PathsConfig`: Dynamic resolution of project root, data directory (`data/raw/HMU-GC-HE-30K`), uploads, models, checkpoints, runs, logs, and artifacts.
  - `DatasetConfig`: Dataset identifier (`GCHTID`), image dimensions ($224 \times 224 \times 3$), sequence length (16 patches of $56 \times 56$), 8 tissue classes (`ADI`, `DEB`, `LYM`, `MUC`, `MUS`, `NOR`, `STR`, `TUM`), and default split ratios ($70\% / 15\% / 15\%$).
  - `ModelConfig`: Architecture definitions, backbone CNN feature dim (128), BiLSTM hidden dim (128), number of recurrent layers (2), and attention mechanisms.
  - `TrainingConfig`: Batch size, epochs, learning rate, optimizer (`Adam`), loss (`CrossEntropyLoss`), scheduler, early stopping patience (5), and gradient clipping (1.0).
  - `EvaluationConfig`: Multi-class metrics (accuracy, macro/weighted F1, precision, recall, top-k accuracy, confusion matrix, AUROC, ECE).
- **Validation:** Implemented via Pydantic V2 with strict type validation, automated path normalization to absolute `Path` instances, and dynamic dictionary overrides (`load_research_config(overrides={"project.seed": 1337})`).

### 2.2 Reproducibility Engine (`research/reproducibility.py`)
- **Deterministic Seeding:**
  - `set_seed(seed=42, deterministic_torch=True)`: Synchronizes seeds across Python's `random`, NumPy's `np.random`, and PyTorch's `torch.manual_seed()` and `torch.cuda.manual_seed_all()`.
  - Sets `PYTHONHASHSEED` in runtime environment.
  - When PyTorch is available, enforces `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`. Configures `torch.use_deterministic_algorithms(True, warn_only=True)` to prevent silent unseeded CUDA or CPU kernel execution.
- **System Provenance Telemetry:**
  - `get_system_provenance()`: Captures Python version, PyTorch version, TorchVision version, CUDA availability and device name, OS platform, CPU architecture, logical CPU core count, timestamp, and the exact current Git commit hash via `subprocess`.
- **Scientific Disclosure:**
  - `get_nondeterministic_operations()`: Explicitly documents operations with residual nondeterminism on modern CPU/GPU architectures (e.g. atomic floating-point summations in CUDA convolutions, parallel multi-threaded CPU matrix operations, and asynchronous image I/O).

### 2.3 Experiment Tracking (`research/tracking/experiment_tracker.py`)
- **Storage Strategy:** 100% local, lightweight, file-based tracker. No third-party cloud telemetry or heavy servers required.
- **Directory Layout:** Each experiment creates an isolated folder under `model_train/runs/<experiment_id>/`:
  - `manifest.json`: Static metadata, configuration snapshot, Git hash, start/end timestamps, tags, and execution status (`running`, `completed`, `failed`).
  - `metrics.jsonl`: Append-only JSON Lines stream recording epoch-level or step-level loss, validation accuracy, F1 scores, learning rates, and timestamps.
  - `artifacts/`: Directory containing checkpoints, confusion matrices, ROC plots, and calibration diagrams.
- **Run Index:** A master catalog `model_train/runs/experiments_index.json` provides instantaneous querying of all past research runs by ID, architecture, or completion status.

### 2.4 Local Model Registry (`research/registry/`)
- **Schema (`model_metadata.py`):** Strongly-typed `ModelMetadata` model card capturing:
  - `model_id`: Unique identifier (e.g. `baseline_gastric_rnn`).
  - `name` & `architecture`: Human-readable name and formal architectural description.
  - `checkpoint_path`: Relative or absolute path to the `.pt` binary.
  - `checkpoint_hash`: Streaming SHA-256 cryptographic checksum of the checkpoint file.
  - `dataset_info`: Training dataset name, class list, image count, and split level.
  - `training_config`: Hyperparameters, optimizer, and loss function used.
  - `evaluation_metrics`: Exact validation accuracy, test accuracy, test loss, and source documentation.
  - `class_labels`: Explicit ordered list of 8 tissue class codes.
  - `input_preprocessing`: Resolution, patch decomposition, and normalization specifications.
  - `git_commit`: Checkpoint provenance commit hash.
  - `model_limitations`: Scientific boundaries and contraindications.
  - `status`: Deployment state (`baseline`, `candidate`, `production`, `archived`).
  - `registered_at`: ISO 8601 UTC timestamp.
- **Registry Implementation (`model_registry.py`):**
  - Manages `research/registry/models.json` with thread-safe atomic file writes.
  - `compute_file_sha256()`: Chunked streaming SHA-256 hash calculation (64 KB chunks), avoiding memory exhaustion on large model weights.
  - **Baseline Checkpoint Registered:** Baseline checkpoint `models/checkpoints/gastric_rnn_classifier.pt` verified with hash `65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8` and registered in `models.json`.

---

## 3. Automated Test Suite & Regression Verification

### 3.1 Test Architecture (`tests/`)
Pytest was configured via `pytest.ini` with `testpaths = tests` and `norecursedirs = data models model_train .git .vscode samples` to prevent scanning the 31,096 raw dataset images.

Test isolation was strictly enforced in `tests/conftest.py`:
- Production SQLite database (`app.db`) and upload directories (`data/uploads/`) are protected by monkeypatching the database engine, `SessionLocal`, and file paths to a temporary directory (`tmp_path`) for the duration of tests.
- Synthetic $224 \times 224$ RGB image fixture provides standardized image bytes without reading the disk dataset.

### 3.2 Test Results Summary

| Test File | Test Name | Purpose / Invariants Checked | Result |
|:---|:---|:---|:---:|
| `test_config.py` | `test_default_config_loading` | Base YAML loads without errors; 8 tissue classes present | **PASSED** |
| `test_config.py` | `test_config_overrides` | Dictionary override mechanisms correctly modify attributes | **PASSED** |
| `test_config.py` | `test_path_resolution` | Relative paths resolve to absolute `Path` instances | **PASSED** |
| `test_reproducibility.py` | `test_set_seed_reproducibility` | Seed control yields identical NumPy and PyTorch random tensors | **PASSED** |
| `test_reproducibility.py` | `test_system_provenance_contains_expected_keys` | Provenance dictionary contains commit, platform, python, torch | **PASSED** |
| `test_reproducibility.py` | `test_nondeterministic_operations_disclosure` | Operational disclosures list is non-empty and descriptive | **PASSED** |
| `test_tracker.py` | `test_experiment_tracker_lifecycle` | Run initiation, manifest creation, metric streaming, artifact logging, index update | **PASSED** |
| `test_registry.py` | `test_sha256_computation` | Streaming SHA-256 returns valid 64-char hex string | **PASSED** |
| `test_registry.py` | `test_model_registry_crud` | Register, retrieve, filter by status, and persist model cards | **PASSED** |
| `test_registry.py` | `test_register_actual_baseline_checkpoint` | Real checkpoint registration produces expected hash and 8 classes | **PASSED** |
| `test_ml_pipeline.py` | `test_image_preprocessing_shapes` | Preprocessing reshapes $224 \times 224 \times 3$ to $[16, 3, 56, 56]$ tensor | **PASSED** |
| `test_ml_pipeline.py` | `test_model_forward_pass_and_attention_shapes` | Model forward pass outputs shape $[1, 8]$ and attention weights sum to $1.0$ | **PASSED** |
| `test_ml_pipeline.py` | `test_cancer_index_calculation` | Weighted cancer index calculation matches formula and stays in $[0, 100]$ | **PASSED** |
| `test_api_regression.py` | `test_health_endpoint` | `GET /api/health` returns status `healthy` and model status dictionary | **PASSED** |
| `test_api_regression.py` | `test_dashboard_index_route` | `GET /` returns HTML document with 200 OK | **PASSED** |
| `test_api_regression.py` | `test_dashboard_stats_endpoint` | `GET /api/dashboard` returns valid metrics and distribution format | **PASSED** |
| `test_api_regression.py` | `test_predictions_endpoint` | `GET /api/predictions` returns list of prediction records | **PASSED** |
| `test_api_regression.py` | `test_predict_endpoint_with_image` | `POST /api/predict` processes image bytes and persists record | **PASSED** |
| `test_api_regression.py` | `test_predict_endpoint_missing_filename` | Missing filename correctly raises HTTP 400 error | **PASSED** |
| `test_database.py` | `test_database_crud` | SQLite `PredictionRecord` insertion, retrieval, and fields | **PASSED** |
| `test_database.py` | `test_database_stats_aggregation` | Aggregation queries correctly tally total cases, high-risk cases, and class dist | **PASSED** |

**Total Execution Time:** 26.33 seconds  
**Pass Rate:** 100% (21/21 passed)

---

## 4. Performance Optimization: Lazy Prototype Loading

### 4.1 Diagnosis of Original Bottleneck
During testing, importing `app.ml.inference` incurred an extreme latency overhead of **51.6 seconds**:
- **Root Cause:** In the original `InferenceService.__init__()`, the method `_build_prototypes()` was invoked unconditionally on startup.
- This method scanned `data/raw/HMU-GC-HE-30K/all_image/` across all 8 class subdirectories, loading, opening, and resizing **480 full-resolution histopathology images** from disk synchronously before the FastAPI server could even begin listening for incoming requests.

### 4.2 Architectural Solution
In `app/ml/inference.py`:
- `self.prototypes` was initialized to `None` in `__init__()`.
- Prototype construction was made lazy: it only executes in `_predict_with_prototypes()` if the trained PyTorch checkpoint is missing or fails to load, falling back to prototype inference only when needed.
- `status()` method was updated to report dataset availability without triggering the 480-image read.

### 4.3 Result
- Application import time dropped from **51.60s** to **0.46s** (~112x speedup).
- The FastAPI server starts instantaneously without file I/O locks.

---

## 5. Non-Destructive Preservation & Baseline Invariants

1. **FastAPI Application Integrity:**
   - Routes `GET /`, `GET /api/health`, `GET /api/dashboard`, `GET /api/predictions`, and `POST /api/predict` remain fully backward-compatible.
   - Migrated startup event to `@asynccontextmanager async def lifespan(app: FastAPI)` to eliminate Starlette/FastAPI deprecation warnings.
   - Resolved Starlette 0.38+ `TemplateResponse(request, ...)` parameter ordering.
2. **Database State (`app.db`):**
   - The production SQLite database `app.db` contains 4 existing historical prediction records.
   - All tests use an in-memory/temporary SQLite database, ensuring zero modifications to `app.db`.
3. **Model Weights & Architecture:**
   - `models/checkpoints/gastric_rnn_classifier.pt` is completely preserved with SHA-256 `65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8`.
   - The model architecture `GastricCancerRNN` (CNN + 2-layer BiLSTM + Additive Attention) remains intact and functional.

---

## 6. Dependency & Environment Analysis

### Added Dependencies:
- `pyyaml>=6.0.0`: Required for centralized YAML configuration parsing.
- `pytest>=8.0.0`: Required for running the automated unit and regression test suite.

### Existing Core Dependencies Preserved:
- `fastapi==0.115.11`
- `uvicorn==0.34.0`
- `torch==2.6.0` (installed runtime: `torch==2.9.1+cpu`)
- `torchvision==0.22.0`
- `pillow==11.1.0`
- `numpy==2.2.3`
- `sqlalchemy==2.0.38`
- `pydantic==2.10.6`

No major dependencies were upgraded or broken.

---

## 7. Git & Version Control Audit

- **Working Branch:** `gastricpath-ai-evolution` (verified branched off `master`).
- **Baseline Commit:** `117d81d` ("chore: baseline commit before GastricPath AI Phase 0 audit").
- **Audit Commit:** `873d4a0` ("docs: add Phase 0 repository audit and implementation status matrix").
- **Phase 1 Commit:** Ready to stage and commit all newly created research and test files.

---

## 8. Remaining Limitations & Recommendations for Phase 2

### Current Scientific Limitations:
1. **Unlinked Clinical & Molecular Datasets:** As documented in Phase 0, `HMU-GC-Clinical.csv` (300 patients) and `Gastric.xlsx` (3,711 miRNA records) do not share keys with the GCHTID image dataset (`data/raw/HMU-GC-HE-30K`). No synthetic linkages have been created.
2. **Patch vs. Patient-Level Splitting:** GCHTID patch filenames (`TUM_1.png` to `TUM_3887.png`) lack patient identifiers. Splitting across patches risks data leakage if multiple patches originated from the same patient's biopsy.
3. **Tumor Area Percentage Representation:** The existing dashboard computes a heuristic `cancer_index` via weighted class probabilities (`CANCER_WEIGHTS`). This must be clearly distinguished from anatomical tumor surface area measurements.

### Recommended Phase 2 Execution Plan:
With Phase 1 research infrastructure verified, **Phase 2 (Dataset Audit, Integrity Scanner, Near-Duplicate Detection & Split Manifests)** can proceed upon authorization:
1. Scan all 31,096 images for file corruption, dimension consistency ($224 \times 224$), channel depth (3 channels), and color distribution.
2. Run perceptual hashing (pHash) across classes to identify and document near-duplicate or overlapping patches.
3. Generate version-controlled, reproducible train/validation/test split manifests (`manifest_train.csv`, `manifest_val.csv`, `manifest_test.csv`) stored in `research/splits/`.
