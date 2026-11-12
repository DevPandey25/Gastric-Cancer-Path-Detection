# GastricPath AI — Reproducibility & Execution Guide

This document provides complete, reproducible instructions for setting up the environment, verifying dataset integrity, executing research benchmarks, running automated test suites, and launching the unified research workbench.

---

## 1. Environment & System Requirements

### Hardware Requirements
- **CPU:** 4+ physical x86_64 cores (Intel Core i5/i7/i9 or AMD Ryzen 5/7/9). Tested on AMD Ryzen 5 5500U.
- **RAM:** Minimum 8 GB (16 GB recommended for whole-slide and foundation model feature caching).
- **Disk:** Minimum 5 GB free disk space (excluding raw whole-slide images).
- **GPU:** Optional. The platform is designed and validated for CPU execution using PyTorch CPU wheels. If CUDA is unavailable, all inference and research modules run natively on CPU with honest capability logging.

### Software Requirements
- **Operating System:** Windows 10/11, Ubuntu 20.04/22.04 LTS, or macOS 12+.
- **Python:** Python 3.10, 3.11, 3.12, or 3.13 (validated on Python 3.13.5).
- **Package Manager:** `pip` (or `uv`).

---

## 2. Environment Setup

### Native Virtual Environment (Primary Recommended Execution)
```bash
# Clone or navigate to the repository root
cd gastric-cancer-histopathology-dashboard-main

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
.venv\Scripts\activate.bat
# On Linux / macOS:
source .venv/bin/activate

# Upgrade pip and install pinned dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Default configuration values:
- `HOST=127.0.0.1`
- `PORT=8000`
- `APP_ENV=production`
- `DATABASE_URL=sqlite:///./gastric_path.db`
- `MODEL_CHECKPOINT=models/checkpoints/gastric_rnn_classifier.pt`
- `MINIMUM_CHECKPOINT_ACCURACY=0.80`
- `UPLOAD_DIR=uploads`

No external cloud API keys, proprietary tokens, or network credentials are required.

---

## 3. Baseline Checkpoint Bitwise Verification

The canonical baseline model checkpoint must remain bitwise identical to the historical release. Verify its SHA-256 digest prior to execution:

### PowerShell:
```powershell
(Get-FileHash -Algorithm SHA256 models\checkpoints\gastric_rnn_classifier.pt).Hash
```
### Bash:
```bash
sha256sum models/checkpoints/gastric_rnn_classifier.pt
```

**Expected Hash:**
`65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8`

---

## 4. Reproducing Research Pipelines & Benchmarks

Each original research phase has dedicated, self-contained execution scripts:

### Phase 2: Dataset Integrity & Perceptual Hash Duplicate Audit
```bash
python research/data/audit_duplicates.py
```
*Output:* `research/data/audit_report.json` (31,096 images scanned, 0 bitwise duplicates, 21 perceptual duplicate clusters).

### Phase 3: Classification Benchmark & Data Split Audit
```bash
python research/benchmarks/evaluate_baseline.py
```
*Output:* `research/splits/split_summary.json` (Leak-proof train: 21,766, val: 4,665, test: 4,665 patches across 8 classes).

### Phase 4.4: Reduced Foundation Model Benchmark (Owkin Phikon vs DINOv2)
```bash
python research/foundation_models/evaluate_probes.py
```
*Output:* `research/foundation_models/cache/subset10pct/` evaluation summaries ($N=467$ held-out test partition). Top performer: Owkin Phikon Mean-Patch Linear Probe (**Accuracy: 66.38%, Macro OvR AUROC: 0.9450, Macro F1: 66.54%, ECE: 0.0524, Brier: 0.4487**). Distinct from Phase 4.2 full-dataset DINOv2 evaluation (65.08%, $N=4,665$).

### Phase 5: Interpretability & Faithfulness Attribution
```bash
python research/explainability/compute_faithfulness.py
```
*Output:* `research/explainability/deletion_faithfulness_summary.json` (Deletion AUC: 0.284, Insertion AUC: 0.612 across Grad-CAM, Integrated Gradients, Occlusion, and ViT token rollout).

### Phase 6: 9-Class Segmentation Infrastructure Audit
```bash
python -c "from research.segmentation.unet import UNet; from research.segmentation.config import ONTOLOGY; print(f'Ontology classes: {len(ONTOLOGY)}, UNet channels: {UNet(in_channels=3, num_classes=9).out_conv.out_channels}')"
```
*Output:* Verifies 9 output channels (Background + 8 canonical classes) and data blockage disclosure (0 pixel-level masks in GCHTID).

### Phase 7: Classical Nuclei Detection & Morphology Pilot
```bash
python research/cellular/detect_candidates.py
```
*Output:* `research/cellular/morphology_summary.json` (64 patches, 3,895 candidate objects, classical Ruifrok HED-watershed).

### Phase 8: Spatial Graph Construction & GNN Comparison
```bash
python research/spatial_graph/train_eval_gnn.py
```
*Output:* `research/spatial_graph/results.json` (Symmetric kNN, $k=6$, 16-D features; Morphology MLP baseline: 23.75% balanced accuracy; GCN: 21.25%).

### Phase 9: Pure-Python Whole-Slide Image & MIL Processing
```bash
python research/wsi/evaluate_mil.py
```
*Output:* Level-0 grid coordinates, HSV Otsu tissue filtering, Attention MIL pooling; honestly documents zero clinical WSI files in GCHTID patch repository.

### Phase 10: Multimodal Clinical Association & miRNA Ranking
```bash
python research/multimodal/evaluate_multimodal.py
```
*Output:* `research/multimodal/results.json` (HMU-GC-Clinical $N=300$ complete cohort, 0 dropped rows, HER-2 IHC 2+/3+ exploratory category [24.67%], in-silico miRNA target ranking [11 high-consensus targets $\ge 7/8$ algorithms for `hsa-miR-497-5p`], individual identifiers withheld from application outputs).

### Phase 12: CPU/GPU Inference Latency & Resource Benchmark
```bash
python research/benchmarks/inference_benchmark.py
```
*Output:* `research/benchmarks/cpu_benchmark_results.json` (Load: 53.85 ms, Warm mean: 38.56 ms, p95: 50.08 ms, Throughput: 25.94 patches/s, Peak RSS: 237.63 MB).

---

## 5. Running the Automated Test Suite

Execute the full comprehensive regression test suite:

```bash
# Run the entire test suite (335+ passing tests)
pytest tests/ -v

# Run only the Phase 12 release tests
pytest tests/test_original_phase12.py -v

# Run with test coverage reporting
pytest --cov=app --cov=research tests/
```

All tests execute completely offline, require zero internet access, create temporary SQLite test databases, and clean up automatically.

---

## 6. Launching the Research Workbench Application

### Native Execution
```bash
# Start the Uvicorn ASGI server
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Open a browser and navigate to:
- **Interactive UI:** `http://127.0.0.1:8000`
- **Workbench Mode:** Click the **"Unified Research Workbench"** tab in the top header to access all 14 research modules.
- **Interactive API Documentation:** `http://127.0.0.1:8000/docs` (Swagger UI) or `http://127.0.0.1:8000/redoc`
- **System Health Probe:** `http://127.0.0.1:8000/api/health`

### Container Deployment (Docker / Docker Compose)
*Note: Docker configuration files (`Dockerfile`, `docker-compose.yml`) are fully authored. In environments where the Docker daemon/CLI is unavailable (such as the current development host), container build/runtime verification is marked unavailable. Use native execution above, which is fully verified and authoritative.*
```bash
# Build and run container with Docker Compose
docker compose up -d --build

# View container logs
docker compose logs -f

# Verify container health
curl -f http://localhost:8000/api/health

# Stop container
docker compose down
```

---

## 7. Artifact & Data Provenance Map

| Phase | Core Code Path | Authoritative Artifact / Cache Path |
|---|---|---|
| Phase 1 | `research/config.py`, `research/experiment_tracker.py` | `research/registry/models.json` |
| Phase 2 | `research/data/audit_duplicates.py` | `research/data/audit_report.json` |
| Phase 3 | `research/splits/create_splits.py`, `app/ml/inference.py` | `research/splits/split_summary.json` |
| Phase 4.4 | `research/foundation_models/evaluate_probes.py` | `research/foundation_models/cache/subset10pct/` |
| Phase 5 | `research/explainability/` | `research/explainability/original_phase5_summary.json` |
| Phase 6 | `research/segmentation/unet.py`, `annotations.py` | `research/segmentation/config.json` |
| Phase 7 | `research/cellular/detect_candidates.py` | `research/cellular/morphology_summary.json` |
| Phase 8 | `research/spatial_graph/` | `research/spatial_graph/results.json` |
| Phase 9 | `research/wsi/pyramid.py`, `tissue_mask.py`, `mil.py` | `research/wsi/mock_wsi_eval.json` |
| Phase 10 | `research/multimodal/` | `research/multimodal/results.json` |
| Phase 11 | `app/research_workbench.py`, `app/templates/index.html` | REST Endpoints at `/api/research/*` |
| Phase 12 | `research/benchmarks/inference_benchmark.py` | `research/benchmarks/cpu_benchmark_results.json` |

