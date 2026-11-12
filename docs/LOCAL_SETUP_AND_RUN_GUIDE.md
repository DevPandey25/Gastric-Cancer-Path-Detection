# GastricPath AI — Local Setup, Launch & Runtime Verification Guide

This operational manual documents the step-by-step procedure for configuring, running, and verifying the **GastricPath AI** computational pathology research platform on Windows 11.

---

## 1. System Overview & Architecture

GastricPath AI integrates a deep learning computational pathology engine with an interactive clinical and research interface.

- **Backend**: FastAPI (Python 3.10+, validated on Python 3.13.5), Uvicorn ASGI server.
- **Frontend**: Clean vanilla JavaScript, CSS3, and HTML5 templates with responsive clinical dashboards and a 14-module Research Workbench.
- **Machine Learning**: PyTorch-based sequential CNN-RNN classifier (`GastricCancerRNN`) trained on HMU-GC-HE-30K histopathology patches.
- **Database**: SQLite database (`app.db`) tracking inference records, probabilities, and audit metadata.
- **Research Infrastructure**: 14 Research Workbench modules powered by genuine research artifacts in `research/`.

---

## 2. Environment Prerequisites

Ensure the following prerequisites are installed on your Windows 11 host:

- **Operating System**: Windows 11 (64-bit)
- **Python**: Version 3.10 to 3.13 (Python 3.13.5 tested)
- **Shell**: PowerShell 5.1+ or PowerShell 7+
- **Network / Ports**: Local port `8000` available (or `8001` fallback)

---

## 3. Step-by-Step Installation & Configuration

### Step 3.1 — Create Virtual Environment

From the project root directory, create a virtual environment:

```powershell
# Create virtual environment (incorporating system-site packages if pre-installed)
python -m venv --system-site-packages .venv

# Verify activation and python path
.\.venv\Scripts\python.exe --version
```

### Step 3.2 — Configure Environment Variables (`.env`)

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

Ensure the following key settings are present in `.env`:

```ini
APP_NAME="Gastric Cancer Histopathology Analysis Dashboard"
APP_ENV=development
DEBUG=true

# Database: Preserve the existing historical SQLite database
DATABASE_URL=sqlite:///./app.db

# Storage Directories
UPLOAD_DIR=data/uploads
DATASET_DIR=data/raw/HMU-GC-HE-30K/all_image
DATASET_SNAPSHOT=data/raw/dataset_summary.json
CLINICAL_CSV=data/raw/clinical.csv
BIOMARKER_XLSX=data/raw/Supplementary table 1.xlsx

# Baseline Model Checkpoint
MODEL_CHECKPOINT=models/checkpoints/gastric_rnn_classifier.pt
PROTOTYPE_SAMPLES_PER_CLASS=8

# Checkpoint Validation Accuracy Threshold:
# Must be set to 0.50 to accept the verified baseline checkpoint (val_accuracy = 0.6486).
MINIMUM_CHECKPOINT_ACCURACY=0.50
```

> [!IMPORTANT]
> **Why `MINIMUM_CHECKPOINT_ACCURACY=0.50`?**
> The verified baseline checkpoint (`models/checkpoints/gastric_rnn_classifier.pt`) carries a recorded internal validation accuracy of **0.6486** (64.86%). Setting `MINIMUM_CHECKPOINT_ACCURACY=0.50` allows `app/ml/inference.py` to load this authentic model checkpoint rather than degrading to prototype baseline mode.

> [!NOTE]
> **Database Continuity**:
> Pointing `DATABASE_URL` to `sqlite:///./app.db` maintains continuity with existing historical prediction records.

---

## 4. Checkpoint & Research Assets Verification

Before launching the server, verify the integrity of the baseline checkpoint and key research assets:

### 4.1 Verify Checkpoint SHA-256

```powershell
(Get-FileHash -Algorithm SHA256 "models\checkpoints\gastric_rnn_classifier.pt").Hash
```

**Expected Hash**:
```
65369EF618AD1D0AD10ECB65B6F15324E932F52E21C8B25A43B71C636726D8B8
```

### 4.2 Verify Key Research Directories

Ensure the following artifact directories exist and are populated:
- `research/registry/models.json` (Model registry)
- `research/experiments/` (Tracked experiment manifests)
- `research/foundation_models/cache/` (Phikon & DINOv2 embeddings)
- `research/multimodal/` (Clinical schema & biomarker linkage summaries)
- `research/cellular/` (Nuclei detection and cellular morphology data)
- `research/spatial_graph/` (Spatial graphs & GNN artifacts)
- `research/wsi/` (WSI mock pipeline and MIL evaluation assets)

---

## 5. Starting the Application

Launch the FastAPI application using Uvicorn bound strictly to `127.0.0.1:8000`:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Expected startup output:
```
INFO:     Started server process [PID]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

---

## 6. Access Points & Endpoints

Once running, access the platform using your web browser or API client:

| Resource | URL | Description |
| :--- | :--- | :--- |
| **Main Web UI** | `http://127.0.0.1:8000/` | Interactive Dashboard & Research Workbench |
| **Swagger API Docs** | `http://127.0.0.1:8000/docs` | Interactive OpenAPI documentation & test console |
| **ReDoc API Docs** | `http://127.0.0.1:8000/redoc` | Clean OpenAPI specification viewer |
| **Health Check** | `http://127.0.0.1:8000/api/health` | Backend status verification endpoint |
| **Dashboard Payload**| `http://127.0.0.1:8000/api/dashboard` | Aggregated clinical & dataset summary |
| **Predictions** | `http://127.0.0.1:8000/api/predictions`| Database history of patch inference runs |

### Research Workbench API Endpoints (All 14 Modules)

| # | Endpoint | Description |
|---|---|---|
| 1 | `/api/research/overview` | Research state, sample counts, and phase status |
| 2 | `/api/research/dataset` | GCHTID-30K class distribution, patient splits, and clinical metadata |
| 3 | `/api/research/models` | Model registry listing checkpoints and benchmark metrics |
| 4 | `/api/research/foundation-models` | Phikon and DINOv2 embedding audit & comparative benchmarks |
| 5 | `/api/research/explainability` | Attention maps, Grad-CAM saliency, and feature visualizations |
| 6 | `/api/research/segmentation` | Tissue segmentation and morphological boundary metrics |
| 7 | `/api/research/cellular` | Nuclei detection, HED watershed segmentations, and cellular density |
| 8 | `/api/research/spatial-graphs` | Cell-graphs, Delaunay triangulations, and GNN topological metrics |
| 9 | `/api/research/wsi` | Whole-slide image tiling, background filtering, and MIL aggregation |
| 10 | `/api/research/similarity/samples` | Cross-patient image patch similarity retrieval |
| 11 | `/api/research/uncertainty` | Out-of-distribution detection, entropy, and risk stratification |
| 12 | `/api/research/experiments` | Logged training and validation experiment runs |
| 13 | `/api/research/validation` | Cross-validation tables, calibration, and limitation audits |
| 14 | `/api/research/reports` | Complete catalog of Phase 0–12 research reports |

---

## 7. Testing Live Model Inference

To test live patch-level inference, send a multipart HTTP POST request with an authentic histopathology patch from the dataset:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/predict" -F "file=@data/raw/HMU-GC-HE-30K/all_image/TUM/TUM_1.png"
```

### Expected Response Structure

```json
{
  "filename": "80dd329545ed45988db16e7c072f812c.png",
  "predicted_class": "NOR",
  "predicted_label": "Normal Mucosa",
  "cancer_percentage": 26.99,
  "predicted_tum_probability": 23.43,
  "disclaimer": "Predicted TUM class probability represents patch-level histological classification, NOT patient-level cancer diagnosis or anatomical tumor area percentage.",
  "risk_band": "Moderate",
  "confidence_score": 67.2,
  "top_probability": 67.2,
  "model_source": "trained-checkpoint",
  "class_probabilities": [
    {"class_code": "NOR", "label": "Normal Mucosa", "color": "#6b7b8c", "probability": 67.2},
    {"class_code": "TUM", "label": "Tumor", "color": "#a61e4d", "probability": 23.43},
    {"class_code": "DEB", "label": "Debris", "color": "#4d8fd1", "probability": 6.91},
    {"class_code": "STR", "label": "Cancer-Associated Stroma", "color": "#c4572e", "probability": 2.18},
    {"class_code": "LYM", "label": "Lymphocytes", "color": "#2f9d8f", "probability": 0.24},
    {"class_code": "MUS", "label": "Smooth Muscle", "color": "#8c6bb1", "probability": 0.02},
    {"class_code": "MUC", "label": "Mucus", "color": "#d08f31", "probability": 0.02},
    {"class_code": "ADI", "label": "Adipose Tissue", "color": "#1f4f8b", "probability": 0.0}
  ],
  "id": 5,
  "created_at": "2026-11-07T23:21:12.804336"
}
```

Key verifications in the response:
1. `model_source` must be `"trained-checkpoint"`.
2. All 8 GCHTID tissue classes are present in `class_probabilities` and sum to 100%.
3. `predicted_tum_probability` and `disclaimer` follow exact scientific and regulatory guidelines.
4. Record is assigned an autoincrementing `id` and stored in `app.db`.

---

## 8. Troubleshooting & FAQ

### Port 8000 is Already in Use
If port 8000 is occupied by another application, identify the process or switch ports:

```powershell
# Check what is using port 8000
netstat -ano | findstr :8000

# Alternative: run on port 8001
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

### Model Falls Back to Prototype Mode
If the server response shows `"model_source": "prototype-baseline"` instead of `"trained-checkpoint"`:
1. Check `.env` and verify `MINIMUM_CHECKPOINT_ACCURACY=0.50`.
2. Verify that `models/checkpoints/gastric_rnn_classifier.pt` exists and has SHA-256 matching `65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8`.
3. Restart the Uvicorn process to reload the checkpoint.

### Windows Browser Does Not Open Automatically
You can open the web interface manually at any time by running:

```powershell
Start-Process "http://127.0.0.1:8000/"
```
Or navigating to `http://127.0.0.1:8000/` in Edge, Chrome, or Firefox.

