# GastricPath AI — Installation & Quickstart Guide

This guide provides tested, step-by-step instructions for installing and running GastricPath AI on Windows 11, Linux, or macOS.

---

## 1. System Requirements

- **Operating System:** Windows 10/11 (64-bit), Ubuntu 20.04/22.04 LTS, or macOS 12+
- **Python Version:** Python `3.10`, `3.11`, `3.12`, or `3.13` (Tested and validated on Python `3.13.5`)
- **Hardware:**
  - Minimum 8 GB RAM (16 GB recommended for feature extraction and whole-slide tile processing)
  - Multi-core CPU (All modules run natively on CPU wheels; GPU is optional and not required)
  - At least 5 GB free disk space (excluding raw image dataset)

---

## 2. Step-by-Step Installation (Windows 11 PowerShell)

### Step 2.1 — Clone or Download Repository
```powershell
git clone https://github.com/DevPandey25/Gastric-Cancer-Path-Detection.git
cd Gastric-Cancer-Path-Detection
```

### Step 2.2 — Create & Activate Virtual Environment
```powershell
# Create virtual environment
python -m venv .venv

# Activate environment (PowerShell)
.\.venv\Scripts\Activate.ps1

# (If PowerShell script execution is restricted, run: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)
```

### Step 2.3 — Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2.4 — Configure Environment
```powershell
# Copy the safe example configuration template
Copy-Item .env.example .env
```
*Note: GastricPath AI operates completely offline. No third-party API keys, cloud tokens, or database passwords are required.*

---

## 3. Verify Model Checkpoint

The repository comes bundled with the certified baseline lightweight classifier checkpoint:
- Path: `models/checkpoints/gastric_rnn_classifier.pt` (~3.33 MB)
- Verified SHA-256 Checksum:
  ```
  65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8
  ```

To verify the checksum in PowerShell:
```powershell
Get-FileHash models/checkpoints/gastric_rnn_classifier.pt -Algorithm SHA256
```

---

## 4. Run the Application

Start the FastAPI application via Uvicorn:
```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open your web browser and navigate to:
- **Main Scientific Workstation:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Interactive Swagger Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Health Check:** [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

---

## 5. Run the Automated Test Suite

Execute the full automated test suite (359 test cases):
```powershell
python -m pytest tests/ -q
```
Expected output:
```
359 passed in ~100s
```

To run a specific module test:
```powershell
# Core API regression
python -m pytest tests/test_api_regression.py -q

# Clinical reporting & PDF generator tests
python -m pytest tests/clinical_reporting/test_clinical_reporting.py -q
```

---

## 6. Optional: Dataset Setup (HMU-GC-HE-30K)

For training experiments or full tile exploration, see [DATASET_SETUP.md](DATASET_SETUP.md) for downloading and placing the open HMU-GC-HE-30K histopathology dataset.

