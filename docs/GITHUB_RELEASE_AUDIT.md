# GastricPath AI — Repository & Security Release Audit
**Document ID:** `GASTRICPATH-AUDIT-2026-RELEASE-001`  
**Audit Timestamp:** `2026-11-12T23:25:00+05:30`  
**Project Workspace:** `c:\Users\KIIT0001\OneDrive\Desktop\gastric-cancer-histopathology-dashboard-main`  
**Git Branch:** `gastricpath-ai-evolution`  
**Target Repository:** `https://github.com/DevPandey25/Gastric-Cancer-Path-Detection`

---

## 1. Baseline Repository & Version Control Audit

### 1.1 Project Root & Working Tree
- **Root Directory:** `C:\Users\KIIT0001\OneDrive\Desktop\gastric-cancer-histopathology-dashboard-main`
- **Active Branch:** `gastricpath-ai-evolution`
- **Baseline Git Commit:** `aa06133` (`phase4.3.1: reconcile nonlinear probe metrics and certify results`)
- **Historical Lineage Anchor:** `117d81d` / `fabdae2` / `3636f2f` / `53a5788` / `8a16b25` / `ced00db` / `d930e3d` / `a821950` / `8dbdce6` / `aa06133`
- **Git Tags:** None previously set.
- **Git Remotes:** No remotes configured prior to release audit.

### 1.2 Checkpoint & Data Integrity
- **Certified Checkpoint Path:** `models/checkpoints/gastric_rnn_classifier.pt`
- **File Size:** `3,496,869 bytes` (~3.33 MB)
- **Verified SHA-256 Checksum:** `65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8` (Bit-for-bit certified match)
- **Tracking Status:** Tracked in Git history since baseline.
- **SQLite Ledger Integrity:** `app.db` contains 5 historical prediction records, preserved intact and explicitly gitignored.

---

## 2. Security, Credentials & Sensitive Data Audit

### 2.1 Secret Scanning Findings
- **Automated Scanner:** Comprehensive AST/regex pattern scan covering OpenAI keys, GitHub PATs, AWS keys, Private Keys, JWTs, MongoDB strings, and passwords.
- **Working Tree Matches:** **0 detected secrets**.
- **Commit History Matches:** **0 detected secrets**.
- **Credentials & API Keys:** ZERO hardcoded tokens. All application modules run strictly offline.
- **Configuration Templates:** `.env` is gitignored (`!! .env`). `.env.example` contains only safe local demonstration placeholders.

### 2.2 Patient & Clinical Privacy Protection
- **Identifiable Data Audit:** No patient names, hospital registration numbers, addresses, phone numbers, or unredacted EHR records exist in the repository.
- **Clinical Reporting Module:** Governed by Pydantic schema validation (`CaseDossier`, `PatientCase`, `HistopathologyReport`) with strict redaction guards. Missing clinical fields are rendered explicitly as `[Missing / Unspecified]`.

### 2.3 Dataset Hygiene & Licensing
- **Public Benchmark Reference:** HMU-GC-HE-30K (Harbin Medical University, 31,096 HE patches across 8 classes).
- **Storage Policy:** `data/raw/*` is gitignored. No private WSI slides or proprietary clinical image dumps are staged.
- **Upload Directory:** `data/uploads/*` is gitignored with `.gitkeep` anchor.

---

## 3. Large File Audit (> 100 MB GitHub Limit Check)

| Path / File Group | Size | Status | Action Taken |
| :--- | :--- | :--- | :--- |
| `models/checkpoints/gastric_rnn_classifier.pt` | 3.33 MB | Tracked | Retained (under 100 MB limit) |
| `research/experiments/**/checkpoints/*.pt` | ~282 MB each | Untracked | Ignored by `.gitignore` |
| `research/foundation_models/cache/full/phikon/*.npy` | ~95.5 MB each | Untracked | Ignored by `*.npy` |
| `research/clinical_reporting/output/*.pdf` | Up to 11.87 MB | Untracked | Ignored by `.gitignore` |
| Evaluation Contact Sheet Visualizations (`.png`) | 7 MB – 11.4 MB | Untracked | Permitted high-resolution plots |

---

## 4. Environment & Test Commands
- **Python Runtime:** Python 3.13.5 (Virtual Environment: `.\.venv\Scripts\python.exe`)
- **FastAPI Startup:** `.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000`
- **Test Suite Command:** `.\.venv\Scripts\python.exe -m pytest tests/ -q`
- **Health Check Endpoint:** `http://127.0.0.1:8000/api/health`
