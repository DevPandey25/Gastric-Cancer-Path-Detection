# GastricPath AI
### AI-Assisted Gastric Histopathology Research and Computational Pathology Workbench

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Tests Passing](https://img.shields.io/badge/tests-359%20passed-success.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Security: Offline First](https://img.shields.io/badge/security-offline--first-brightgreen.svg)](docs/SECURITY.md)

---

## 🔬 Project Overview

**GastricPath AI** is an open-source, comprehensive computational pathology and deep learning research platform specifically designed for **gastric adenocarcinoma histopathology analysis**. 

Operating as an offline, private scientific workstation, GastricPath AI unifies deep learning patch classification, foundation-model feature comparisons, interpretable gradient attributions, cellular morphology deconvolution, spatial graph neural networks, whole-slide image (WSI) tile processing, and evidence-grounded clinical research reporting into a single dark-mode pathology workstation interface.

The platform is anchored on the **HMU-GC-HE-30K** (Gastric Cancer Histopathology Tissue Image Dataset) benchmark (31,096 FFPE H&E patches across 8 classes) and validated across **359 automated test cases**.

---

## ⚠️ Important Research Limitations & Regulatory Disclaimer

> [!WARNING]
> **RESEARCH SOFTWARE ONLY — NOT A CLINICAL DIAGNOSTIC DEVICE**
>
> GastricPath AI is developed strictly for computational pathology methodology research, academic benchmarking, and educational investigation. It has **NOT** been approved, cleared, or certified as a medical device by the US FDA, European EMA/CE-IVD, China NMPA, or any regulatory body.
>
> - **Patch-Level Output:** Model classifications represent patch-level categorical estimates for an isolated $224 \times 224$ pixel tile. The predicted `TUM` probability reflects local tile morphology, **not** patient-level diagnosis or cancer staging.
> - **Model Confidence:** Softmax probability is a mathematical model output, not clinical certainty.
> - **Supervision Required:** Model outputs must never substitute for microscopic examination by a board-certified anatomical pathologist.
>
> For full details, see [docs/RESEARCH_LIMITATIONS.md](docs/RESEARCH_LIMITATIONS.md) and [docs/VALIDATION_AND_LIMITATIONS.md](docs/VALIDATION_AND_LIMITATIONS.md).

---

## ✨ Verified Capabilities & Modules

| Module / Capability | Implementation Description | Status |
| :--- | :--- | :---: |
| **01. Live Inference & Telemetry** | Interactive single-patch upload, 8-class prediction distribution, latency metrics, and verified telemetry | `Implemented` |
| **02. Dataset Explorer** | Distribution viewer across 31,096 patches in 8 classes (`TUM`, `STR`, `LYM`, `NOR`, `MUC`, `DEB`, `MUS`, `ADI`) | `Implemented` |
| **03. Custom CNN-BiLSTM-Attention** | Lightweight custom architecture (~850K params, 3.49 MB) optimized for fast CPU inference | `Implemented` |
| **04. Foundation Models** | Self-supervised comparative linear probing and nonlinear MLPs with **Phikon** and **DINOv2** | `Implemented` |
| **05. Model Explainability** | High-resolution Grad-CAM, Integrated Gradients, and sequential recurrent attention maps | `Implemented` |
| **06. Robustness Evaluation** | Perturbation stress tests against Gaussian noise, motion blur, and stain color jitter | `Implemented` |
| **07. Uncertainty & OOD** | Temperature scaling calibration, predictive entropy, and Mahalanobis distance OOD detection | `Implemented` |
| **08. Tissue Segmentation** | 9-class U-Net encoder-decoder infrastructure (Strict integrity: zero synthetic masks) | `Infrastructure Only` |
| **09. Cellular Morphology** | Ruifrok HED optical density color deconvolution and morphological watershed segmentation | `Implemented` |
| **10. Spatial Graphs & GNNs** | PyTorch Geometric spatial graph representations modeling cellular topology and Delaunay meshes | `Experimental` |
| **11. Whole-Slide Imaging (WSI)** | Pure-Python pyramidal tile reading, coordinate mapping, and Attention-MIL patch aggregation | `Implemented` |
| **12. Error Analysis** | Top false-positive and false-negative misclassification galleries with confusion matrix audits | `Implemented` |
| **13. Evidence Matrix** | 32-paper verified gastric cancer computational pathology literature database (RIS / BibTeX) | `Implemented` |
| **14. Clinical Research & PDF Workbench** | Case review, molecular biomarkers (HER-2, MSI/MMR, PD-L1, CLDN18.2), and ReportLab Platypus PDF export | `Implemented` |

---

## 🛠️ Technology Stack

- **Backend & REST Services:** [FastAPI](https://fastapi.tiangolo.com/) (asynchronous REST framework), Pydantic v2, Uvicorn, SQLite
- **Machine Learning & Vision:** [PyTorch](https://pytorch.org/), [Torch Geometric](https://pyg.org/), OpenCV, NumPy, Pandas, Scikit-learn
- **Document & PDF Generation:** [ReportLab](https://www.reportlab.com/) Platypus Flowable Architecture (Canvas-level rendering, headers, tables)
- **Frontend & Workstation UI:** Vanilla ES6+ JavaScript, Modern CSS3 with dark scientific workstation design tokens, SVG iconography (Zero bulky frontend frameworks, zero build steps)
- **Testing & Quality Assurance:** Pytest (359 passing automated test cases)

---

## 🚀 Quickstart Installation (Windows 11)

### 1. Clone the Repository
```powershell
git clone https://github.com/DevPandey25/Gastric-Cancer-Path-Detection.git
cd Gastric-Cancer-Path-Detection
```

### 2. Configure Virtual Environment & Install Dependencies
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Initialize Environment
```powershell
Copy-Item .env.example .env
```
*(GastricPath AI operates 100% offline and requires zero API keys or external services).*

### 4. Verify Model Checkpoint
GastricPath AI includes the certified baseline model checkpoint:
- Path: `models/checkpoints/gastric_rnn_classifier.pt`
- Verified SHA-256: `65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8`

```powershell
Get-FileHash models/checkpoints/gastric_rnn_classifier.pt -Algorithm SHA256
```

### 5. Launch the Web Application
```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Open **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)** in your browser.

Interactive API Documentation:
- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 🧪 Automated Testing

To run the complete test suite:
```powershell
python -m pytest tests/ -q
```
```
============================= 359 passed in 105s ==============================
```

---

## 📁 Repository Structure

```
Gastric-Cancer-Path-Detection/
├── app/                                 # FastAPI application and presentation views
│   ├── main.py                          # Application entry point and security middleware
│   ├── clinical_reporting_api.py        # Case dossier and PDF generation endpoints
│   ├── research_workbench.py            # 14 Research module endpoint implementations
│   ├── static/                          # Dark-mode styling and workstation logic (app.js, styles.css)
│   └── templates/                       # Jinja2 single-page application template
├── research/                            # 14 Computational pathology research modules
├── models/checkpoints/                  # Certified baseline model checkpoint (.pt)
├── data/                                # Clinical tables (HMU-GC-Clinical.csv, Gastric.xlsx)
├── docs/                                # Technical, architecture, security, and setup guides
├── tests/                               # Comprehensive automated test suite (359 tests)
├── .env.example                         # Safe configuration template
├── .gitignore                           # Research-grade exclusion rules
├── .gitattributes                       # Line ending normalization
├── CITATION.cff                         # Academic citation schema
├── CONTRIBUTING.md                      # Community guidelines
├── requirements.txt                     # Pinned project dependencies
└── README.md                            # Primary documentation
```

---

## 📊 Dataset & Model Download References

- **HMU-GC-HE-30K (GCHTID):** Mendeley Data [DOI: 10.17632/v5w884n43c.1](https://data.mendeley.com/datasets/v5w884n43c/1) | Figshare [DOI: 10.6084/m9.figshare.25954813](https://figshare.com/articles/dataset/Gastric_Cancer_Histopathology_Tissue_Image_Dataset_GCHTID_/25954813)
- See [docs/DATASET_SETUP.md](docs/DATASET_SETUP.md) for full instructions on extracting the 31,096 images to `data/raw/HMU-GC-HE-30K/all_image/`.

---

## 📄 Citation

If you use GastricPath AI in academic studies or computational pathology benchmarks, please cite:

```bibtex
@software{pandey2026gastricpath,
  author       = {Dev Prasad Pandey},
  title        = {GastricPath AI: AI-Assisted Gastric Histopathology Research and Computational Pathology Workbench},
  year         = {2026},
  version      = {1.0.0},
  url          = {https://github.com/DevPandey25/Gastric-Cancer-Path-Detection}
}
```

---

## 📜 Licensing & Permissions

This repository is maintained for research and educational purposes. Please consult the repository maintainer regarding licensing permissions and collaboration terms.
