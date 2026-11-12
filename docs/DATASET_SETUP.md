# GastricPath AI — Dataset Setup & Acquisition Guide

This document describes the primary dataset utilized by GastricPath AI, its scientific provenance, file structure, and instructions for local acquisition.

---

## 1. Dataset Provenance & Overview

The primary benchmark dataset is the **HMU-GC-HE-30K** (Gastric Cancer Histopathology Tissue Image Dataset - GCHTID), released by Harbin Medical University (HMU):
- **Specimens:** Formalin-Fixed Paraffin-Embedded (FFPE) hematoxylin and eosin (H&E) stained gastric tissue sections.
- **Total Images:** 31,096 patch images ($224 \times 224$ pixels, 24-bit RGB PNG format).
- **Tissue Classes (8 categories):**
  1. `TUM` — Gastric Adenocarcinoma (Tumor epithelium)
  2. `STR` — Cancer-Associated Stroma
  3. `LYM` — Lymphocytes (Immune infiltration)
  4. `NOR` — Normal Gastric Mucosa
  5. `MUC` — Extracellular Mucus
  6. `DEB` — Necrotic Debris
  7. `MUS` — Smooth Muscle
  8. `ADI` — Adipose (Fat) Tissue
- **Supporting Clinical Metadata:** `HMU-GC-Clinical.csv` ($N=300$ complete patient demographic and staging records).
- **Biomarker Profile:** `Gastric.xlsx` (Consensus miRNA-target interactions for `hsa-miR-497-5p`).

---

## 2. Directory Layout & Expected Structure

To perform full retraining or comprehensive similarity search, extract the dataset into:
```
data/raw/HMU-GC-HE-30K/all_image/
├── ADI/
│   ├── ADI_1.png
│   └── ...
├── DEB/
├── LYM/
├── MUC/
├── MUS/
├── NOR/
├── STR/
└── TUM/
```

*Note: The raw image archive is excluded from GitHub version control to keep the repository lightweight and conform to repository hosting limits. The repository includes sample tile fixtures and the certified model checkpoint, so the application runs completely out-of-the-box even without the raw 31,096 images.*

---

## 3. Public Download Repositories

Legitimate public download repositories for academic research:
1. **Mendeley Data / Springer Nature:**  
   DOI: `10.17632/v5w884n43c.1`  
   URL: [https://data.mendeley.com/datasets/v5w884n43c/1](https://data.mendeley.com/datasets/v5w884n43c/1)
2. **Figshare Open Archive:**  
   DOI: `10.6084/m9.figshare.25954813`  
   URL: [https://figshare.com/articles/dataset/Gastric_Cancer_Histopathology_Tissue_Image_Dataset_GCHTID_/25954813](https://figshare.com/articles/dataset/Gastric_Cancer_Histopathology_Tissue_Image_Dataset_GCHTID_/25954813)
3. **Kaggle Mirror:**  
   URL: [https://www.kaggle.com/datasets/orvile/gastric-cancer-histopathology-tissue-image-dataset](https://www.kaggle.com/datasets/orvile/gastric-cancer-histopathology-tissue-image-dataset)

---

## 4. Manifest Preparation

Once the dataset is placed at `data/raw/HMU-GC-HE-30K/all_image/`, run the dataset manifest generator to verify tile integrity and generate reproducible train/val/test splits:
```powershell
python -m training.prepare_manifest
```

