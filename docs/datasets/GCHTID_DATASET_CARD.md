# Dataset Card: Gastric Cancer Histopathology Tile Image Dataset (GCHTID)

## 1. Dataset Summary
- **Dataset Name:** Gastric Cancer Histopathology Tile Image Dataset (GCHTID)
- **Primary Domain:** Computational Histopathology / Gastric Carcinoma Subtyping
- **Image Modality:** Brightfield Hematoxylin & Eosin (H&E) stained tissue tiles
- **Tile Resolution:** $224 \times 224$ pixels, 24-bit RGB PNG format
- **Magnification Level:** 20× optical equivalent ($0.50\ \mu\text{m/pixel}$ nominal)
- **Total Cataloged Images:** 31,096 image patches
- **License / Usage:** Open Academic Research Use Only; non-commercial clinical evaluation.

---

## 2. Taxonomy & Tissue Classes

GCHTID defines an 8-class histological classification taxonomy representing epithelial carcinoma, stromal elements, immune infiltration, and normal anatomical gastric tissue:

| Class Code | Class Name | Description | Count | Split (Train / Val / Test) |
|:---|:---|:---|:---|:---|
| **ADI** | Adipose Tissue | Mature adipocytes, clear lipid droplets with peripheral nuclei | 3,887 | 2,721 / 583 / 583 |
| **DEB** | Cellular Debris | Necrotic material, apoptotic debris, hemorrhage, amorphous proteinaceous fluid | 3,887 | 2,721 / 583 / 583 |
| **LYM** | Lymphocytes | Dense chronic inflammatory infiltrate, mature lymphocytic aggregates | 3,887 | 2,721 / 583 / 583 |
| **MUC** | Mucus | Extracellular mucin pools, glandular mucin secretion | 3,887 | 2,721 / 583 / 583 |
| **MUS** | Smooth Muscle | Gastric muscularis propria / mucosae spindle-shaped smooth muscle bundles | 3,887 | 2,721 / 583 / 583 |
| **NOR** | Normal Mucosa | Benign gastric foveolar epithelium, specialized parietal/chief cell glands | 3,887 | 2,721 / 583 / 583 |
| **STR** | Cancer-Associated Stroma | Reactive desmoplastic myofibroblastic stroma, fibrillar collagen matrix | 3,887 | 2,721 / 583 / 583 |
| **TUM** | Gastric Adenocarcinoma | Invasive adenocarcinoma epithelium, cytologically atypical tumor glands | 3,887 | 2,721 / 583 / 583 |
| **Total** | 8 Classes | Balanced patch collection | **31,096** | **21,766 / 4,665 / 4,665** |

---

## 3. Dataset Integrity & Deduplication Audit (Phase 2 Findings)
- **Integrity Scan:** 31,096 of 31,096 images successfully decoded without byte-level corruption or header defects.
- **Bitwise Duplicate Check:** 0 exact SHA-256 duplicate image files identified.
- **Perceptual Duplicate Audit (pHash):** 21 perceptual duplicate clusters identified (Hamming distance $\le 4$ on 64-bit perceptual hash).
- **Leak-Proof Partitioning:** Canonical train (70%), validation (15%), and test (15%) partitions were generated with strict leak-proof controls, ensuring that near-duplicate clusters do not span across train and evaluation splits.

---

## 4. Key Limitations & Disclosures
1. **Patch-Level Labeling Only:** Every image in GCHTID is a single $224 \times 224$ crop with a discrete patch-level categorical label. The dataset contains **0 pixel-level masks, 0 polygon contour annotations, and 0 QuPath GeoJSON exports**.
2. **Missing WSI Provenance:** Individual tiles are distributed without whole-slide image (WSI) identifiers, patient demographic metadata, anatomical biopsy source locations, or clinical staging outcomes.
3. **Tumor Area Caveat:** A patch classified as `TUM` reflects patch-level categorization by the model. It must **never** be interpreted as whole-organ tumor percentage or patient cancer diagnosis.

