# GastricPath AI — Phase 4.2.2 Verification Report
### Final Per-Class Metric Table Integrity & Mathematical Formulation Verification

**Document ID:** `GP-VERIFY-PHASE4-2-2-2026-V1`  
**Execution Phase:** Phase 4.2.2 (Final Per-Class Metric Table Integrity Correction)  
**Target Architectures:** Frozen Meta AI DINOv2 ViT-S/14 Linear Softmax Probes (CLS and Mean Spatial Patch)  
**Historical Baseline Checkpoint:** `models/checkpoints/gastric_rnn_classifier.pt`  
**Baseline SHA-256:** `65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8` (Bitwise Preserved)  
**Git Branch:** `gastricpath-ai-evolution`  
**Date:** November 2026  
**Status:** Certified, Mathematically Verified & Formally Reconciled  

---

## 1. Executive Summary & Verification Mandate

Phase 4.2.2 was executed to eliminate all remaining column ambiguities and establish strict mathematical consistency across the downstream linear probe per-class benchmark tables.

### Core Objectives Executed:
1. **Recompute all class metrics directly from canonical test predictions:**
   - [`linear_probe_cls_test_predictions.csv`](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/foundation_models/evaluation/linear_probe_cls_test_predictions.csv)
   - [`linear_probe_mean_patch_test_predictions.csv`](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/foundation_models/evaluation/linear_probe_mean_patch_test_predictions.csv)
2. **Standardize table schema to exactly 8 metric columns:**
   - CLS Precision, CLS Recall, CLS Specificity, CLS F1
   - Mean-Patch Precision, Mean-Patch Recall, Mean-Patch Specificity, Mean-Patch F1
   - Zero extraneous or ambiguous columns.
3. **Verify mathematical formula invariants for every canonical class:**
   - $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$
   - $\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$
   - $\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}$
   - $\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
   - Confusion matrix row sums equal class support ($\text{TP} + \text{FN} = \text{Support}$)
   - Confusion matrix column sums equal predicted counts ($\text{TP} + \text{FP} = \text{Predicted Count}$)
   - Total sample conservation ($\text{TP} + \text{FP} + \text{FN} + \text{TN} = 4,665$)
4. **Independent calculation of Macro and Support-Weighted averages:**
   - Computed from unrounded full-precision float64 metrics before rounding to percentiles.
   - Verified that neither aggregate assumes equivalence with the other.
5. **Implement automated regression tests:** Added [`tests/test_audit_phase4_2_2.py`](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/tests/test_audit_phase4_2_2.py).

---

## 2. Canonical Confusion Matrices & Confusion Invariants

Evaluated on the Phase 2 test split ($N = 4,665$ images, 8 canonical classes: `ADI, DEB, LYM, MUC, MUS, NOR, STR, TUM`):

### A. CLS Linear Probe Confusion Matrix ($C^*=20.0$):
*Rows: True Ground Truth; Columns: Model Predictions*

| True \ Pred | ADI | DEB | LYM | MUC | MUS | NOR | STR | TUM | Row Sum (Support) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ADI** | 446 | 13 | 0 | 77 | 26 | 17 | 1 | 3 | **583** |
| **DEB** | 5 | 330 | 50 | 33 | 28 | 35 | 45 | 57 | **583** |
| **LYM** | 3 | 36 | 472 | 2 | 3 | 25 | 15 | 27 | **583** |
| **MUC** | 76 | 38 | 5 | 338 | 24 | 55 | 26 | 22 | **584** |
| **MUS** | 30 | 39 | 11 | 20 | 387 | 20 | 65 | 11 | **583** |
| **NOR** | 31 | 45 | 49 | 45 | 6 | 281 | 41 | 85 | **583** |
| **STR** | 1 | 31 | 27 | 14 | 56 | 38 | 354 | 62 | **583** |
| **TUM** | 2 | 30 | 20 | 18 | 6 | 76 | 37 | 394 | **583** |
| **Col Sum (Predicted)** | **594** | **562** | **634** | **547** | **536** | **547** | **584** | **661** | **Total: 4665** |

### B. Mean Spatial Patch Linear Probe Confusion Matrix ($C^*=200.0$):
*Rows: True Ground Truth; Columns: Model Predictions*

| True \ Pred | ADI | DEB | LYM | MUC | MUS | NOR | STR | TUM | Row Sum (Support) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ADI** | 456 | 13 | 0 | 67 | 16 | 24 | 1 | 6 | **583** |
| **DEB** | 10 | 339 | 48 | 24 | 32 | 33 | 37 | 60 | **583** |
| **LYM** | 3 | 36 | 468 | 4 | 1 | 28 | 21 | 22 | **583** |
| **MUC** | 74 | 42 | 1 | 350 | 25 | 58 | 19 | 15 | **584** |
| **MUS** | 22 | 47 | 3 | 18 | 395 | 20 | 72 | 6 | **583** |
| **NOR** | 32 | 43 | 38 | 39 | 11 | 301 | 44 | 75 | **583** |
| **STR** | 0 | 28 | 28 | 13 | 72 | 36 | 349 | 57 | **583** |
| **TUM** | 2 | 35 | 19 | 17 | 4 | 75 | 53 | 378 | **583** |
| **Col Sum (Predicted)** | **599** | **583** | **605** | **532** | **556** | **575** | **596** | **619** | **Total: 4665** |

---

## 3. Class-by-Class Quadrant Derivations & Full-Precision Values

Every metric was computed from first-principles integer quadrant values ($\text{TP}, \text{FP}, \text{FN}, \text{TN}$):

### CLS Linear Probe ($C^*=20.0$):
1. **ADI:** $\text{TP}=446, \text{FP}=148, \text{FN}=137, \text{TN}=3934 \implies P=0.750842, R=0.765009, S=0.963743, F_1=0.757859$
2. **DEB:** $\text{TP}=330, \text{FP}=232, \text{FN}=253, \text{TN}=3850 \implies P=0.587189, R=0.566038, S=0.943165, F_1=0.576419$
3. **LYM:** $\text{TP}=472, \text{FP}=162, \text{FN}=111, \text{TN}=3920 \implies P=0.744479, R=0.809605, S=0.960314, F_1=0.775678$
4. **MUC:** $\text{TP}=338, \text{FP}=209, \text{FN}=246, \text{TN}=3872 \implies P=0.617916, R=0.578767, S=0.948787, F_1=0.597701$
5. **MUS:** $\text{TP}=387, \text{FP}=149, \text{FN}=196, \text{TN}=3933 \implies P=0.722015, R=0.663808, S=0.963498, F_1=0.691689$
6. **NOR:** $\text{TP}=281, \text{FP}=266, \text{FN}=302, \text{TN}=3816 \implies P=0.513711, R=0.481990, S=0.934836, F_1=0.497345$
7. **STR:** $\text{TP}=354, \text{FP}=230, \text{FN}=229, \text{TN}=3852 \implies P=0.606164, R=0.607204, S=0.943655, F_1=0.606684$
8. **TUM:** $\text{TP}=394, \text{FP}=267, \text{FN}=189, \text{TN}=3815 \implies P=0.596067, R=0.675815, S=0.934591, F_1=0.633441$

### Mean Spatial Patch Linear Probe ($C^*=200.0$):
1. **ADI:** $\text{TP}=456, \text{FP}=143, \text{FN}=127, \text{TN}=3939 \implies P=0.761269, R=0.782161, S=0.964968, F_1=0.771574$
2. **DEB:** $\text{TP}=339, \text{FP}=244, \text{FN}=244, \text{TN}=3838 \implies P=0.581475, R=0.581475, S=0.940225, F_1=0.581475$
3. **LYM:** $\text{TP}=468, \text{FP}=137, \text{FN}=115, \text{TN}=3945 \implies P=0.773554, R=0.802744, S=0.966438, F_1=0.787879$
4. **MUC:** $\text{TP}=350, \text{FP}=182, \text{FN}=234, \text{TN}=3899 \implies P=0.657895, R=0.599315, S=0.955403, F_1=0.627240$
5. **MUS:** $\text{TP}=395, \text{FP}=161, \text{FN}=188, \text{TN}=3921 \implies P=0.710432, R=0.677530, S=0.960559, F_1=0.693591$
6. **NOR:** $\text{TP}=301, \text{FP}=274, \text{FN}=282, \text{TN}=3808 \implies P=0.523478, R=0.516295, S=0.932876, F_1=0.519862$
7. **STR:** $\text{TP}=349, \text{FP}=247, \text{FN}=234, \text{TN}=3835 \implies P=0.585570, R=0.598628, S=0.939490, F_1=0.592027$
8. **TUM:** $\text{TP}=378, \text{FP}=241, \text{FN}=205, \text{TN}=3841 \implies P=0.610662, R=0.648370, S=0.940960, F_1=0.628952$

---

## 4. Independent Aggregation Verification (Macro vs. Support-Weighted)

Because the GCHTID Phase 2 test set contains 583 images for 7 classes and 584 images for 1 class (`MUC`), Macro (unweighted arithmetic mean $\frac{1}{8}\sum$) and Support-Weighted ($\sum w_i m_i$) averages are computed from independent mathematical formulas:

### A. CLS Probe Aggregates (Raw Float64 Precision):
* **Macro Precision:** $\frac{1}{8}\sum_{i=1}^8 P_i = 0.642297875 \implies \mathbf{64.23\%}$
* **Weighted Precision:** $\frac{\sum P_i \cdot \text{Supp}_i}{4665} = 0.642293128 \implies \mathbf{64.23\%}$ ($\Delta = -0.0000047$)
* **Macro Recall:** $\frac{1}{8}\sum_{i=1}^8 R_i = 0.643529500 \implies \mathbf{64.35\%}$
* **Weighted Recall:** $\frac{\text{Total TP}}{4665} = \frac{3002}{4665} = 0.643515541 \implies \mathbf{64.35\%}$ ($\Delta = -0.0000140$)
* **Macro Specificity:** $\frac{1}{8}\sum_{i=1}^8 S_i = 0.949073625 \implies \mathbf{94.91\%}$
* **Weighted Specificity:** $\frac{\sum S_i \cdot \text{Supp}_i}{4665} = 0.949073965 \implies \mathbf{94.91\%}$ ($\Delta = +0.0000003$)
* **Macro F1:** $\frac{1}{8}\sum_{i=1}^8 F1_i = 0.642102000 \implies \mathbf{64.21\%}$
* **Weighted F1:** $\frac{\sum F1_i \cdot \text{Supp}_i}{4665} = 0.642091727 \implies \mathbf{64.21\%}$ ($\Delta = -0.0000103$)

### B. Mean Spatial Patch Probe Aggregates (Raw Float64 Precision):
* **Macro Precision:** $\frac{1}{8}\sum_{i=1}^8 P_i = 0.650541875 \implies \mathbf{65.05\%}$
* **Weighted Precision:** $\frac{\sum P_i \cdot \text{Supp}_i}{4665} = 0.650543116 \implies \mathbf{65.05\%}$ ($\Delta = +0.0000012$)
* **Macro Recall:** $\frac{1}{8}\sum_{i=1}^8 R_i = 0.650814750 \implies \mathbf{65.08\%}$
* **Weighted Recall:** $\frac{\text{Total TP}}{4665} = \frac{3036}{4665} = 0.650803858 \implies \mathbf{65.08\%}$ ($\Delta = -0.0000109$)
* **Macro Specificity:** $\frac{1}{8}\sum_{i=1}^8 S_i = 0.950114875 \implies \mathbf{95.01\%}$
* **Weighted Specificity:** $\frac{\sum S_i \cdot \text{Supp}_i}{4665} = 0.950115524 \implies \mathbf{95.01\%}$ ($\Delta = +0.0000006$)
* **Macro F1:** $\frac{1}{8}\sum_{i=1}^8 F1_i = 0.650325000 \implies \mathbf{65.03\%}$
* **Weighted F1:** $\frac{\sum F1_i \cdot \text{Supp}_i}{4665} = 0.650320079 \implies \mathbf{65.03\%}$ ($\Delta = -0.0000049$)

---

## 5. Canonical Standardized Per-Class Benchmark Table

Below is the verified, standardized benchmark table adhering strictly to the required schema (exactly the 8 metric columns, no extraneous columns):

| Class | CLS Precision | CLS Recall | CLS Specificity | CLS F1 | Mean-Patch Precision | Mean-Patch Recall | Mean-Patch Specificity | Mean-Patch F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ADI** | 75.08% | 76.50% | 96.37% | 75.79% | 76.13% | 78.22% | 96.50% | 77.16% |
| **DEB** | 58.72% | 56.60% | 94.32% | 57.64% | 58.15% | 58.15% | 94.02% | 58.15% |
| **LYM** | 74.45% | 80.96% | 96.03% | 77.57% | 77.36% | 80.27% | 96.64% | 78.79% |
| **MUC** | 61.79% | 57.88% | 94.88% | 59.77% | 65.79% | 59.93% | 95.54% | 62.72% |
| **MUS** | 72.20% | 66.38% | 96.35% | 69.17% | 71.04% | 67.75% | 96.06% | 69.36% |
| **NOR** | 51.37% | 48.20% | 93.48% | 49.73% | 52.35% | 51.63% | 93.29% | 51.99% |
| **STR** | 60.62% | 60.72% | 94.37% | 60.67% | 58.56% | 59.86% | 93.95% | 59.20% |
| **TUM** | 59.61% | 67.58% | 93.46% | 63.34% | 61.07% | 64.84% | 94.10% | 62.90% |
| **Macro Average** | 64.23% | 64.35% | 94.91% | 64.21% | 65.05% | 65.08% | 95.01% | 65.03% |
| **Weighted Average** | 64.23% | 64.35% | 94.91% | 64.21% | 65.05% | 65.08% | 95.01% | 65.03% |

---

## 6. System & Asset Preservation Verification

The integrity of all core assets was verified prior to certification:
- **Baseline Checkpoint:** `models/checkpoints/gastric_rnn_classifier.pt` SHA-256 is bitwise identical (`65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8`).
- **Database:** SQLite `app.db` contains exactly 4 historical records.
- **Cache & Manifests:** 31,096 records in `cls_embeddings.npy` and `mean_patch_embeddings.npy` remain untouched.
- **Test Predictions:** `linear_probe_cls_test_predictions.csv` and `linear_probe_mean_patch_test_predictions.csv` unmodified.

*Report certified by GastricPath AI Research Engine.*

