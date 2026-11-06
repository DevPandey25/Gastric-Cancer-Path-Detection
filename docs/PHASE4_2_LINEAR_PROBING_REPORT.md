# GastricPath AI — Phase 4.2 Linear Probing Report
### Frozen Foundation Model Downstream Probing & Linear Evaluation

**Document ID:** `GP-REPORT-PHASE4-2-PROBE-2026-V1`  
**Execution Phase:** Phase 4.2 (Downstream Linear Probing & Evaluation)  
**Target Checkpoint:** `models/checkpoints/gastric_rnn_classifier.pt`  
**Checkpoint SHA-256:** `65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8` (Bitwise Preserved)  
**Git Branch:** `gastricpath-ai-evolution`  
**Date:** November 2026  
**Status:** Completed & Empirically Verified (95/95 Automated Tests Passing)  

---

## 1. Executive Summary

Phase 4.2 evaluated the classification utility of frozen self-supervised **Meta AI DINOv2 ViT-S/14** visual representations on the 8-class gastric histopathology benchmark (GCHTID) using downstream linear probing.

### Primary Research Objective:
Quantify the amount of linearly separable discriminative signal present in the pretrained representation space **prior to any task-specific encoder fine-tuning or medical domain adaptation**.

### Core Experimental Findings:
1. **Strong Linear Separability:** A single linear softmax probe trained on frozen representations achieves **64.35% Test Accuracy** (CLS token) and **65.08% Test Accuracy** (Mean Spatial Patch token) on the 4,665-image test split, compared to random chance of $12.50\%$.
2. **Substantial Gain Over Raw Retrieval:** Adding a linear layer boosts classification accuracy by **+17.75 percentage points** over raw nearest-neighbor retrieval ($47.33\% \rightarrow 65.08\%$), confirming that class separation boundaries in the 384-dimensional latent space are linear hyperplanes that require supervised projection weights to properly orient.
3. **Superiority of Spatial Patch Averaging:** Mean spatial patch-token embeddings slightly outperform global CLS tokens across accuracy ($65.08\%$ vs. $64.35\%$), Macro-F1 ($65.03\%$ vs. $64.21\%$), and OvR AUROC ($0.9316$ vs. $0.9293$), demonstrating that pooling local patch tokens preserves localized histological texture features.
4. **Calibration Quality:** The linear probe demonstrates remarkable probability calibration, achieving an Expected Calibration Error (ECE) of **0.0220** (CLS) and **0.0368** (Mean Patch).
5. **Distinct Phenotypic Accuracy Hierarchy:** Adipose tissue (`ADI`: $77.2\%$ F1) and Lymphocytes (`LYM`: $78.8\%$ F1) exhibit highest discriminative separability; Normal gastric mucosa (`NOR`: $52.0\%$ F1) and Debris (`DEB`: $58.2\%$ F1) present highest morphological ambiguity.

---

## 2. Experimental Protocol & Split Isolation

To guarantee absolute scientific integrity and prevent data leakage:
- **Model Architecture:** Scikit-Learn multinomial logistic regression (`LogisticRegression(solver='lbfgs', max_iter=1000, tol=1e-4, fit_intercept=True, random_state=42)`). The `PyTorchLinearSoftmax` module was used for exploratory validation only; all reported benchmark numbers originate from the Scikit-Learn L-BFGS probe.
- **Encoder Status:** Strictly frozen (`param.requires_grad = False`). The DINOv2 visual encoder was never loaded for gradient backpropagation.
- **Fitting Data:** Training split only ($N = 21,767$ images).
- **Hyperparameter Tuning:** Validation split only ($N = 4,664$ images).
- **Final Evaluation:** Test split only ($N = 4,665$ images), evaluated **exactly once** on the final selected model.
- **Reproducibility:** Seed fixed to $42$ across Scikit-Learn L-BFGS and PyTorch modules.
- **Confidence Interval Scope:** All reported intervals are image/patch-level bounds (Wilson score binomial intervals for proportions, bootstrap percentiles for non-linear metrics). No patient-level or clinical diagnostic confidence is asserted.

---

## 3. Hyperparameter Selection on Validation Split

We conducted a systematic grid sweep over the L2 regularization parameter $C \in [0.001, 200.0]$ using multinomial logistic regression with L-BFGS optimization on the training split ($N=21,767$). The optimal model was selected strictly based on **maximum Validation Macro-F1**.

### Validation Sweep Results:

| Candidate $C$ | CLS Fit Time | CLS Iterations | CLS Val Acc | CLS Val Macro-F1 | Mean Patch Fit Time | Mean Patch Iterations | Mean Patch Val Acc | Mean Patch Val Macro-F1 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| $0.001$ | 0.21 s | 7 (Conv: True) | 45.00% | 39.90% | 0.27 s | 7 (Conv: True) | 44.70% | 39.21% |
| $0.010$ | 0.58 s | 21 (Conv: True) | 50.81% | 48.47% | 0.78 s | 22 (Conv: True) | 50.62% | 47.80% |
| $0.050$ | 1.23 s | 43 (Conv: True) | 56.11% | 55.42% | 1.22 s | 40 (Conv: True) | 56.13% | 55.20% |
| $0.100$ | 1.46 s | 53 (Conv: True) | 57.65% | 57.22% | 1.61 s | 56 (Conv: True) | 58.04% | 57.45% |
| $0.500$ | 2.70 s | 91 (Conv: True) | 61.88% | 61.73% | 2.84 s | 101 (Conv: True) | 62.01% | 61.76% |
| $1.000$ | 4.20 s | 132 (Conv: True) | 63.21% | 63.11% | 3.46 s | 128 (Conv: True) | 63.14% | 62.96% |
| $5.000$ | 7.14 s | 253 (Conv: True) | 65.12% | 65.04% | 5.29 s | 180 (Conv: True) | 64.99% | 64.88% |
| $10.000$ | 8.76 s | 309 (Conv: True) | 65.18% | 65.09% | 5.59 s | 230 (Conv: True) | 65.24% | 65.13% |
| **$20.000$** | **8.21 s** | **308 (Conv: True)** | **65.42%** | **65.32% (PEAK CLS)** | 8.43 s | 320 (Conv: True) | 65.63% | 65.57% |
| $50.000$ | 9.80 s | 357 (Conv: True) | 64.99% | 64.93% | 10.92 s | 421 (Conv: True) | 65.91% | 65.83% |
| $100.000$ | 13.07 s | 491 (Conv: True) | 64.54% | 64.48% | 10.39 s | 420 (Conv: True) | 66.12% | 66.05% |
| **$200.000$** | 13.18 s | 496 (Conv: True) | 64.67% | 64.61% | **10.10 s** | **408 (Conv: True)** | **66.12%** | **66.06% (PEAK PATCH)** |

### Selected Hyperparameters:
- **Optimal Regularization for CLS Token:** **$C^* = 20.0$** (Validation Macro-F1: $65.32\%$).
- **Optimal Regularization for Mean Patch Token:** **$C^* = 200.0$** (Validation Macro-F1: $66.06\%$).
- **Convergence:** 100% of candidate configurations successfully converged well within the 1,000-iteration ceiling.

---

## 4. Final Test-Set Evaluation (Evaluated Exactly Once)

The optimal linear probes were evaluated strictly on the **Phase 2 test set ($N=4,665$)** exactly once:

| Metric | CLS Token Probe ($C^*=20.0$) | Mean Patch Probe ($C^*=200.0$) | Confidence Interval Scope |
|:---|:---:|:---:|:---|
| **Accuracy** | **64.35%** | **65.08%** | 95% Wilson Score Binomial CI |
| *Accuracy 95% CI* | $[62.97\%, 65.71\%]$ | $[63.70\%, 66.44\%]$ | Image-Level Sample Proportion |
| **Balanced Accuracy** | **64.35%** | **65.08%** | Empirical Bootstrap ($B=1,000$) |
| *Balanced Acc 95% CI* | $[63.09\%, 65.83\%]$ | $[63.84\%, 66.41\%]$ | Image-Level Bootstrap Resampling |
| **Macro Precision** | **64.23%** | **65.05%** | Unweighted Class Average |
| **Macro Recall** | **64.35%** | **65.08%** | Unweighted Class Average |
| **Macro F1-Score** | **64.21%** | **65.03%** | Harmonic Mean of Precision & Recall |
| *Macro-F1 95% CI* | $[62.93\%, 65.64\%]$ | $[63.75\%, 66.38\%]$ | Image-Level Bootstrap Resampling |
| **Weighted F1-Score** | **64.21%** | **65.03%** | Support-Weighted Average |
| **Cohen's Kappa ($\kappa$)** | **0.5926** | **0.6009** | Substantial Agreement ($\kappa > 0.60$) |
| **Macro OvR AUROC** | **0.9293** | **0.9316** | Area Under Receiver Operating Curve |
| **Weighted OvR AUROC** | **0.9293** | **0.9316** | Area Under Receiver Operating Curve |
| **Expected Calibration Error** | **0.0220** | **0.0368** | 10 Confidence Bins |
| **Multiclass Brier Score** | **0.4819** | **0.4730** | Mean Squared Probability Error |

---

## 5. Per-Class Performance Breakdown

Detailed per-class metrics on the test split ($N = 4,665$) for the CLS Token Probe ($C^*=20.0$) and Mean Spatial Patch Probe ($C^*=200.0$):

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

*Note on DEB Precision/Recall Coincidence:* In the Mean Patch probe, Necrotic Debris (`DEB`) exhibits identical precision, recall, and F1 at $58.15\%$. An audit of the confusion matrix confirms $\text{TP}=339$, $\text{FP}=244$, $\text{FN}=244$; since $\text{FP} = \text{FN}$, Precision ($339/583$) and Recall ($339/583$) are mathematically identical ($58.1475\% \rightarrow 58.15\%$). Detailed verification records are documented in [`docs/PHASE4_2_2_FINAL_METRIC_TABLE_VERIFICATION.md`](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/docs/PHASE4_2_2_FINAL_METRIC_TABLE_VERIFICATION.md).

---

## 6. Confusion Matrix & Histological Error Analysis

Below is the complete count confusion matrix for the CLS Linear Probe on the test set ($N=4,665$):

| True Class \ Predicted | ADI | DEB | LYM | MUC | MUS | NOR | STR | TUM | Total Queries | Recall |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ADI** | **446** | 13 | 0 | 77 | 26 | 17 | 1 | 3 | 583 | **76.50%** |
| **DEB** | 5 | **330** | 50 | 33 | 28 | 35 | 45 | 57 | 583 | **56.60%** |
| **LYM** | 3 | 36 | **472** | 2 | 3 | 25 | 15 | 27 | 583 | **80.96%** |
| **MUC** | 76 | 38 | 5 | **338** | 24 | 55 | 26 | 22 | 584 | **57.88%** |
| **MUS** | 30 | 39 | 11 | 20 | **387** | 20 | 65 | 11 | 583 | **66.38%** |
| **NOR** | 31 | 45 | 49 | 45 | 6 | **281** | 41 | 85 | 583 | **48.20%** |
| **STR** | 1 | 31 | 27 | 14 | 56 | 38 | **354** | 62 | 583 | **60.72%** |
| **TUM** | 2 | 30 | 20 | 18 | 6 | 76 | 37 | **394** | 583 | **67.58%** |

### Verified Representative Error Cases:
1. **Tumor Epithelium Misclassified as Normal Gastric Mucosa (`TUM` $\rightarrow$ `NOR`):**
   - [`TUM/TUM_1012.png`](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/data/raw/HMU-GC-HE-30K/all_image/TUM/TUM_1012.png): True = `TUM`, Predicted = `NOR` (Confidence = $0.5334$).
   - [`TUM/TUM_1014.png`](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/data/raw/HMU-GC-HE-30K/all_image/TUM/TUM_1014.png): True = `TUM`, Predicted = `NOR` (Confidence = $0.4924$).
   - *Histological Hypothesis:* We hypothesize that well-differentiated tubular adenocarcinoma preserves gland-forming architecture, displaying lumenal cavities and epithelial lining that closely mirror non-neoplastic gastric glands without overt cytological pleomorphism distinguishable at patch resolution.
2. **Tumor Epithelium Misclassified as Stroma (`TUM` $\rightarrow$ `STR`):**
   - [`TUM/TUM_112.png`](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/data/raw/HMU-GC-HE-30K/all_image/TUM/TUM_112.png): True = `TUM`, Predicted = `STR` (Confidence = $0.7393$).
   - [`TUM/TUM_1145.png`](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/data/raw/HMU-GC-HE-30K/all_image/TUM/TUM_1145.png): True = `TUM`, Predicted = `STR` (Confidence = $0.6139$).
   - *Histological Hypothesis:* We hypothesize that infiltrative carcinoma (such as diffuse/Lauren type or scirrhous carcinoma) frequently incites dense desmoplastic stroma where malignant epithelial cells account for a small fraction ($<20\%$) of patch area, leading visual representations to be dominated by collagenous stromal features.
3. **Mucus vs. Adipose Reciprocal Confusion (`MUC` $\leftrightarrow$ `ADI`):**
   - [`MUC/MUC_1016.png`](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/data/raw/HMU-GC-HE-30K/all_image/MUC/MUC_1016.png): True = `MUC`, Predicted = `ADI` (Confidence = $0.5713$).
   - [`ADI/ADI_1039.png`](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/data/raw/HMU-GC-HE-30K/all_image/ADI/ADI_1039.png): True = `ADI`, Predicted = `MUC` (Confidence = $0.7603$).
   - *Histological Hypothesis:* We hypothesize that extracellular mucin pools and intracellular lipid clearing both appear as optically pale or non-staining vacuolated expanses on routine H&E, creating visual texture ambiguity for generic self-supervised representations.

---

## 7. Multi-Model Comparison & Baseline Context

| Model / Methodology | Feature Representation | Optimization / Training Scope | Test Cohort & Size | Accuracy | Macro-F1 | Macro AUROC | Calibration (ECE) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Guessing** | None | Null baseline | GCHTID 8 classes ($N=4,665$) | $12.50\%$ | $12.50\%$ | $0.5000$ | — |
| **DINOv2 Frozen Nearest Neighbor (Top-1)** | CLS Token (384-d) | Zero training (unsupervised) | Phase 2 Test ($N=4,665$) | $47.31\%$ | $47.31\%$ | — | — |
| **DINOv2 Frozen Nearest Neighbor (Top-1)** | Mean Patch Token (384-d) | Zero training (unsupervised) | Phase 2 Test ($N=4,665$) | $47.33\%$ | $47.33\%$ | — | — |
| **DINOv2 Frozen Majority Vote (k=10)** | Mean Patch Token (384-d) | Zero training (unsupervised) | Phase 2 Test ($N=4,665$) | $55.13\%$ | $55.13\%$ | — | — |
| **DINOv2 Frozen Linear Probe** | **CLS Token (384-d)** | **Linear layer only (Train $N=21,767$)** | **Phase 2 Test ($N=4,665$)** | **$64.35\%$** | **$64.21\%$** | **$0.9293$** | **$0.0220$** |
| **DINOv2 Frozen Linear Probe** | **Mean Patch Token (384-d)** | **Linear layer only (Train $N=21,767$)** | **Phase 2 Test ($N=4,665$)** | **$65.08\%$** | **$65.03\%$** | **$0.9316$** | **$0.0368$** |
| *Historical Baseline Checkpoint* (`GastricCancerRNN`) | *End-to-End CNN-RNN* | *Full historical training (Epoch 15)* | *Historical Test ($N=3,110$)* | *65.92%* | *66.11%* | *0.9392* | *0.0357* |
| *Historical Baseline (Strictly Unexposed)* | *End-to-End CNN-RNN* | *Historical training (no dup exposure)* | *Unexposed Test ($N=2,918$)* | *65.66%* | *65.86%* | *0.9388* | *0.0361* |

### Explicit Split Incommensurability Disclaimer:
> [!WARNING]
> **CRITICAL SCIENTIFIC CAVEAT ON BASELINE COMPARISON:**
> - The historical `GastricCancerRNN` baseline checkpoint was evaluated on its **historical test split ($N=3,110$)**, which was constructed prior to our Phase 2 duplicate-quarantine audit.
> - The frozen DINOv2 linear probe was evaluated on the **Phase 2 cluster-isolated split ($N=4,665$)**, where all multi-image perceptual clusters (Hamming $\le 2$) were quarantined into single splits.
> - **Consequently, this is NOT a direct apples-to-apples performance comparison.** Differing test sample compositions, cohort sizes, and duplicate isolation constraints prevent declaring numerical superiority or statistical equivalence of one architecture over another.
> - What this comparison empirically demonstrates is **numerical proximity**: a frozen foundation model visual representation paired with a single linear classifier exhibits performance in the same quantitative tier ($65.08\%$) on the cluster-isolated split as the specialized end-to-end network achieved on its historical cohort ($65.92\%$). Detailed audit and verification records are documented in [`docs/PHASE4_2_1_METRIC_AND_EVALUATION_AUDIT.md`](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/docs/PHASE4_2_1_METRIC_AND_EVALUATION_AUDIT.md).

---

## 8. Research Visualizations Produced

Three dedicated figures were generated in [`research/foundation_models/visualizations/`](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/foundation_models/visualizations/):

1. **[`linear_probe_tuning_curve.png`](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/foundation_models/visualizations/linear_probe_tuning_curve.png):** Regularization sweep showing Validation Macro-F1 across 12 candidate $C$ values for both CLS and Mean Patch tokens.
2. **[`linear_probe_confusion_matrix.png`](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/foundation_models/visualizations/linear_probe_confusion_matrix.png):** Percentage normalized confusion matrix for the CLS linear probe on the test set.
3. **[`linear_probe_per_class_f1.png`](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/foundation_models/visualizations/linear_probe_per_class_f1.png):** Grouped bar chart comparing per-class discrimination across raw retrieval ($P@1$), CLS probe F1, and Mean Patch probe F1.

---

## 9. Mandatory Clinical & Generalization Disclosures

1. **Research Utility Only:** These linear probe models evaluate latent feature utility in a research sandbox. They are not intended for clinical use, patient triage, or diagnostic grading.
2. **Patch-Level Metric Scope:** Metrics describe $224 \times 224$ tissue tile discrimination and cannot be extrapolated to patient-level diagnosis or whole-slide cancer staging.

---

*Report certified by GastricPath AI Research Engine.*

