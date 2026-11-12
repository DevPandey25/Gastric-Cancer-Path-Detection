# GastricPath AI — Supplementary Research Analysis: Error Analysis and Class-Level Findings Report

**Category:** Supplementary Research Analysis Work Package (Post-Hoc Foundation Model Evaluation)  
**Date:** November 7, 2026  
**Status:** Complete & Statistically Verified  
**Evaluated Model:** Frozen `owkin/phikon` ViT-B/16 Encoder + Mean Patch Pooling + Multinomial Logistic Probe ($C=10$)  
**Primary Evaluation Population:** Held-Out Clean Test Split ($N=467$ images, Seed 42)  
**Total Predictions:** 310 Correct ($66.38\%$), 157 Misclassifications ($33.62\%$)  
**Macro Metrics:** Precision $67.33\%$, Recall $66.39\%$, F1-Score $66.55\%$, Specificity $95.20\%$

---

## Executive Summary

This report delivers a comprehensive, empirical error analysis of the winning **Phikon Mean Patch** foundation model representation on the held-out test cohort ($N=467$) from the GCHTID benchmark. Rather than treating model performance as a monolithic aggregate ($66.38\%$ accuracy), this investigation dissects per-class capabilities, quantifies off-diagonal confusion pairs, isolates high-confidence misclassifications, and bridges findings across the supplementary research lineage (Phases 5 through 8).

> [!NOTE]
> All visual texture descriptions and biological associations discussed in this report are exploratory, unverified hypotheses based on optical image features. The GCHTID dataset provides tile-level class labels without pixel-level histopathological annotations; findings do not claim verified histological causation.

### Primary Conclusions
1. **Performance Stratification Across Classes:**
   - **Highest Performance:** Adipose tissue (**ADI**, F1 $76.11\%$, Precision $79.63\%$) and Lymphocytes (**LYM**, F1 $72.90\%$, Precision $81.25\%$) achieved the highest discriminative fidelity, benefiting from distinct optical texture and cellular density.
   - **Lowest Performance:** Debris (**DEB**, F1 $59.46\%$, Recall $55.93\%$), Normal Mucosa (**NOR**, F1 $61.42\%$, Precision $56.52\%$), and Stroma (**STR**, F1 $62.50\%$, Precision $57.14\%$) exhibited the highest error rates.
2. **Dominant Off-Diagonal Confusion Clusters (157 Total Errors):**
   - **Bidirectional Muscle–Stroma Confusion ($\text{MUS} \leftrightarrow \text{STR}$):** 16 total errors ($10.19\%$ of all errors). 8 MUS samples were predicted as STR, and 8 STR samples were predicted as MUS. Both classes share elongated, fibrillar eosinophilic textures (unverified visual similarity hypothesis).
   - **Normal Mucosa to Tumor ($\text{NOR} \to \text{TUM}$):** 9 errors ($15.52\%$ of NOR tiles, $5.73\%$ of all errors). Glandular crowding and complex mucosal crypts are misidentified as adenocarcinoma.
   - **Adipose to Mucus ($\text{ADI} \to \text{MUC}$):** 9 errors ($15.25\%$ of ADI tiles, $5.73\%$ of all errors). Both classes feature optically clear, pale, low-density regions.
   - **Lymphocytes to Debris ($\text{LYM} \to \text{DEB}$):** 7 errors ($11.86\%$ of LYM tiles, $4.46\%$ of all errors). Pyknotic, fragmented debris particles visually mimic dense lymphocytic infiltrate.
3. **Confidence–Error Dynamics:**
   - Correct predictions exhibit high average confidence ($0.7846 \pm 0.1802$, median $0.8268$).
   - Misclassifications exhibit substantially lower confidence ($0.5755 \pm 0.1659$, median $0.5487$).
   - However, **18 test tiles suffered high-confidence misclassifications** ($\text{conf} > 0.80$), where the linear probe assigned high posterior probability to an incorrect class.
4. **Cross-Phase Convergence (Phases 5–8):**
   - **Phase 5 (Explainability):** Misclassified tiles exhibit diffuse or off-target token attribution patterns rather than focal tissue grounding.
   - **Phase 6 (Robustness):** Confusable class pairs ($\text{MUS}/\text{STR}$, $\text{MUC}/\text{ADI}$) were the most vulnerable to JPEG compression (78 flips, $16.70\%$) and brightness shifts (70 flips, $14.99\%$).
   - **Phase 7 (OOD Screening):** Clean errors exhibited higher mean Mahalanobis distances ($808.85 \pm 373.86$) than correct predictions ($756.97 \pm 320.80$).
   - **Phase 8 (Spatial Analysis):** Misclassified tiles demonstrated lower regional quadrant cosine similarity ($0.6959 \pm 0.0655$) than correct tiles ($0.7338 \pm 0.1034$). This is an exploratory, descriptive finding showing that within-tile spatial heterogeneity correlates with classification difficulty, not a causal relationship.

---

## 1. Dataset & Evaluation Population

All analyses are grounded strictly on the unperturbed, held-out clean test split from:
`research/foundation_models/cache/subset10pct/manifest_subset.csv`

The primary predictions originate from the frozen Phase 4.4 evaluation file:
`research/foundation_models/cache/subset10pct/evaluation/phikon_mean_patch_test_predictions.csv`

- **Total Test Samples:** $N=467$
- **Class Balance:** Approximately balanced across all 8 canonical classes (58 to 59 tiles per class):
  - `ADI`: 59 tiles ($12.63\%$)
  - `DEB`: 59 tiles ($12.63\%$)
  - `LYM`: 59 tiles ($12.63\%$)
  - `MUC`: 58 tiles ($12.42\%$)
  - `MUS`: 58 tiles ($12.42\%$)
  - `NOR`: 58 tiles ($12.42\%$)
  - `STR`: 58 tiles ($12.42\%$)
  - `TUM`: 58 tiles ($12.42\%$)
- **Data Invariants:** Patient duplicate-group isolation is strictly preserved. No test samples were leaked into probe training or validation.

---

## 2. Overall & Class-Level Performance Metrics

The table below details true positives (TP), false positives (FP), false negatives (FN), true negatives (TN), precision, recall (sensitivity), specificity, and F1-score across all 8 classes. Aggregate count sums are explicitly distinguished from macro and weighted rate averages:

| Class Name | Support | TP | FP | FN | TN | Precision | Recall (Sens.) | Specificity | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ADI** (Adipose) | 59 | 43 | 11 | 16 | 397 | **0.7963** | **0.7288** | 0.9730 | **0.7611** |
| **DEB** (Debris) | 59 | 33 | 19 | 26 | 389 | 0.6346 | 0.5593 | 0.9534 | 0.5946 |
| **LYM** (Lymphocytes) | 59 | 39 | 9 | 20 | 399 | **0.8125** | 0.6610 | **0.9779** | **0.7290** |
| **MUC** (Mucus) | 58 | 36 | 18 | 22 | 391 | 0.6667 | 0.6207 | 0.9560 | 0.6429 |
| **MUS** (Muscle) | 58 | 39 | 16 | 19 | 393 | 0.7091 | 0.6724 | 0.9609 | 0.6903 |
| **NOR** (Normal Mucosa) | 58 | 39 | 30 | 19 | 379 | 0.5652 | 0.6724 | 0.9267 | 0.6142 |
| **STR** (Stroma) | 58 | 40 | 30 | 18 | 379 | 0.5714 | 0.6897 | 0.9267 | 0.6250 |
| **TUM** (Tumor) | 58 | 41 | 24 | 17 | 385 | 0.6308 | **0.7069** | 0.9413 | 0.6667 |
| **Cohort Total Counts** | **467** | **310** | **157** | **157** | **3112** | — | — | — | — |
| **Macro Average (Rates)** | **467** | — | — | — | — | **0.6733** | **0.6639** | **0.9520** | **0.6655** |
| **Weighted Average (Rates)** | **467** | — | — | — | — | **0.6738** | **0.6638** | **0.9521** | **0.6657** |

*Note: Total errors = $\sum \text{FP} = \sum \text{FN} = 157$. Overall accuracy = $\text{TP}_{\text{total}} / N = 310 / 467 = 66.38\%$.*

---

## 3. Confusion Matrix Analysis

### A. Raw Count Confusion Matrix

| True \ Pred | ADI | DEB | LYM | MUC | MUS | NOR | STR | TUM | Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ADI** | **43** | 1 | 0 | 9 | 1 | 5 | 0 | 0 | 59 |
| **DEB** | 1 | **33** | 4 | 4 | 1 | 5 | 6 | 5 | 59 |
| **LYM** | 0 | 7 | **39** | 0 | 0 | 5 | 2 | 6 | 59 |
| **MUC** | 5 | 1 | 1 | **36** | 4 | 6 | 4 | 1 | 58 |
| **MUS** | 4 | 2 | 0 | 2 | **39** | 3 | 8 | 0 | 58 |
| **NOR** | 1 | 2 | 1 | 1 | 2 | **39** | 3 | 9 | 58 |
| **STR** | 0 | 1 | 0 | 2 | 8 | 4 | **40** | 3 | 58 |
| **TUM** | 0 | 5 | 3 | 0 | 0 | 2 | 7 | **41** | 58 |
| **Total Predicted** | 54 | 52 | 48 | 54 | 55 | 69 | 70 | 65 | 467 |

### B. Row-Normalized Proportions (Sensitivity)

| True \ Pred | ADI | DEB | LYM | MUC | MUS | NOR | STR | TUM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ADI** | **0.73** | 0.02 | 0.00 | 0.15 | 0.02 | 0.08 | 0.00 | 0.00 |
| **DEB** | 0.02 | **0.56** | 0.07 | 0.07 | 0.02 | 0.08 | 0.10 | 0.08 |
| **LYM** | 0.00 | 0.12 | **0.66** | 0.00 | 0.00 | 0.08 | 0.03 | 0.10 |
| **MUC** | 0.09 | 0.02 | 0.02 | **0.62** | 0.07 | 0.10 | 0.07 | 0.02 |
| **MUS** | 0.07 | 0.03 | 0.00 | 0.03 | **0.67** | 0.05 | 0.14 | 0.00 |
| **NOR** | 0.02 | 0.03 | 0.02 | 0.02 | 0.03 | **0.67** | 0.05 | 0.16 |
| **STR** | 0.00 | 0.02 | 0.00 | 0.03 | 0.14 | 0.07 | **0.69** | 0.05 |
| **TUM** | 0.00 | 0.09 | 0.05 | 0.00 | 0.00 | 0.03 | 0.12 | **0.71** |

Visualizations:
- [confusion_matrix_counts.png](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/error_analysis/confusion_matrix_counts.png)
- [confusion_matrix_normalized.png](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/error_analysis/confusion_matrix_normalized.png)

---

## 4. Frequent Off-Diagonal Confusion Pairs

Out of 56 possible off-diagonal error pairs, **43 pairs occurred at least once**. The top 10 error pairs account for **72 of the 157 total errors (45.86%)**:

| Rank | True Class | Predicted Class | Error Count | % of True Class | % of All Errors | Visual Texture Hypothesis (Unverified / Exploratory) |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **ADI** | **MUC** | 9 | $15.25\%$ | $5.73\%$ | Optically clear, pale lipid vacuoles visually resemble pools of extracellular mucin. |
| **2** | **NOR** | **TUM** | 9 | $15.52\%$ | $5.73\%$ | Crowded glandular epithelium and crypt distortion resemble neoplastic glandular contours. |
| **3** | **MUS** | **STR** | 8 | $13.79\%$ | $5.10\%$ | Eosinophilic bundles of smooth muscle fibers visually overlap with dense collagenous stroma. |
| **4** | **STR** | **MUS** | 8 | $13.79\%$ | $5.10\%$ | Collagen fibers and spindle cells visually resemble smooth muscle fascicles. |
| **5** | **LYM** | **DEB** | 7 | $11.86\%$ | $4.46\%$ | Dark, punctate lymphocytic nuclear clusters resemble granular necrotic debris particles. |
| **6** | **TUM** | **STR** | 7 | $12.07\%$ | $4.46\%$ | Adenocarcinoma tiles containing substantial reactive stroma in the field of view. |
| **7** | **MUC** | **NOR** | 6 | $10.34\%$ | $3.82\%$ | Mucinous background containing residual glandular epithelial fragments. |
| **8** | **DEB** | **STR** | 6 | $10.17\%$ | $3.82\%$ | Amorphous cellular debris overlying background stromal collagen. |
| **9** | **LYM** | **TUM** | 6 | $10.17\%$ | $3.82\%$ | Densely packed lymphocytic sheets visually resemble solid poorly differentiated tumor nests. |
| **10** | **DEB** | **NOR** | 5 | $8.47\%$ | $3.18\%$ | Debris fragments adjacent to mucosal epithelial surface. |

Visualization:
- [top_confusion_pairs.png](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/error_analysis/top_confusion_pairs.png)

---

## 5. Deterministic Correct and Incorrect Prediction Gallery

A deterministic gallery of **48 exemplars** (3 correct and 3 incorrect per class) was extracted and indexed:

### Selection Criteria
- **Correct Exemplars (24 tiles):**
  1. *Representative High-Confidence:* Highest model probability for that class.
  2. *Borderline Low-Confidence:* Lowest model probability among correct predictions (entropy threshold test).
  3. *Cross-Phase Verified / Median:* Tiles appearing in Phase 5 attribution or Phase 8 spatial analysis, or median confidence.
- **Incorrect Exemplars (24 tiles):**
  1. *High-Confidence Misclassification:* Misclassified tile with maximum probability (model confidence error).
  2. *Frequent Confusion Pair:* Misclassified tile belonging to the top off-diagonal confusion pair for that class.
  3. *Cross-Phase Documented / Ambiguous Error:* Documented in prior phases or exhibiting lowest error probability.

### Gallery Manifest Summary (Exemplar Sample)

| True | Pred | Type | Conf. | Selection Reason | Sample Path | Phase 5 | Phase 8 | Mahal. OOD |
| :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: |
| ADI | ADI | Correct | 0.9996 | High-Confidence Canonical | `ADI/ADI_1118.png` | Yes | Yes | 586.2 |
| ADI | ADI | Correct | 0.3842 | Borderline Low-Confidence | `ADI/ADI_2112.png` | No | No | 785.4 |
| ADI | MUC | Error | 0.7612 | High-Confidence Misclassification | `ADI/ADI_2355.png` | No | Yes | 891.1 |
| ADI | MUC | Error | 0.6541 | Frequent Confusion Pair | `ADI/ADI_3345.png` | No | No | 742.3 |
| MUS | MUS | Correct | 0.9984 | High-Confidence Canonical | `MUS/MUS_1303.png` | Yes | Yes | 612.0 |
| MUS | STR | Error | 0.7410 | Frequent Confusion Pair | `MUS/MUS_2491.png` | No | Yes | 915.6 |
| STR | MUS | Error | 0.6173 | Cross-Phase Documented Error | `STR/STR_1000.png` | Yes | Yes | 845.2 |
| NOR | TUM | Error | 0.8124 | High-Confidence Misclassification | `NOR/NOR_3110.png` | No | No | 1145.7 |
| TUM | STR | Error | 0.6845 | Frequent Confusion Pair | `TUM/TUM_2180.png` | No | Yes | 930.8 |

Full gallery data: [error_analysis_manifest.csv](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/error_analysis/error_analysis_manifest.csv)

Visual Galleries:
- [correct_prediction_gallery.png](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/error_analysis/correct_prediction_gallery.png)
- [incorrect_prediction_gallery.png](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/error_analysis/incorrect_prediction_gallery.png)

---

## 6. Confidence & Calibration Error Analysis

Prediction confidence (maximum softmax probability) differs substantially between correct predictions and errors:

| Metric | Correct Predictions ($N=310$) | Misclassifications ($N=157$) | Difference |
| :--- | :---: | :---: | :---: |
| **Mean Confidence** | **$0.7846$** | **$0.5755$** | $-0.2091$ |
| **Std Dev** | $0.1802$ | $0.1659$ | $-0.0143$ |
| **Median Confidence** | **$0.8268$** | **$0.5487$** | $-0.2781$ |
| **Min Confidence** | $0.2312$ | $0.2415$ | $+0.0103$ |
| **Max Confidence** | $0.9998$ | $0.8845$ | $-0.1153$ |

### High-Confidence Misclassifications & Underconfidence Observations
1. **High-Confidence Misclassifications ($\text{conf} > 0.80$):**
   18 test tiles ($11.46\%$ of all errors) were classified incorrectly with confidence $> 80\%$. Examples include:
   - `DEB_2077.png` (DEB $\to$ TUM, conf $0.8497$)
   - `NOR_3110.png` (NOR $\to$ TUM, conf $0.8124$)
   - `MUS_3419.png` (MUS $\to$ ADI, conf $0.8251$)
   These cases represent failure modes where the feature-space representation strongly clusters within an incorrect class centroid.
2. **Low-Confidence Correct Predictions ($\text{conf} < 0.40$):**
   8 test tiles ($2.58\%$ of correct predictions) were correctly predicted despite high uncertainty ($\text{conf} < 0.40$), indicating borderline boundaries between classes.

Visualization:
- [confidence_error_analysis.png](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/error_analysis/confidence_error_analysis.png)

---

## 7. Cross-Phase Syntheses (Phases 5 through 8)

### Phase 5 — Explainability & Attribution Overlap
- Of the 32 explainability gallery tiles, **23 were correctly classified and 9 were misclassified**.
- In correct predictions (e.g., `ADI_1118`, `MUS_1303`, `TUM_1005`), LOO attribution mass was densely concentrated on prominent optical features.
- In misclassifications (e.g., `STR_1000` $\to$ MUS, `DEB_2077` $\to$ TUM, `NOR_1084` $\to$ TUM), attribution mass was scattered diffusely across background areas, indicating absence of localized focal saliency.

### Phase 6 — Controlled Robustness Vulnerability
- Across the 467 test samples, perturbation-induced prediction flips were heavily concentrated in the most confusable classes:
  - **JPEG Compression (78 flips, $16.70\%$):** Strongly accelerated $\text{MUS} \to \text{STR}$ and $\text{STR} \to \text{MUS}$ confusion due to loss of high-frequency fibrillar detail.
  - **Brightness & Contrast (70 flips, $14.99\%$):** Severely impacted $\text{MUC} \to \text{ADI}$ and $\text{ADI} \to \text{MUC}$, because optical density shifts erase the subtle contrast separating lipid clearance from faintly basophilic mucin.
  - **Stain Variation (12 flips, $2.57\%$):** Demonstrates that Phikon's self-supervised pre-training is highly robust to optical density stain shifts, confirming that stain differences are not the primary driver of classification failure.

### Phase 7 — OOD and Feature-Space Uncertainty
- On the clean test cohort, misclassified samples exhibited higher mean Mahalanobis distances ($808.85 \pm 373.86$) than correct samples ($756.97 \pm 320.80$).
- When calibrated at the 95th percentile on validation data, the Mahalanobis detector flagged **25 clean test tiles as OOD** ($5.35\%$ false rejection rate). Of these 25, **7 were classification errors and 18 were correct predictions**.
- **Scientific distinction:** The Mahalanobis metric identifies *feature-space novelty* (distance from empirical class centroids in $\mathbb{R}^{768}$), whereas MSP identifies *posterior entropy*. High Mahalanobis distance in a correct prediction indicates an atypical or unusual tissue sample that the linear probe still classified correctly.

### Phase 8 — Intra-Patch Spatial Descriptors
- On the 96 spatial analysis tiles (12/class), misclassified tiles ($N=33$) showed:
  - **Lower Regional Cosine Similarity:** $0.6959 \pm 0.0655$ for errors vs. $0.7338 \pm 0.1034$ for correct predictions.
  - **Higher Regional Feature Dispersion:** $0.2281 \pm 0.0491$ for errors vs. $0.1996 \pm 0.0775$ for correct predictions.
- **Exploratory Finding:** Single tiles containing high spatial heterogeneity across their four quadrants produce dispersed regional representations that descriptively correlate with downstream classification error. This is an observational finding, not proof of a causal biological mechanism.

---

## 8. Summary of Observed Visual Pattern Hypotheses (Non-Causal / Unverified by Histopathologist)

Without attributing errors to unverified biology, four visual and representation-level hypotheses describe the dominant observed patterns:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              OBSERVED VISUAL PATTERN HYPOTHESES (DESCRIPTIVE)               │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Fibrous Texture Equivalence (MUS <-> STR, 16 errors):                    │
│    - Parallel fibrillar arrays of smooth muscle vs. dense stromal collagen  │
│    - Shared eosinophilic hue and spindle-shaped optical orientations        │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. Optically Clear / Low-Density Clearance (ADI <-> MUC, 14 errors):        │
│    - Clear lipid spaces in adipocytes vs. extracellular mucin pools         │
│    - High-frequency tile brightness with low structural edge density        │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. Epithelial Architecture Distortion (NOR -> TUM, 9 errors):               │
│    - Glandular branching, crowded crypts, and dense epithelial contours     │
│    - Overlapping feature representations between normal and neoplastic glands│
├─────────────────────────────────────────────────────────────────────────────┤
│ 4. Punctate Dark Nuclear Aggregation (LYM -> DEB / TUM, 13 errors):         │
│    - Dense lymphocytic infiltrates vs. fragmented necrotic pyknotic debris  │
│    - Solid tumor sheets mimicking dense lymphoid aggregates                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Scientific Limitations

1. **Descriptive, Post-Hoc Analysis:** This investigation does not alter model weights, tune hyperparameters, or train secondary classifiers. Findings describe the empirical behavior of the frozen Phikon representation.
2. **Absence of Pixel-Level Expert Annotations:** GCHTID supplies tile-level class labels without nuclear segmentations, stromal boundaries, or pathologist heatmaps. Visual descriptions are neutral and structural; they do not represent confirmed clinical grading.
3. **Tile-Level Scope ($224 \times 224$ px):** This analysis is restricted to isolated tiles. It does not account for whole-slide spatial architecture, surgical margin context, or patient-level prognosis.

---

## 10. Research Artifact Provenance

All Phase 9 artifacts are persisted in the repository:

- **Class-Level Metrics CSV:** [class_metrics.csv](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/error_analysis/class_metrics.csv)
- **Frequent Confusion Pairs CSV:** [confusion_pairs.csv](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/error_analysis/confusion_pairs.csv)
- **Deterministic 48-Exemplar Gallery Manifest:** [error_analysis_manifest.csv](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/error_analysis/error_analysis_manifest.csv)
- **Machine-Readable Summary JSON:** [error_analysis_summary.json](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/error_analysis/error_analysis_summary.json)
- **Visualizations:**
  - [confusion_matrix_counts.png](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/error_analysis/confusion_matrix_counts.png)
  - [confusion_matrix_normalized.png](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/error_analysis/confusion_matrix_normalized.png)
  - [top_confusion_pairs.png](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/error_analysis/top_confusion_pairs.png)
  - [correct_prediction_gallery.png](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/error_analysis/correct_prediction_gallery.png)
  - [incorrect_prediction_gallery.png](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/error_analysis/incorrect_prediction_gallery.png)
  - [confidence_error_analysis.png](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/error_analysis/confidence_error_analysis.png)
- **Automated Test Suite:** [test_phase9.py](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/tests/test_phase9.py) (9/9 tests passing; 59/59 regression tests passing)

### Related Research Reports
- Phase 4.4 Reduced Benchmark: `docs/PHASE4_4_REDUCED_COMPARISON_REPORT.md`
- Phase 5 Explainability: `docs/PHASE5_EXPLAINABILITY_REPORT.md`
- Phase 6 Robustness: `docs/PHASE6_ROBUSTNESS_REPORT.md`
- Phase 7 OOD Screening: `docs/PHASE7_OOD_UNCERTAINTY_REPORT.md`
- Phase 8 Spatial Analysis: `docs/PHASE8_SPATIAL_ANALYSIS_REPORT.md`
