# GastricPath AI — Phase 6: Controlled Robustness Analysis Report

**Document Version:** 1.0.0  
**Phase Status:** Phase 6 Complete  
**Execution Timestamp:** 2026-11-06T20:13:23Z (UTC) / 2026-11-07T01:43:23+05:30 (IST)  
**Evaluator Architecture:** Frozen `owkin/phikon` ViT-B/16 (768-d Mean-Patch Pooling) + Multinomial Logistic Probe ($C=10.0$, $\ell_2$-normalized)  
**Evaluation Cohort:** Full Held-Out Test Split from 10% Stratified Subset ($N=467$ image tiles across 8 canonical classes)  
**Test Suite Verification:** 9/9 Phase 6 Tests Passing (`tests/test_phase6.py`); 15/15 Regression Tests Passing (`tests/test_phase5.py`, `tests/test_phase4_4.py`)

---

## 1. Executive Summary

Phase 6 implements a rigorous, controlled robustness stress-test of the primary Phase 4.4 foundation model classifier—frozen **Phikon ViT-B/16 with Mean-Patch Pooling** ($d=768$, $C=10.0$)—against four predeclared synthetic perturbation families representing realistic digital pathology acquisition variations:

1. **Brightness & Contrast Shift:** Global illumination shift ($\times 1.1$ brightness, $\times 1.1$ contrast).
2. **Mild Optical Gaussian Blur:** Focal plane blur ($5\times 5$ kernel, $\sigma=1.0$).
3. **Lossy JPEG Compression:** Transmission and tile storage compression artifacts (Quality $50$).
4. **H&E Stain Density Variation:** Optical density decomposition in Beer-Lambert space using the normalized Ruifrok-Johnston / scikit-image HED matrix, scaling Hematoxylin and Eosin channels by $\times 1.1$ (+10% stain uptake) while holding DAB constant.

### Key Empirical Findings

- **Exceptional Stain Invariance:** The pathology-pretrained Phikon representation exhibits remarkable resilience to moderate H&E stain concentration shifts. Under the $+10\%$ HED stain perturbation, accuracy degraded by only **$-0.64$ percentage points (pp)** (from $66.38\%$ to $65.74\%$), macro-F1 dropped by only **$-0.65\text{ pp}$** ($66.54\%$ to $65.89\%$), and the prediction label-flip rate was only **$2.57\%$** (12 out of 467 tiles). Neither McNemar's test ($p=0.55$) nor paired bootstrap testing ($95\%\text{ CI}: [-2.14, +0.86]\text{ pp}$, $p=0.46$) showed statistically significant degradation.
- **Robustness to Mild Optical Blur:** Under mild Gaussian blur ($\sigma=1.0$), performance showed negligible degradation: **$-0.64\text{ pp}$** in accuracy and **$-1.18\text{ pp}$** in macro-F1, with a high prediction consistency of **$90.58\%$** ($p=0.72$ McNemar; bootstrap $95\%\text{ CI}: [-3.00, +1.71]\text{ pp}$).
- **Vulnerability to Compression & Illumination Shifts:** In contrast, lossy JPEG compression (Quality 50) and global brightness/contrast shifts caused statistically significant degradation:
  - **JPEG Compression:** Accuracy fell by **$-4.28\text{ pp}$** ($66.38\% \to 62.10\%$, McNemar $p=0.0097$, bootstrap $95\%\text{ CI}: [-7.28, -1.29]\text{ pp}$), with a label-flip rate of **$16.70\%$** (78 tiles). Smooth muscle (`MUS`) recall dropped severely by $-18.97\text{ pp}$ due to high-frequency block artifacts disrupting muscle fiber texture.
  - **Brightness & Contrast:** Accuracy fell by **$-3.64\text{ pp}$** ($66.38\% \to 62.74\%$, McNemar $p=0.022$, bootstrap $95\%\text{ CI}: [-6.42, -0.64]\text{ pp}$), with a label-flip rate of **$14.99\%$** (70 tiles). Mucus (`MUC`) recall collapsed by **$-34.48\text{ pp}$** ($62.07\% \to 27.59\%$), as brightened pale mucinous areas were conflated with adipose tissue (`ADI`).

---

## 2. Evaluation Dataset and Selection Protocol

In strict accordance with the Phase 5–12 Blueprint and the Phase 4.4 Reduced benchmark specification:
- **Source Manifest:** `research/foundation_models/cache/subset10pct/manifest_subset.csv`.
- **Target Population:** Every single image assigned to the held-out test split was evaluated ($N=467$ image tiles).
- **Stratification:** The cohort is balanced across all 8 canonical tissue classes:
  - `ADI` (Adipose): 59 tiles
  - `DEB` (Debris): 59 tiles
  - `LYM` (Lymphocytes): 59 tiles
  - `MUC` (Mucus): 58 tiles
  - `MUS` (Muscle): 58 tiles
  - `NOR` (Normal mucosa): 58 tiles
  - `STR` (Stroma): 58 tiles
  - `TUM` (Tumor adenocarcinoma): 58 tiles
- **Integrity Guarantee:** Exactly the same 467 sample IDs, paths, ground-truth labels, and duplicate-group isolation established in earlier phases were maintained. The evaluation manifest was saved to `research/robustness/robustness_subset_manifest.csv`.
- **Clean Reproduction Check:** Prior to evaluating perturbations, the clean predictions of the pre-fitted linear probe on the 467 test embeddings were verified against `phikon_mean_patch_test_predictions.csv`. The match was **$100.0\%$ identical** ($0$ mismatches across all 467 predictions; maximum probability delta $< 5\times 10^{-5}$ due to rounding in CSV storage).

---

## 3. Perturbation Methodology & Parameters

Each perturbation family was implemented with **exactly one predeclared severity level**, without tuning or hyperparameter searching against test labels. All transformations operate on standard $224\times 224$ RGB images and strictly preserve spatial dimensions and color channels.

| Perturbation Family | Implementation Mechanism | Predeclared Severity Parameters | Operational Justification |
|---|---|---|---|
| **Clean Baseline** | Original uncorrupted PNG tiles | No transformation | Ground truth reference |
| **Brightness & Contrast** | `PIL.ImageEnhance` | Brightness factor: $1.1$, Contrast factor: $1.1$ | Simulates variable microscope illumination and sensor exposure |
| **Gaussian Blur** | `cv2.GaussianBlur` | Kernel: $5\times 5$, $\sigma_X=\sigma_Y=1.0$ | Simulates minor out-of-focus plane acquisition |
| **JPEG Compression** | Pillow in-memory encode/decode | Quality level: $50$ (standard lossy) | Simulates image compression during slide transmission and tile storage |
| **H&E Stain Variation** | Beer-Lambert Optical Density decomposition | Ruifrok-Johnston HED matrix; $H \times 1.1$, $E \times 1.1$, DAB $\times 1.0$ | Simulates batch-to-batch variation in tissue staining intensity |

### Mathematical Specification of H&E Optical Density Variation

Following the Beer-Lambert law and the standard Ruifrok & Johnston (2001) color deconvolution framework (implemented identically to `scikit-image`):

1. **Transmission to Optical Density:**
   $$\text{OD} = -\log_{10}\left(\max\left(\frac{I}{I_0}, 10^{-6}\right)\right), \quad I_0 = 1.0$$
2. **Stain Decomposition:**
   Using the normalized HED transformation matrix $M_{\text{HED}} = M_{\text{RGB}\to\text{HED}}^{-1}$:
   $$S = \text{OD} \cdot M_{\text{HED}}$$
   where $S = [S_H, S_E, S_{\text{DAB}}]$.
3. **Controlled Scaling:**
   $$S_H' = 1.10 \times S_H, \quad S_E' = 1.10 \times S_E, \quad S_{\text{DAB}}' = S_{\text{DAB}}$$
4. **Recombination & Inversion:**
   $$\text{OD}_{\text{mod}} = S' \cdot M_{\text{RGB}\to\text{HED}}$$
   $$I_{\text{mod}} = \text{clip}\left(10^{-\text{OD}_{\text{mod}}}, 0.0, 1.0\right) \times 255$$

---

## 4. Overall Benchmark Results

All metrics were computed across all 467 test samples for each of the five experimental conditions.

| Condition | Accuracy (%) | Balanced Acc (%) | Macro-F1 (%) | Weighted-F1 (%) | GT Error Rate (%) | Consistency (%) | Label-Flip Rate (%) | Mean Max Prob | Degradation (pp) |
|---|---|---|---|---|---|---|---|---|---|
| **Clean (Original)** | **66.38%** | **66.39%** | **66.54%** | **66.56%** | **33.62%** | **100.0%** | **0.00%** | 0.6800 | Baseline ($0.0\text{ pp}$) |
| **Gaussian Blur ($\sigma=1.0$)** | 65.74% | 65.73% | 65.37% | 65.38% | 34.26% | 90.58% | 9.42% | 0.6775 | $-0.64\text{ pp}$ |
| **H&E Stain Var. (HED $\times 1.1$)** | 65.74% | 65.75% | 65.89% | 65.91% | 34.26% | 97.43% | 2.57% | 0.6802 | $-0.64\text{ pp}$ |
| **Brightness & Contrast ($\times 1.1$)** | 62.74% | 62.69% | 61.72% | 61.76% | 37.26% | 85.01% | 14.99% | 0.6973 | $-3.64\text{ pp}$ |
| **JPEG Quality 50** | 62.10% | 62.11% | 62.78% | 62.81% | 37.90% | 83.30% | 16.70% | 0.6826 | $-4.28\text{ pp}$ |

> [!IMPORTANT]
> **Mathematical Distinction: Label-Flip Rate vs. Ground-Truth Error Rate**  
> - **Prediction Consistency** is the proportion of samples where $\hat{y}_{\text{perturbed}} = \hat{y}_{\text{clean}}$.  
> - **Label-Flip Rate** is the proportion of samples where $\hat{y}_{\text{perturbed}} \neq \hat{y}_{\text{clean}}$ ($\text{Consistency} + \text{Flip Rate} = 1.0$).  
> - **Ground-Truth Error Rate** is the proportion of samples where $\hat{y}_{\text{perturbed}} \neq y_{\text{true}}$ ($\text{Accuracy} + \text{Error Rate} = 1.0$).  
> For example, under H&E stain variation, the ground-truth error rate is $34.26\%$, but only $2.57\%$ of predictions flipped relative to the clean model! This proves that most errors on the stain-varied images were already errors on the clean images, demonstrating outstanding model stability.

---

## 5. Statistical Significance Testing (Paired Tests vs. Clean)

To evaluate whether the observed performance changes reflect genuine shifts rather than random sampling fluctuations, two rigorous paired statistical tests were computed on the identical 467 test samples:
1. **McNemar's Paired Test** (with continuity correction) on the binary classification correctness contingency table ($n_{00}, n_{01}, n_{10}, n_{11}$).
2. **Paired Bootstrap Test** ($B=2,000$ resamples, random seed 42) computing $95\%$ percentile confidence intervals and two-sided empirical $p$-values for $\Delta \text{Accuracy}$, $\Delta \text{Macro-F1}$, and $\Delta \text{Balanced Accuracy}$.

| Perturbation Condition | McNemar Discordant ($n_{01} / n_{10}$) | McNemar $\chi^2$ | McNemar $p$-value | Bootstrap $\Delta \text{Acc}$ (95% CI) | Bootstrap $\Delta \text{Acc } p$ | Significant at $\alpha=0.05$? |
|---|---|---|---|---|---|---|
| **Gaussian Blur** | $14 / 17$ | $0.1290$ | $0.7194$ | $[-0.0300, +0.0171]$ | $0.657$ | **No** (Robust) |
| **H&E Stain Variation** | $4 / 7$ | $0.3636$ | $0.5465$ | $[-0.0214, +0.0086]$ | $0.461$ | **No** (Robust) |
| **Brightness & Contrast** | $16 / 33$ | $5.2245$ | $0.0223$ | $[-0.0642, -0.0064]$ | $0.019$ | **Yes** (Significant drop) |
| **JPEG Quality 50** | $17 / 37$ | $6.6852$ | $0.0097$ | $[-0.0728, -0.0129]$ | $0.007$ | **Yes** (Significant drop) |

### Additional Bootstrap Results for F1 and Balanced Accuracy

- **Brightness & Contrast:**
  - $\Delta \text{Macro-F1}$: $-4.83\text{ pp}$ ($95\%\text{ CI}: [-7.64, -1.90]\text{ pp}$, $p=0.001$).
  - $\Delta \text{Balanced Accuracy}$: $-3.70\text{ pp}$ ($95\%\text{ CI}: [-6.34, -0.98]\text{ pp}$, $p=0.005$).
- **JPEG Quality 50:**
  - $\Delta \text{Macro-F1}$: $-3.77\text{ pp}$ ($95\%\text{ CI}: [-6.72, -0.82]\text{ pp}$, $p=0.013$).
  - $\Delta \text{Balanced Accuracy}$: $-4.28\text{ pp}$ ($95\%\text{ CI}: [-7.31, -1.30]\text{ pp}$, $p=0.009$).

---

## 6. Per-Class Robustness and Vulnerability Analysis

The eight canonical tissue classes exhibited distinct resilience and vulnerability profiles under each perturbation family:

| Class | Clean Recall (%) | Blur Recall (%) | Blur $\Delta$ (pp) | Stain Recall (%) | Stain $\Delta$ (pp) | B/C Recall (%) | B/C $\Delta$ (pp) | JPEG Recall (%) | JPEG $\Delta$ (pp) |
|---|---|---|---|---|---|---|---|---|---|
| **ADI** (Adipose) | 72.88% | 77.97% | $+5.08$ | 69.49% | $-3.39$ | 91.53% | $+18.64$ | 69.49% | $-3.39$ |
| **DEB** (Debris) | 55.93% | 49.15% | $-6.78$ | 55.93% | $0.00$ | 54.24% | $-1.69$ | 49.15% | $-6.78$ |
| **LYM** (Lymphocytes) | 66.10% | 76.27% | $+10.17$ | 64.41% | $-1.69$ | 64.41% | $-1.69$ | 61.02% | $-5.08$ |
| **MUC** (Mucus) | 62.07% | 48.28% | $-13.79$ | 62.07% | $0.00$ | 27.59% | **$-34.48$** | 58.62% | $-3.45$ |
| **MUS** (Muscle) | 67.24% | 63.79% | $-3.45$ | 65.52% | $-1.72$ | 65.52% | $-1.72$ | 48.28% | **$-18.97$** |
| **NOR** (Normal Mucosa) | 67.24% | 72.41% | $+5.17$ | 68.97% | $+1.72$ | 55.17% | $-12.07$ | 63.79% | $-3.45$ |
| **STR** (Stroma) | 68.97% | 68.97% | $0.00$ | 67.24% | $-1.72$ | 70.69% | $+1.72$ | 82.76% | $+13.79$ |
| **TUM** (Tumor) | 70.69% | 68.97% | $-1.72$ | 72.41% | $+1.72$ | 72.41% | $+1.72$ | 63.79% | $-6.90$ |

### Class-Specific Sensitivity Observations

1. **Tumor (`TUM`) Stability:** Tumor tissue classification remained remarkably stable under H&E stain variation ($72.41\%$, $+1.72\text{ pp}$), brightness/contrast ($72.41\%$, $+1.72\text{ pp}$), and Gaussian blur ($68.97\%$, $-1.72\text{ pp}$). It experienced a moderate decline only under JPEG compression ($63.79\%$, $-6.90\text{ pp}$).
2. **Mucus (`MUC`) Vulnerability to Brightness:** Mucus was extremely sensitive to brightness and contrast enhancement ($\Delta \text{Recall} = -34.48\text{ pp}$, dropping to $27.59\%$). Examination of predictions revealed that brightened mucinous pools lost delicate extracellular matrix gradients and were misclassified as empty lipid vacuoles (`ADI`).
3. **Smooth Muscle (`MUS`) Vulnerability to JPEG:** Muscle tissue showed high sensitivity to JPEG compression ($\Delta \text{Recall} = -18.97\text{ pp}$, dropping to $48.28\%$). $8\times 8$ discrete cosine transform block artifacts interfered with the parallel fibrillar striations characteristic of smooth muscle, leading to misclassification into stroma (`STR`).
4. **Debris (`DEB`) Baseline Difficulty:** Debris had the lowest baseline recall ($55.93\%$) and showed mild sensitivity to blur and compression (falling to $49.15\%$).

---

## 7. Visualizations

### 7.1 Multi-Tissue Contact Sheet Across Perturbations

The contact sheet displays one representative tile per canonical class across the clean baseline and all four perturbations, with true class labels and model predictions annotated.

![Robustness Contact Sheet](file:///C:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/robustness/contact_sheet_robustness.png)

### 7.2 Overall Performance Comparison Plot

The grouped bar chart compares Accuracy, Balanced Accuracy, and Macro-F1 across all five conditions, highlighting baseline reference thresholds.

![Overall Robustness Plot](file:///C:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/robustness/overall_robustness_plot.png)

### 7.3 Per-Class Sensitivity Breakdown Plot

The per-class recall plot illustrates tissue-specific vulnerability patterns across the four perturbation families.

![Per-Class Robustness Plot](file:///C:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/robustness/per_class_robustness.png)

---

## 8. Computational Cost & Resource Tracking

All evaluations were executed on the local CPU workstation (Windows 11, Intel CPU, 10 PyTorch inference threads) without GPU acceleration.

- **Total Execution Runtime:** $829.75\text{ seconds}$ ($\approx 13.8\text{ minutes}$).
- **Inference Throughput:**
  - Brightness & Contrast: $2.44\text{ images/second}$ ($191.3\text{ s}$ for 467 tiles).
  - Gaussian Blur: $2.47\text{ images/second}$ ($188.9\text{ s}$ for 467 tiles).
  - JPEG Compression: $2.47\text{ images/second}$ ($189.3\text{ s}$ for 467 tiles).
  - H&E Stain Variation: $2.52\text{ images/second}$ ($185.7\text{ s}$ for 467 tiles).
- **Peak RAM Footprint:** Kept under $1.5\text{ GB}$ throughout by processing in memory-bounded batches of 16 images and extracting embeddings into preallocated $467 \times 768$ float32 arrays ($1.43\text{ MB}$ each).
- **Disk Footprint:** All generated Phase 6 artifacts (CSVs, JSONs, 3 visualization PNGs) consume $<15\text{ MB}$ total.

---

## 9. Literature Context & Scientific Discussion

### Optical Density & Stain Decomposition
Stain variability in computational pathology arises primarily from differences in tissue thickness, fixation duration, stain manufacturer formulations, and staining protocol timing. As shown by **Macenko et al. (2009)** and **Ruifrok & Johnston (2001)**, light attenuation through stained histology sections is linear in optical density (OD) space rather than linear RGB space. Perturbations applied directly to RGB channels (such as generic HSV jitter) distort the physical absorption spectrum of Hematoxylin and Eosin. By applying perturbations in Beer-Lambert optical density space via the calibrated HED matrix, our Phase 6 stain perturbation accurately simulates biological variation in dye uptake while keeping counterstains intact.

### Foundation Model Invariance
The finding that Phikon is highly resilient to stain variation ($\Delta \text{Acc} = -0.64\text{ pp}$, flip rate $2.57\%$) aligns with recent findings in computational pathology literature (**Tellez et al., 2019 / PMC6778842**). Because Phikon was pretrained using the iBOT self-supervised objective on over 40 million pan-cancer tiles from The Cancer Genome Atlas (TCGA)—which inherently encompasses diverse scanning hardware and multi-institutional staining variations—its learned spatial-token representations have developed substantial intrinsic stain invariance.

### Scientific Scope and Critical Limitations
1. **Synthetic vs. Multi-Center Validation:** Controlled mathematical perturbations provide informative stress tests of model stability, but **they do not constitute clinical validation or external multi-center domain generalization**. Real multi-scanner cohorts involve complex optical aberrations, sensor noise characteristics, and biological variations that synthetic transformations only partially capture.
2. **Patch-Level Classification Scope:** All findings pertain to independent $224\times 224$ image tiles from the GCHTID dataset. No claims are made regarding whole-slide slide-level aggregation, patient survival prediction, or clinical diagnosis.
3. **Single Severity Constraint:** As mandated by the Phase 5–12 Blueprint, a single predeclared severity level was tested per family to avoid unprincipled post-hoc tuning. Future work could map continuous degradation curves across parametric sweeps.

---

## 10. Audit Trail & Deliverables

The complete set of Phase 6 code, evaluation data, metrics, and visualization deliverables is archived under `research/robustness/`:

- `research/robustness/perturbations.py`: Deterministic transformation functions for all four perturbation families.
- `research/robustness/pipeline.py`: End-to-end evaluation, inference, metric computation, and visualization script.
- `research/robustness/config.json`: Complete parameter specification and execution metadata.
- `research/robustness/robustness_subset_manifest.csv`: Manifest of the 467 evaluated test tiles with class indices and paths.
- `research/robustness/predictions_robustness.csv`: Sample-by-sample predictions across all 5 conditions ($2,335$ rows).
- `research/robustness/predictions_robustness.json`: JSON format of complete per-sample prediction records.
- `research/robustness/metrics_robustness.json`: Aggregate metrics, per-class sensitivity/recall, degradation values, McNemar tests, and paired bootstrap confidence intervals.
- `research/robustness/contact_sheet_robustness.png`: 8-class $\times$ 5-condition visual contact sheet ($11.7\text{ MB}$).
- `research/robustness/overall_robustness_plot.png`: Publication-ready grouped bar chart of overall metrics.
- `research/robustness/per_class_robustness.png`: Publication-ready plot of per-class sensitivity degradation.
- `tests/test_phase6.py`: 9 automated tests verifying manifest coverage, metric identities, numerical validity, and artifact integrity.

