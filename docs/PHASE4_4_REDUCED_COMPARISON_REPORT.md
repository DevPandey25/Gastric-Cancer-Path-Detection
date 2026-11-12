# GastricPath AI — Phase 4.4 Reduced Comparative Evaluation Report
## Pathology-Specific Foundation Model (Phikon) vs. Generic Foundation Model (DINOv2) on 10% Stratified GCHTID Benchmark

**Document ID:** `GP-REPORT-4.4-REDUCED-V2`  
**Phase:** 4.4 Reduced Runtime Benchmark  
**Date:** November 2026  
**Status:** Certified & Reproducible  
**Benchmark Scope:** 10% Stratified, Cluster-Isolated Research Subset ($N=3,110$ tiles; not a full-dataset evaluation)  
**Evaluated Encoders:**
1. **Owkin Phikon** (`owkin/phikon`): ViT-B/16, TCGA pan-cancer iBOT pretraining, $d=768$, non-commercial academic license
2. **Meta AI DINOv2** (`dinov2_vits14`): ViT-S/14, LVD-142M general visual pretraining, $d=384$, Apache 2.0 license

---

## Executive Summary

Phase 4.4 evaluated whether domain-specific foundation pretraining on clinical whole-slide histopathology images yields more linearly separable representations for gastric tissue classification compared with general visual pretraining. 

To maintain computational feasibility and eliminate unnecessary environmental/runtime costs, evaluation was performed on a reproducible **10% stratified, cluster-isolated subset** ($N=3,110$ tiles: 2,177 Train, 466 Val, 467 Test) with strict sample-for-sample alignment across both encoders.

### Key Empirical Findings:

1. **Phikon Performance on the Reduced GCHTID Benchmark**:
   - Linear probing on frozen **Phikon Mean Patch** representations achieved **66.38% test accuracy** [95% Wilson CI: 61.98%–70.52%] and **66.54% Macro-F1** [95% Bootstrap CI: 61.58%–70.85%].
   - Linear probing on frozen **DINOv2 Mean Patch** representations achieved **59.31% test accuracy** [95% Wilson CI: 54.80%–63.68%] and **59.24% Macro-F1** [95% Bootstrap CI: 54.54%–63.69%].
   - **Observed Difference**: Phikon Mean Patch exceeded DINOv2 Mean Patch by **+7.07 percentage points (pp)** in accuracy and **+7.30 pp** in Macro-F1.

2. **Paired Statistical Significance on Identical Test Samples ($N=467$)**:
   - **McNemar Paired Test (continuity-corrected)**:
     - Contingency table: $n_{00} = 119$ (both incorrect), $n_{01} = 71$ (DINOv2 incorrect, Phikon correct), $n_{10} = 38$ (DINOv2 correct, Phikon incorrect), $n_{11} = 239$ (both correct).
     - Discordant pairs: $109$.
     - $\chi^2 = 9.3945$, two-sided $p = 0.00218$ ($p < 0.01$).
   - **Paired Bootstrap ($B=2,000$ resamples, seed 42)**:
     - $\Delta \text{Accuracy}$: **+7.07 pp** [95% CI: +2.78 pp, +11.56 pp] (lower bound strictly $> 0$).
     - $\Delta \text{Macro-F1}$: **+7.30 pp** [95% CI: +3.01 pp, +11.83 pp] (lower bound strictly $> 0$).
     - $\Delta \text{Balanced Accuracy}$: **+7.12 pp** [95% CI: +2.81 pp, +11.61 pp] (lower bound strictly $> 0$).
   - On this 467-tile test cohort, the performance difference between Phikon and DINOv2 mean-patch representations is statistically significant.

3. **Spatial Mean-Patch Pooling Representation**:
   - For both encoders, mean-patch pooling outperformed the classification (`[CLS]`) token:
     - **Phikon**: Mean Patch (**66.38%**) vs CLS (**63.81%**), a $+2.57\text{ pp}$ difference.
     - **DINOv2**: Mean Patch (**59.31%**) vs CLS (**56.96%**), a $+2.35\text{ pp}$ difference.
   - *Technical Specification*: Mean-patch pooling computes the unweighted average across all spatial patch token embeddings ($\frac{1}{P}\sum_{p=1}^P \mathbf{z}_p$). It yields a summary vector of patch-level representations across the tile; **it does not preserve explicit 2D spatial arrangements, relative coordinates, or tissue topology**.

4. **Probability Calibration**:
   - Both Mean Patch representations showed improved calibration compared to CLS: Phikon Mean Patch achieved an Expected Calibration Error (ECE) of **0.0524** and Brier score of **0.4487**, whereas CLS tokens had ECEs of $0.1154$ and $0.1165$.

---

## 1. 10% Stratified Subset Design & Protocol

### 1.1 Subset Construction Rationale
The full GCHTID dataset contains 31,096 images. Full extraction on CPU requires ~3.5 hours. To deliver rigorous, reproducible findings while preserving compute, a 10% subset was drawn under strict invariants:
- **Sample Count**: Exactly 3,110 tiles (10.001% of master dataset).
- **Split Preservation**: Selected strictly within existing Phase 2 splits (Train: 2,177, Val: 466, Test: 467).
- **Near-Duplicate Isolation**: Group boundaries established via dHash / DSU connected components in Phase 2 were strictly maintained; no near-duplicate pairs cross train/val/test splits.
- **Stratification**: All 8 canonical classes are balanced across splits.
- **RNG Reproducibility**: Generated with `seed=42`, serialized to `research/foundation_models/cache/subset10pct/manifest_subset.csv`.

### 1.2 Class and Split Distribution

| Class Name | Tissue Subtype | Train ($N=2,177$) | Val ($N=466$) | Test ($N=467$) | Total Subset ($N=3,110$) | Master Class Balance |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **ADI** | Adipose Tissue | 272 | 59 | 59 | 390 | 12.54% |
| **DEB** | Necrotic Debris | 272 | 58 | 59 | 389 | 12.51% |
| **LYM** | Lymphocytes | 272 | 58 | 59 | 389 | 12.51% |
| **MUC** | Mucus | 272 | 58 | 58 | 388 | 12.48% |
| **MUS** | Smooth Muscle | 272 | 58 | 58 | 388 | 12.48% |
| **NOR** | Normal Mucosa | 272 | 58 | 58 | 388 | 12.48% |
| **STR** | Cancer-Assoc. Stroma | 272 | 59 | 58 | 389 | 12.51% |
| **TUM** | Adenocarcinoma Epithelium | 273 | 58 | 58 | 389 | 12.51% |
| **Total** | — | **2,177** | **466** | **467** | **3,110** | **100.0%** |

---

## 2. Encoder Provenance & Licensing

| Attribute | **Owkin Phikon** (`owkin/phikon`) | **Meta AI DINOv2** (`dinov2_vits14`) |
|:---|:---|:---|
| **Architecture** | Vision Transformer Base (`ViT-B/16`) | Vision Transformer Small (`ViT-S/14`) |
| **Pretraining Domain** | Clinical Pathology (TCGA pan-cancer WSI) | Natural Vision (LVD-142M curated web images) |
| **Pretraining Algorithm** | iBOT (masked image modeling + distillation) | DINOv2 (patch + image-level distillation) |
| **Patch Resolution** | $16 \times 16$ ($196$ spatial patches) | $14 \times 14$ ($256$ spatial patches) |
| **Embedding Dimension ($d$)**| **768** | **384** |
| **Parameter Count** | 86,389,248 (frozen) | 22,056,576 (frozen) |
| **Model Weights SHA-256** | `d01642bfeb683c6b039622eca273e0a8e4208065a4595b63edba178c1c4cfa68` | Verified via official TorchHub cache |
| **License** | **Owkin Non-Commercial Academic License** | **Apache 2.0** |
| **Commercial Use** | **Restricted / Non-commercial research only** | Permitted under Apache 2.0 |
| **Output Representations** | 1. CLS Token (`[768]`)<br>2. Spatial Patch Mean (`[768]`) | 1. CLS Token (`[384]`)<br>2. Spatial Patch Mean (`[384]`) |
| **Normalization** | Unit L2 norm ($\|x\|_2 = 1.0$) across all rows | Unit L2 norm ($\|x\|_2 = 1.0$) across all rows |

---

## 3. Downstream Linear Probing Methodology

To evaluate linear separability and representation utility:
1. **Model**: Multinomial Logistic Regression (`L-BFGS` solver, `max_iter=1000`, `tol=1e-4`, fixed `random_state=42`).
2. **Train Split ($N=2,177$)**: Used exclusively to fit the linear decision boundaries.
3. **Validation Split ($N=466$)**: Used exclusively for hyperparameter tuning. A 12-point regularization grid $C \in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]$ was searched to maximize unweighted Macro-F1.
4. **Test Split ($N=467$)**: Evaluated strictly once after optimal $C^*$ was frozen. No test metrics influenced any hyperparameter choices.

### Validation Split Regularization Tuning Progression (Macro-F1 %):

| Candidate $C$ | Phikon CLS | **Phikon Mean Patch** | DINOv2 CLS | **DINOv2 Mean Patch** |
|:---:|:---:|:---:|:---:|:---:|
| $0.001$ | $50.49\%$ | $55.04\%$ | $31.09\%$ | $31.70\%$ |
| $0.01$ | $53.73\%$ | $57.11\%$ | $37.75\%$ | $37.52\%$ |
| $0.05$ | $56.35\%$ | $59.36\%$ | $40.08\%$ | $41.31\%$ |
| $0.1$ | $57.50\%$ | $61.20\%$ | $44.68\%$ | $44.83\%$ |
| $0.5$ | $65.30\%$ | $65.72\%$ | $52.56\%$ | $52.17\%$ |
| $1.0$ | **$67.06\%$** ($C^*$) | $66.85\%$ | $54.56\%$ | $54.09\%$ |
| $5.0$ | $67.01\%$ | $70.16\%$ | $57.37\%$ | $57.54\%$ |
| $10.0$ | $66.39\%$ | **$71.09\%$** ($C^*$) | $58.86\%$ | **$59.32\%$** ($C^*$) |
| $20.0$ | $66.08\%$ | $70.57\%$ | $57.79\%$ | $58.75\%$ |
| $50.0$ | $64.92\%$ | $69.51\%$ | $58.01\%$ | $58.43\%$ |
| $100.0$ | $64.23\%$ | $69.34\%$ | **$58.97\%$** ($C^*$) | $57.76\%$ |
| $200.0$ | $63.99\%$ | $66.30\%$ | $56.91\%$ | $57.49\%$ |

---

## 4. Test Set Comparative Results ($N=467$)

| Evaluation Metric | **Phikon Mean Patch** | Phikon CLS | **DINOv2 Mean Patch** | DINOv2 CLS | Observed Difference (Phikon vs DINOv2 Patch) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Optimal Regularization ($C^*$)** | $10.0$ | $1.0$ | $10.0$ | $100.0$ | — |
| **Test Accuracy** | **66.38%** | 63.81% | **59.31%** | 56.96% | **+7.07 pp** |
| *Wilson 95% CI* | [61.98%, 70.52%] | [59.36%, 68.04%] | [54.80%, 63.68%] | [52.43%, 61.38%] | *McNemar $p = 0.00218$* |
| **Balanced Accuracy** | **66.39%** | 63.82% | **59.27%** | 56.93% | **+7.12 pp** |
| *Bootstrap 95% CI* | [61.64%, 70.88%] | [59.16%, 68.14%] | [54.76%, 64.02%] | [52.58%, 61.47%] | *Paired CI: [+2.81, +11.61] pp* |
| **Macro-Precision** | **67.33%** | 64.24% | **59.37%** | 57.11% | **+7.96 pp** |
| **Macro-Recall** | **66.39%** | 63.82% | **59.27%** | 56.93% | **+7.12 pp** |
| **Macro-F1** | **66.54%** | 63.75% | **59.24%** | 56.87% | **+7.30 pp** |
| *Bootstrap 95% CI* | [61.58%, 70.85%] | [58.95%, 67.78%] | [54.54%, 63.69%] | [52.31%, 61.13%] | *Paired CI: [+3.01, +11.83] pp* |
| **Weighted-F1** | **66.56%** | 63.77% | **59.30%** | 56.90% | **+7.26 pp** |
| **Cohen's Kappa ($\kappa$)** | **0.6158** | 0.5865 | **0.5350** | 0.5081 | **+0.0808** |
| **Macro OvR AUROC** | **0.9450** | 0.9326 | **0.9196** | 0.9108 | **+0.0254** |
| **Weighted OvR AUROC**| **0.9451** | 0.9327 | **0.9197** | 0.9109 | **+0.0254** |
| **Expected Calibration Error (ECE)** | **0.0524** | 0.1154 | **0.0485** | 0.1165 | *High Calibration* |
| **Multiclass Brier Score** | **0.4487** | 0.4947 | **0.5198** | 0.5562 | **-0.0711** |
| **Inference Time (467 tiles, CPU)** | 4.0 ms | 4.0 ms | 4.0 ms | 2.0 ms | Instantaneous |

---

## 5. Granular Per-Class Performance Breakdown

### 5.1 Per-Class Sensitivity / Recall (%)

| Class | Support | Phikon Mean Patch | Phikon CLS | DINOv2 Mean Patch | DINOv2 CLS | Phikon vs DINOv2 Mean Patch Difference |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ADI** (Adipose) | 59 | **72.88%** | 67.80% | 76.27% | 71.19% | $-3.39\text{ pp}$ |
| **DEB** (Debris) | 59 | **55.93%** | 54.24% | 52.54% | 45.76% | $+3.39\text{ pp}$ |
| **LYM** (Lymphocytes) | 59 | **66.10%** | 67.80% | 71.19% | 66.10% | $-5.09\text{ pp}$ |
| **MUC** (Mucus) | 58 | **62.07%** | 50.00% | 50.00% | 46.55% | **+12.07 pp** |
| **MUS** (Muscle) | 58 | **67.24%** | 68.97% | 53.45% | 55.17% | **+13.79 pp** |
| **NOR** (Normal Mucosa)| 58 | **67.24%** | 63.79% | 51.72% | 53.45% | **+15.52 pp** |
| **STR** (Stroma) | 58 | **68.97%** | 63.79% | 56.90% | 62.07% | **+12.07 pp** |
| **TUM** (Tumor) | 58 | **70.69%** | 74.14% | 62.07% | 55.17% | **+8.62 pp** |
| **Macro Average** | **467** | **66.39%** | **63.82%** | **59.27%** | **56.93%** | **+7.12 pp** |

### 5.2 Per-Class F1 Score (%)

| Class | Support | Phikon Mean Patch | Phikon CLS | DINOv2 Mean Patch | DINOv2 CLS | Phikon vs DINOv2 Mean Patch Difference |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ADI** | 59 | **76.11%** | 68.97% | 75.63% | 70.59% | $+0.48\text{ pp}$ |
| **DEB** | 59 | **59.46%** | 59.26% | 53.91% | 47.79% | $+5.55\text{ pp}$ |
| **LYM** | 59 | **72.90%** | 71.43% | 72.41% | 66.67% | $+0.49\text{ pp}$ |
| **MUC** | 58 | **64.29%** | 54.21% | 53.70% | 50.47% | **+10.59 pp** |
| **MUS** | 58 | **69.03%** | 68.97% | 53.45% | 58.18% | **+15.58 pp** |
| **NOR** | 58 | **61.42%** | 61.67% | 50.85% | 51.67% | **+10.57 pp** |
| **STR** | 58 | **62.50%** | 57.81% | 55.46% | 56.69% | $+7.04\text{ pp}$ |
| **TUM** | 58 | **66.67%** | 67.72% | 58.54% | 52.89% | **+8.13 pp** |
| **Macro Average** | **467** | **66.54%** | **63.75%** | **59.24%** | **56.87%** | **+7.30 pp** |

---

## 6. Confusion Matrix & Diagnostic Boundaries

Inspection of predictions on the 467 test samples reveals the histological bases of errors:
1. **Tumor Epithelium (`TUM`)**:
   - Phikon correctly classified **41 / 58 (70.69%)** tumor tiles. Misclassifications were Stroma (7 tiles) and Debris (5 tiles), aligning with true biological interfaces (desmoplastic reaction and necrosis).
   - DINOv2 correctly identified **36 / 58 (62.07%)** tumor tiles, with errors dispersed across Normal mucosa (7 tiles), Debris (4 tiles), and Lymphocytes (4 tiles).
2. **Normal Mucosa (`NOR`) vs Tumor**:
   - DINOv2 misclassified 13 normal tiles as Tumor.
   - Phikon correctly identified **39 / 58 (67.24%)** normal mucosa tiles, showing sharper glandular discrimination.
3. **Smooth Muscle (`MUS`) and Stroma (`STR`)**:
   - In H&E staining, both muscle bundles and fibrous stroma exhibit pink fibrillar cytoplasm.
   - Phikon Mean Patch achieved an F1 score of **69.03%** on MUS vs DINOv2's **53.45%** (a $+15.58\text{ pp}$ margin), indicating that pathology self-supervision captures fine fibrillar organization.

---

## 7. Comparative Context with Historical Baseline

| Cohort | Model / Representation | Split Details | Test $N$ | Accuracy | Macro-F1 | Notes |
|:---|:---|:---|:---:|:---:|:---:|:---|
| **Historical Baseline (Phase 3.3.1)** | Historical CNN-BiLSTM Checkpoint | Historical Test Split (contains 80% overlap with historical train) | 3,110 | 65.92% | 66.11% | Contaminated historical split |
| **Historical Baseline (Phase 3.3.1)** | Historical CNN-BiLSTM Checkpoint | Strictly Unexposed Cohort (zero train duplicates) | 2,918 | 65.66% | 65.84% | True unexposed historical generalization |
| **Phase 4.4 Reduced (Certified)** | **DINOv2 Mean Patch (Frozen Linear)** | Leak-Proof 10% Stratified Split | 467 | 59.31% | 59.24% | Generic visual foundation model |
| **Phase 4.4 Reduced (Certified)** | **Phikon Mean Patch (Frozen Linear)** | Leak-Proof 10% Stratified Split | 467 | **66.38%** | **66.54%** | Pathology-specific foundation model |

> [!IMPORTANT]
> **Independent Test Population Disclosure**:
> The historical CNN-RNN baseline was evaluated on historical splits from the original repository (which suffered from train-test overlap as established in Phase 3.2). The Phase 4.4 benchmark was conducted on a leak-proof, cluster-isolated 10% stratified subset ($N=467$). These test populations differ; metrics should not be directly equated. However, the result demonstrates that frozen Phikon representations evaluated with a simple linear probe perform comparably to a fully trained, domain-specialized sequence baseline on unseen test samples.

---

## 8. Preserved Invariants & Audit Status

All invariants were verified and remain intact:
1. **Historical Baseline Checkpoint**: `models/checkpoints/gastric_rnn_classifier.pt` (SHA-256 `65369ef6...`, intact).
2. **Full DINOv2 Embedding Cache**: `research/foundation_models/cache/full/dinov2_vits14/` ($31096 \times 384$, intact).
3. **Partial Full Phikon Cache**: `research/foundation_models/cache/full/phikon/` (10,400 images completed, checkpoint preserved).
4. **SQLite Database**: `app.db` (exactly 4 records in `prediction_records`, intact).
5. **Automated Unit & Regression Tests**: 44/44 passing in test suite.

---

## 9. Scientific & Clinical Disclosures

1. **Academic Tile-Level Evaluation**: Evaluates representation separability on static $224 \times 224$ patches.
2. **Missing Patient Linkage**: GCHTID filenames do not encode patient or slide IDs. Patient-level clustering cannot be guaranteed beyond perceptual hash near-duplicate clustering.
3. **Non-Clinical Research Status**: Not approved, cleared, or validated for clinical diagnostic use.
4. **Licensing**: `owkin/phikon` is distributed under an academic research non-commercial license. Commercial deployment requires appropriate licensing from Owkin.

Phase 4.4 is complete, verified, and certified under the reduced 10% stratified benchmark protocol.
