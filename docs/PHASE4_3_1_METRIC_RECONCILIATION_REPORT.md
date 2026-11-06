# GastricPath AI — Phase 4.3.1 Metric Reconciliation & Scientific Certification Report

**Document ID:** `GP-AUDIT-PHASE4-3-1-RECONCILIATION-2026-V1`  
**Execution Phase:** Phase 4.3.1 (Metric Reconciliation, Nonlinear Probe Audit & Statistical Certification)  
**Target Repository:** `gastric-cancer-histopathology-dashboard`  
**Git Branch:** `gastricpath-ai-evolution`  
**Audit Base Commit:** `8dbdce6441c779f3eeddc75687c4ed829ca4233b` (`phase4.3: frozen DINOv2 nonlinear probing`)  
**Status:** Certified & Verified (Full Numerical Consistency Established)  

---

## 1. Executive Summary & Audit Mandate

Phase 4.3.1 was commissioned as an independent forensic audit and numerical reconciliation pass following the completion of Phase 4.3. The objectives of this phase were:
1. **Per-Class and Aggregate Metric Table Reconciliation:** Independently recompute all per-class metrics and aggregate rows from the raw persistent test prediction artifacts to eliminate all manual transcription discrepancies and ensure mathematical consistency.
2. **Phase 4.2 Linear Probe Metric Reconciliation:** Trace and resolve differences in linear-probe AUROC, ECE, and Brier score between Phase 4.2/4.2.1 and Phase 4.3 documentation.
3. **Historical CNN-RNN Baseline Reconciliation:** Investigate and document the provenance of the historical baseline figures, formally clarifying the distinct evaluation cohorts ($N=3,110$ vs. $N=4,665$ vs. $N=442$) and addressing unverified figures.
4. **Calibration Protocol Audit:** Document the exact mathematical conventions governing Expected Calibration Error (ECE) and multiclass Brier score calculation.
5. **Paired Statistical Comparison:** Execute paired bootstrap resampling ($B=2,000$) and McNemar tests over the identical $N=4,665$ test samples to establish confidence intervals for metric differences.
6. **Calibration of Scientific Terminology:** Ensure that all reporting strictly distinguishes observed sample statistics from broader clinical or biological hypotheses.

---

## 2. Reconciled Per-Class Metric Table

The table below was computed by the deterministic audit script ([`research/foundation_models/audit_phase4_3_metrics.py`](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/foundation_models/audit_phase4_3_metrics.py)) directly from the persistent test prediction CSV files (`nonlinear_probe_cls_test_predictions.csv` and `nonlinear_probe_mean_patch_test_predictions.csv`) evaluated on the Phase 2 test split ($N = 4,665$).

| Class Code | Class Name | CLS Precision | CLS Recall | CLS Specificity | CLS F1 | Mean-Patch Precision | Mean-Patch Recall | Mean-Patch Specificity | Mean-Patch F1 | Support |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ADI** | Adipose Tissue | 81.55% | 70.50% | 97.72% | 75.62% | 79.96% | 73.93% | 97.35% | 76.83% | 583 |
| **DEB** | Necrotic Debris | 53.02% | 63.29% | 91.99% | 57.70% | 55.11% | 62.95% | 92.68% | 58.77% | 583 |
| **LYM** | Lymphocytes | 80.95% | 75.81% | 97.45% | 78.30% | 82.01% | 74.27% | 97.67% | 77.95% | 583 |
| **MUC** | Mucus Lakes | 64.98% | 61.64% | 95.25% | 63.27% | 57.87% | 70.55% | 92.65% | 63.58% | 584 |
| **MUS** | Smooth Muscle | 75.00% | 64.32% | 96.94% | 69.25% | 67.27% | 70.50% | 95.10% | 68.84% | 583 |
| **NOR** | Normal Mucosa | 51.57% | 50.77% | 93.19% | 51.17% | 55.79% | 50.43% | 94.29% | 52.97% | 583 |
| **STR** | Cancer Stroma | 60.89% | 56.09% | 94.86% | 58.39% | 61.96% | 51.97% | 95.44% | 56.53% | 583 |
| **TUM** | Tumor Epithelium | 57.03% | 73.76% | 92.06% | 64.32% | 64.08% | 65.18% | 94.78% | 64.63% | 583 |
| **Macro Average** | — | **65.62%** | **64.52%** | **94.93%** | **64.75%** | **65.50%** | **64.97%** | **95.00%** | **65.01%** | **4,665** |
| **Weighted Average** | — | **65.62%** | **64.52%** | **94.93%** | **64.75%** | **65.50%** | **64.97%** | **95.00%** | **65.01%** | **4,665** |

### Mathematical Identity and Integrity Verification:
1. **Macro Recall == Balanced Accuracy:**
   - CLS MLP: Macro Recall = $64.5225\% \equiv 64.52\%$; Balanced Accuracy = $64.5225\% \equiv 64.52\%$. Difference $< 10^{-14}$.
   - Mean Patch MLP: Macro Recall = $64.9725\% \equiv 64.97\%$; Balanced Accuracy = $64.9725\% \equiv 64.97\%$. Difference $< 10^{-14}$.
2. **Weighted Recall == Overall Accuracy:**
   - CLS MLP: $\sum \text{TP} / N = 3,010 / 4,665 = 64.5230\% \equiv 64.52\%$.
   - Mean Patch MLP: $\sum \text{TP} / N = 3,031 / 4,665 = 64.9732\% \equiv 64.97\%$.
3. **One-vs-Rest Specificity Formula:**
   - Evaluated as $\text{Specificity}_k = \frac{\text{TN}_k}{\text{TN}_k + \text{FP}_k}$, where $\text{TN}_k$ is the count of non-$k$ samples correctly predicted as non-$k$, and $\text{FP}_k$ is the count of non-$k$ samples incorrectly predicted as $k$.
   - For every class $k$, $\text{TP}_k + \text{FP}_k + \text{FN}_k + \text{TN}_k \equiv 4,665$.
4. **Resolution of Chat Transcription Discrepancy:**
   - In the chat interface summary generated at the conclusion of Phase 4.3, an unverified markdown table was output displaying class supports of `581`, `584`, and `585`, and recall figures that did not average to the Macro Recall row.
   - Forensic inspection proved that the underlying serialized files (`nonlinear_probe_cls_results.json`, `nonlinear_probe_mean_patch_results.json`, and both prediction CSVs) have always contained the correct class supports ($583 \times 7 + 584 = 4,665$) and consistent metrics. The table above is certified as the single canonical per-class benchmark for Phase 4.3.

---

## 3. Reconciliation of Metric Discrepancies

### A. Linear Probe Metrics Audit (Phase 4.2 vs. Phase 4.3)
An audit was conducted to reconcile the linear-probe metrics across Phase 4.2, Phase 4.2.1, Phase 4.3, and independent recomputation:

| Metric | Model Feature | Phase 4.2 Original | Phase 4.2.1 Certified | Phase 4.3 Report (`PHASE4_3_NONLINEAR_PROBING_REPORT.md`) | Chat Interface Table (Phase 4.3 Turn) | Independently Recomputed (Phase 4.3.1 Audit) | Discrepancy Explanation | Canonical Certified Value | Source Artifact |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|:---:|:---|
| **Accuracy** | CLS Linear | 64.35% | 64.35% | 64.35% | 64.35% | 64.35% | Exact agreement across all phases | **64.35%** | `linear_probe_cls_test_predictions.csv` |
| **Accuracy** | Mean Patch Linear | 65.08% | 65.08% | 65.08% | 65.08% | 65.08% | Exact agreement across all phases | **65.08%** | `linear_probe_mean_patch_test_predictions.csv` |
| **Macro-F1** | CLS Linear | 64.21% | 64.21% | 64.21% | 64.21% | 64.21% | Exact agreement across all phases | **64.21%** | `linear_probe_cls_test_predictions.csv` |
| **Macro-F1** | Mean Patch Linear | 65.03% | 65.03% | 65.03% | 65.03% | 65.03% | Exact agreement across all phases | **65.03%** | `linear_probe_mean_patch_test_predictions.csv` |
| **Macro OvR AUROC** | CLS Linear | 0.9293 | 0.9293 | 0.9293 | 0.9329 | 0.9293 | Chat table transposed/transcribed an unverified figure; file artifact contains 0.9293 | **0.9293** | `linear_probe_cls_test_predictions.csv` |
| **Macro OvR AUROC** | Mean Patch Linear | 0.9316 | 0.9316 | 0.9316 | 0.9358 | 0.9316 | Chat table transcribed an unverified figure; file artifact contains 0.9316 | **0.9316** | `linear_probe_mean_patch_test_predictions.csv` |
| **ECE (10 bins)** | CLS Linear | 0.0220 | 0.0220 / 0.0217 | 0.0220 | 0.0321 | 0.0217 | Chat table transcribed an unverified figure. $0.0220$ is full float32 in-memory; $0.0217$ is 4-decimal CSV rounded | **0.0217** (CSV) / **0.0220** (In-Memory) | `linear_probe_cls_test_predictions.csv` |
| **ECE (10 bins)** | Mean Patch Linear | 0.0368 | 0.0368 | 0.0368 | 0.0368 | 0.0368 | Exact agreement across all phases | **0.0368** | `linear_probe_mean_patch_test_predictions.csv` |
| **Brier Score** | CLS Linear | 0.4819 | 0.4819 | — | 0.4839 | 0.4819 | Chat table transcribed an unverified figure; file artifact contains 0.4819 | **0.4819** | `linear_probe_cls_test_predictions.csv` |
| **Brier Score** | Mean Patch Linear | 0.4730 | 0.4730 | — | 0.4800 | 0.4730 | Chat table transcribed an unverified figure; file artifact contains 0.4730 | **0.4730** | `linear_probe_mean_patch_test_predictions.csv` |

### B. Historical CNN-RNN Baseline Reconciliation
An investigation was conducted into the historical baseline accuracy figures:

1. **The Phase 3.3 Historical Test Cohort ($N = 3,110$):**
   - Evaluated on the original test partition `historical_test_manifest.csv`.
   - Verified Accuracy: **65.92%** (Wilson 95% CI: $[64.24\%, 67.57\%]$).
   - Strictly duplicate-unexposed subset ($N=2,918$): **65.66%** (Wilson 95% CI: $[63.92\%, 67.36\%]$).
   - Provenance: Checkpoint `gastric_rnn_classifier.pt` (SHA-256 `65369ef6...`), predictions serialized in `research/audits/phase3_3/historical_test_predictions.csv`.
2. **The Phase 3.1 / Phase 2 Test Cohort ($N = 4,665$):**
   - Evaluated on the Phase 2 test partition `research/splits/manifest_test.csv`.
   - Reported Accuracy: **71.04%**.
   - Provenance: Serialized in `research/experiments/exp_20261105_221027_2463a2de/predictions_test.csv`.
   - **Forensic Caveat (Phase 3.2 Audit):** 80.56% of this cohort ($3,758$ images) were in the checkpoint's gradient training split. On the genuinely unseen subset ($N=442$ images), historical checkpoint accuracy was **65.84%**.
3. **Investigation of the 55.03% Figure:**
   - A search of all JSON, CSV, log, and markdown files across the entire git repository confirms that the figure `55.03%` does not exist in any serialized artifact.
   - It originated solely as a typing hallucination inside the chat response markdown table of the previous turn.
   - **Audit Verdict:** The figure `55.03%` is formally certified as **UNVERIFIED AND SPURIOUS**. It is permanently removed from all benchmark tables.

---

## 4. Calibration Audit & Protocol Specification

### Mathematical Formulation of Metrics:
1. **Expected Calibration Error (ECE):**
   - Partition: $M = 10$ equal-width bins over $[0, 1]$: $B_m = [\frac{m-1}{M}, \frac{m}{M})$ for $m \in \{1, \dots, M-1\}$ and $B_M = [\frac{M-1}{M}, 1.0]$.
   - Sample Confidence: $\hat{p}_i = \max_{k \in \{0, \dots, 7\}} p_{ik}$.
   - Sample Prediction: $\hat{y}_i = \arg\max_{k \in \{0, \dots, 7\}} p_{ik}$.
   - Bin Accuracy: $\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbf{1}(\hat{y}_i = y_i)$.
   - Bin Confidence: $\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$.
   - Empty Bins: If $|B_m| = 0$, contribution is strictly $0.0$.
   - Aggregation: $\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$.
2. **Multiclass Brier Score:**
   - Evaluated using the standard sum over classes convention:
     $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K (p_{ik} - y_{ik})^2$$
     where $y_{ik} \in \{0, 1\}$ is the one-hot indicator of the ground-truth class, and $K = 8$.

### Calibration Results Summary:
| Model / Readout | Expected Calibration Error (ECE) | Multiclass Brier Score | Relative ECE Change vs. Linear |
|:---|:---:|:---:|:---:|
| **CLS Linear Probe** | $0.0217$ (CSV) / $0.0220$ (In-Memory) | $0.4819$ | Baseline CLS |
| **CLS Nonlinear MLP Probe** | $0.0261$ | $0.4736$ | $+20.3\%$ (Slightly worse calibration) |
| **Mean Patch Linear Probe** | $0.0368$ | $0.4730$ | Baseline Mean Patch |
| **Mean Patch Nonlinear MLP Probe** | **$0.0171$** | **$0.4651$** | **$-53.5\%$ (Substantial calibration improvement)** |

*Scientific Note:* While the Mean Patch MLP achieves a $53.5\%$ relative reduction in ECE, lower calibration error on an image-level test cohort does not constitute evidence of clinical diagnostic safety or pathologist-level reliability.

---

## 5. Paired Statistical Comparison on Identical Test Samples ($N=4,665$)

Because all four probes were evaluated on the identical test split ($N = 4,665$), paired resampling procedures were conducted ($B = 2,000$ paired bootstrap iterations, seed = $42$) alongside McNemar's test:

| Paired Comparison ($A$ vs. $B$) | $\Delta \text{Accuracy}$ ($A - B$) [95% Bootstrap CI] | Bootstrap $p$-value (Acc) | McNemar $\chi^2$ (Discordant Counts) | McNemar $p$-value (Acc) | $\Delta \text{Macro-F1}$ ($A - B$) [95% Bootstrap CI] | Bootstrap $p$-value (F1) | Statistical Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **CLS MLP vs. CLS Linear** | $+0.16\%$ [$-0.84\%, +1.12\%$] | $0.7590$ | $\chi^2=0.09$ ($n_{01}=264, n_{10}=272$) | $0.7624$ | $+0.53\%$ [$-0.47\%, +1.49\%$] | $0.3130$ | **No statistically significant difference** |
| **Mean Patch MLP vs. Mean Patch Linear** | $-0.10\%$ [$-1.18\%, +0.99\%$] | $0.8650$ | $\chi^2=0.03$ ($n_{01}=317, n_{10}=312$) | $0.8733$ | $-0.02\%$ [$-1.09\%, +1.07\%$] | $0.9820$ | **No statistically significant difference** |
| **Mean Patch MLP vs. CLS MLP** | $+0.46\%$ [$-0.60\%, +1.54\%$] | $0.4390$ | $\chi^2=0.60$ ($n_{01}=321, n_{10}=342$) | $0.4373$ | $+0.26\%$ [$-0.75\%, +1.31\%$] | $0.6290$ | **No statistically significant difference** |
| **Mean Patch Linear vs. CLS Linear** | $+0.71\%$ [$-0.30\%, +1.74\%$] | $0.1910$ | $\chi^2=1.75$ ($n_{01}=293, n_{10}=327$) | $0.1851$ | $+0.81\%$ [$-0.21\%, +1.85\%$] | $0.1300$ | **No statistically significant difference** |

### Statistical Insights:
- In all four pairwise comparisons, the 95% paired bootstrap confidence interval spans zero.
- Both the bootstrap empirical $p$-values and McNemar's exact discordant tests exceed $0.10$ by a wide margin.
- There is no statistical evidence that a 2-layer MLP improves overall accuracy over a linear probe on frozen DINOv2 representations for this dataset split.

---

## 6. Evidence-Aligned Scientific Conclusions

To maintain strict alignment with scientific reporting integrity:

> **Certified Scientific Statement:**  
> "On the evaluated image-level cluster-isolated GCHTID split, compact nonlinear heads provided negligible changes in overall classification accuracy relative to linear probing of frozen DINOv2 features ($\Delta = +0.16\%$ for CLS, $p=0.76$; $\Delta = -0.10\%$ for Mean Patch, $p=0.87$). Some class-specific F1 changes (notably $+3.50\%$ in Mucus Lakes on CLS) and calibration differences (Mean Patch MLP ECE $0.0171$ vs. Linear $0.0368$) were observed. These findings apply strictly to the evaluated frozen representations, hyperparameter configurations, and image-level test cohort."

### Strict Prohibitions Affirmed:
- No claim of patient-level generalization or clinical readiness is made.
- No claim of encoder superiority over pathology-specific models is asserted.
- All per-class explanations remain plausible morphological hypotheses rather than proven causal mechanisms.
