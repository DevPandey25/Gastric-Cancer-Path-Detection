# GastricPath AI — Phase 4.2.1 Audit Report
### Scientific Reporting, Evaluation Lineage & Linear Probe Metric Audit

**Document ID:** `GP-AUDIT-PHASE4-2-1-2026-V1`  
**Execution Phase:** Phase 4.2.1 (Scientific Reporting and Linear Probe Evaluation Audit)  
**Target Architecture:** Frozen Meta AI DINOv2 ViT-S/14 Linear Softmax Probe  
**Historical Baseline Checkpoint:** `models/checkpoints/gastric_rnn_classifier.pt`  
**Baseline SHA-256:** `65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8` (Bitwise Preserved)  
**Git Branch:** `gastricpath-ai-evolution`  
**Date:** November 2026  
**Status:** Certified & Verified (Full Numerical Consistency Established)  

---

## 1. Executive Summary & Audit Mandate

Phase 4.2.1 was commissioned as an independent forensic audit and correction pass on Phase 4.2 ("Frozen Foundation Model Downstream Probing & Linear Evaluation"). The audit mandate encompassed:
1. **Model Specification Audit:** Formally identifying the exact statistical probe architecture that generated the reported benchmark metrics, resolving any ambiguity between the Scikit-Learn L-BFGS implementation and the exploratory PyTorch module.
2. **Execution History & Evaluation Lineage Audit:** Inspecting execution transcripts and task logs to determine the exact chronology of test split evaluation, establishing whether test predictions were reused, and verifying that test split metrics never leaked into or influenced hyperparameter selection.
3. **Independent Metric Recomputation:** Programmatically recomputing all overall and per-class classification metrics directly from stored test prediction CSV files, verifying canonical class alignment, checking probability normalization, and auditing numerical precision.
4. **Correction of Benchmark Reporting Tables:** Regenerating the canonical per-class table for both CLS and Mean Patch representations, including explicit Macro and Weighted summary rows, and eliminating transcription ambiguities.
5. **Scientific Terminology Calibration:** Replacing scientifically overconfident claims of "equivalence" or "outperforming" historical baselines with objective descriptions of numerical proximity, explicitly declaring cohort incommensurability ($N=3,110$ historical test set vs. $N=4,665$ Phase 2 cluster-isolated split), clarifying image-level confidence interval bounds, and reframing histological error explanations strictly as plausible scientific hypotheses.

---

## 2. Final Probe Implementation Verification

The repository contains two linear probe definitions: a Scikit-Learn multinomial logistic regression implementation and an exploratory `PyTorchLinearSoftmax` class. This audit explicitly confirms the single production model responsible for all reported Phase 4.2 results:

### Confirmed Production Model Specification:
- **Software Library:** Scikit-Learn (`sklearn.linear_model.LogisticRegression`)
- **Model Class:** Multinomial Logistic Regression (`multi_class="multinomial"` default in modern Scikit-Learn)
- **Loss Formulation:** Multiclass cross-entropy (negative log-likelihood) with L2 weight regularization:
  $$\mathcal{L}(\mathbf{W}) = -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K y_{ik} \log \left( \frac{\exp(\mathbf{w}_k^\top \mathbf{x}_i + b_k)}{\sum_{j=1}^K \exp(\mathbf{w}_j^\top \mathbf{x}_i + b_j)} \right) + \frac{1}{2C} \sum_{k=1}^K \|\mathbf{w}_k\|_2^2$$
- **Regularization Definition:** Parameter $C$ represents the inverse regularization strength ($C = 1/\lambda$). Smaller values specify stronger penalization; larger values specify weaker penalization.
- **Intercept Handling:** `fit_intercept=True` (Scikit-Learn default). The bias vector $\mathbf{b} \in \mathbb{R}^K$ is unpenalized.
- **Optimization Algorithm:** Limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm (`solver="lbfgs"`), a quasi-Newton second-order method.
- **Convergence Parameters:** Tolerance `tol=1e-4`, maximum iterations `max_iter=1000`, deterministic random seed `random_state=42`.
- **Final Selected Hyperparameters:**
  - **CLS Token Probe:** $C^* = 20.0$ (Selected on Validation Macro-F1 = $65.32\%$, 308 L-BFGS iterations, converged).
  - **Mean Patch Token Probe:** $C^* = 200.0$ (Selected on Validation Macro-F1 = $66.06\%$, 408 L-BFGS iterations, converged).
- **Saved Model Weight Artifacts:**
  - `research/foundation_models/checkpoints/linear_probe_cls.joblib` (Size: 25,543 bytes; coefficient matrix shape: $8 \times 384$; intercept shape: $8$).
  - `research/foundation_models/checkpoints/linear_probe_mean_patch.joblib` (Size: 25,543 bytes; coefficient matrix shape: $8 \times 384$; intercept shape: $8$).

### Clarification on `PyTorchLinearSoftmax`:
The `PyTorchLinearSoftmax` class defined in `research/foundation_models/linear_probe.py` (lines 45–54) was implemented during Phase 4.2 prototyping to verify forward-pass numerical stability and integration compatibility for potential future end-to-end architectures. It was validated in `tests/test_phase4_2.py::test_pytorch_linear_softmax_probe`. However, **`PyTorchLinearSoftmax` was NOT used to produce the reported benchmark metrics**. All reported metrics, predictions, tables, and serialized checkpoint files originate exclusively from the Scikit-Learn L-BFGS model.

---

## 3. Test Evaluation History & Execution Lineage Audit

A forensic inspection of the conversation trajectory, execution transcripts (`transcript.jsonl`), and task logs was conducted to verify that the test split evaluation was leak-free and compliant with rigorous machine learning standards.

### Chronological Execution Timeline:

```
[Phase 4.2 Step 2581-2593] Exploratory tuning sweeps on Train (N=21,767) vs Val (N=4,664)
            │
            ▼
[Task task-2603: 2026-11-06 21:05:42 - 21:08:02]
- Ran full 12-point grid sweep across candidate C in [0.001, ..., 200.0]
- Evaluated strictly on Validation split (N=4,664)
- Selected optimal C* = 20.0 (CLS) and C* = 200.0 (Mean Patch)
- Saved tuning log to: linear_probe_tuning_log.json (Timestamp: 21:07:54)
- Initiated Final Test Evaluation for CLS probe
- Fitted final CLS model on Train (N=21,767) with C*=20.0
- Saved model weights to: linear_probe_cls.joblib (Timestamp: 21:08:02)
- Generated test predictions on Test split (N=4,665)
- CRASH at line 261: KeyError: 'accuracy' during result dictionary formatting
  (Task failed with exit code 1; Mean Patch test evaluation never executed)
            │
            ▼
[Steps 2612 & 2618: Code Correction]
- Fixed dictionary indexing in linear_probe.py (unpacked nested metrics["overall"])
- No changes made to model architecture, seeds, or hyperparameter selection logic
            │
            ▼
[Task task-2621: 2026-11-06 21:09:23 - 21:09:47]
- Reran python -m research.foundation_models.linear_probe
- Detected existing linear_probe_tuning_log.json -> hyperparameter search SKIPPED
- Re-used optimal C* values (C*=20.0, C*=200.0) directly from validation tuning log
- Fitted final CLS probe on Train (N=21,767) -> Evaluated Test (N=4,665)
- Fitted final Mean Patch probe on Train (N=21,767) -> Evaluated Test (N=4,665)
- Saved linear_probe_cls_results.json & linear_probe_cls_test_predictions.csv
- Saved linear_probe_mean_patch_results.json & linear_probe_mean_patch_test_predictions.csv
- Rendered 3 publication-quality visualizations
- Completed successfully with exit code 0
```

### Forensic Findings on Test Integrity:
1. **Which command first generated final test predictions?**
   Final test predictions were generated by `python -m research.foundation_models.linear_probe` executed in background task `task-2621` on November 6, 2026 at 21:09:23 UTC+05:30.
2. **Were test predictions subsequently reused?**
   Yes. The prediction outputs from `task-2621` were serialized to `linear_probe_cls_test_predictions.csv` and `linear_probe_mean_patch_test_predictions.csv`. All subsequent metric summaries, confusion matrices, and visualization plots read directly from these persistent artifacts.
3. **Did test metrics ever influence hyperparameter selection?**
   **Strictly NO.** Forensic analysis of `task-2603.log` proves that the hyperparameter sweep over $C \in [0.001, 200.0]$ completed at `21:07:54`, and the resulting log `linear_probe_tuning_log.json` was written to disk **prior** to any test evaluation. When `task-2621` ran, it detected the existing tuning log and bypassed tuning entirely. The optimal regularization parameters were selected solely based on validation split Macro-F1.
4. **Did the failed command perform inference or only post-processing?**
   In `task-2603`, the model had completed inference on the CLS test split and called `compute_comprehensive_metrics`. However, during post-processing dictionary packaging, a `KeyError: 'accuracy'` occurred because `compute_comprehensive_metrics` returned a nested dictionary (`{"overall": {"accuracy": ...}}`). Mean Patch test evaluation had not yet begun.
5. **Repeated Test Evaluation vs. Test-Driven Hyperparameter Tuning:**
   The repetition of CLS test inference between `task-2603` and `task-2621` was entirely the result of fixing a cosmetic Python dictionary unpacking error in the reporting script. At no point was the test set evaluated iteratively to optimize hyperparameters, adjust architectures, or select thresholds. The protocol was completely leakage-free and methodologically sound.

---

## 4. Independent Metric Recomputation & Verification

An independent recomputation script ([`research/audits/phase4_2_1/audit_recompute.py`](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/audits/phase4_2_1/audit_recompute.py)) was executed in background task `task-2738`. It loaded `linear_probe_cls_test_predictions.csv` and `linear_probe_mean_patch_test_predictions.csv`, extracted the true class indices and predicted class probabilities (`prob_ADI`, ..., `prob_TUM`), and independently calculated all overall and per-class metrics.

### Overall Metric Audit Comparison:

| Metric | CLS Reported | CLS Recomputed | CLS $\Delta$ | Patch Reported | Patch Recomputed | Patch $\Delta$ | Audit Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Test Accuracy** | $0.6435$ | $0.6435$ | $0.0000$ | $0.6508$ | $0.6508$ | $0.0000$ | **Exact Match** |
| **Balanced Accuracy** | $0.6435$ | $0.6435$ | $0.0000$ | $0.6508$ | $0.6508$ | $0.0000$ | **Exact Match** |
| **Macro Precision** | $0.6423$ | $0.6423$ | $0.0000$ | $0.6505$ | $0.6505$ | $0.0000$ | **Exact Match** |
| **Macro Recall** | $0.6435$ | $0.6435$ | $0.0000$ | $0.6508$ | $0.6508$ | $0.0000$ | **Exact Match** |
| **Macro-F1** | $0.6421$ | $0.6421$ | $0.0000$ | $0.6503$ | $0.6503$ | $0.0000$ | **Exact Match** |
| **Weighted Precision** | $0.6423$ | $0.6423$ | $0.0000$ | $0.6505$ | $0.6505$ | $0.0000$ | **Exact Match** |
| **Weighted Recall** | $0.6435$ | $0.6435$ | $0.0000$ | $0.6508$ | $0.6508$ | $0.0000$ | **Exact Match** |
| **Weighted-F1** | $0.6421$ | $0.6421$ | $0.0000$ | $0.6503$ | $0.6503$ | $0.0000$ | **Exact Match** |
| **Cohen's Kappa** ($\kappa$) | $0.5926$ | $0.5926$ | $0.0000$ | $0.6009$ | $0.6009$ | $0.0000$ | **Exact Match** |
| **Expected Calibration Error (ECE)** | $0.0220$ | $0.0217$ | $-0.0003$* | $0.0368$ | $0.0368$ | $0.0000$ | **Verified** |
| **Multiclass Brier Score** | $0.4819$ | $0.4819$ | $0.0000$ | $0.4730$ | $0.4730$ | $0.0000$ | **Exact Match** |
| **Macro OvR AUROC** | $0.9293$ | $0.9293$ | $0.0000$ | $0.9316$ | $0.9316$ | $0.0000$ | **Exact Match** |

*\*Note on CLS ECE: The reported ECE of $0.0220$ was computed in-memory using full 32-bit floating-point probabilities. The recomputed ECE of $0.0217$ was evaluated from CSV probabilities rounded to 4 decimal places. The absolute difference of $0.0003$ is an expected quantization artifact of CSV serialization and does not represent a metric discrepancy.*

### Class Label Order and Probability Normalization Verification:
- **Canonical Label Ordering:** Strictly verified as `['ADI', 'DEB', 'LYM', 'MUC', 'MUS', 'NOR', 'STR', 'TUM']` (indices $0$ through $7$).
- **Probability Sum Normalization:** The sum of predicted class probabilities across all 8 columns was verified for every test row. Maximum absolute deviation from $1.0$ across all $4,665$ rows is $\le 0.0003$ (attributable solely to 4-decimal CSV rounding).

---

## 5. Corrected Per-Class Benchmark Table

Below is the verified, publication-grade per-class benchmark table for both the CLS Token Probe ($C^*=20.0$) and the Mean Spatial Patch Probe ($C^*=200.0$) evaluated on the Phase 2 test split ($N=4,665$ images). This table incorporates all 8 canonical tissue classes along with both unweighted (**Macro**) and support-weighted (**Weighted**) summary rows:

| Class Code | Histological Tissue Description | Test Support | CLS Precision | CLS Recall (Sens) | CLS Specificity | CLS F1-Score | Patch Precision | Patch Recall (Sens) | Patch Specificity | Patch F1-Score |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ADI** | Adipose Tissue | 583 | 75.08% | 76.50% | 96.37% | **75.79%** | 76.13% | 78.22% | 96.50% | **77.16%** |
| **DEB** | Necrotic Debris | 583 | 58.72% | 56.60% | 94.32% | **57.64%** | 58.15% | 58.15% | 94.02% | **58.15%** |
| **LYM** | Lymphocyte Aggregates | 583 | 74.45% | 80.96% | 96.03% | **77.57%** | 77.36% | 80.27% | 96.64% | **78.79%** |
| **MUC** | Extracellular Mucin Lakes | 584 | 61.79% | 57.88% | 94.88% | **59.77%** | 65.79% | 59.93% | 95.54% | **62.72%** |
| **MUS** | Smooth Muscle Bundles | 583 | 72.20% | 66.38% | 96.35% | **69.17%** | 71.04% | 67.75% | 96.06% | **69.36%** |
| **NOR** | Normal Gastric Mucosa | 583 | 51.37% | 48.20% | 93.48% | **49.73%** | 52.35% | 51.63% | 93.29% | **51.99%** |
| **STR** | Cancer-Associated Stroma | 583 | 60.62% | 60.72% | 94.37% | **60.67%** | 58.56% | 59.86% | 93.95% | **59.20%** |
| **TUM** | Adenocarcinoma Epithelium | 583 | 59.61% | 67.58% | 93.46% | **63.34%** | 61.07% | 64.84% | 94.10% | **62.90%** |
| **Macro** | *Unweighted Arithmetic Mean* | 4665 | 64.23% | 64.35% | 94.91% | **64.21%** | 65.05% | 65.08% | 95.01% | **65.03%** |
| **Weighted** | *Support-Weighted Mean* | 4665 | 64.23% | 64.35% | 94.91% | **64.21%** | 65.05% | 65.08% | 95.01% | **65.03%** |

### Mathematical Note on DEB Metrics:
In the Mean Patch probe results, the Precision, Recall, and F1-Score for Necrotic Debris (`DEB`) are all identical at $58.15\%$. An audit of the underlying confusion matrix confirms that this is not a copy-paste error:
$$\text{True Positives } (\text{TP}) = 339, \quad \text{False Positives } (\text{FP}) = 244, \quad \text{False Negatives } (\text{FN}) = 244$$
$$\text{Precision} = \frac{339}{339 + 244} = \frac{339}{583} \approx 0.581475 \implies \mathbf{58.15\%}$$
$$\text{Recall} = \frac{339}{339 + 244} = \frac{339}{583} \approx 0.581475 \implies \mathbf{58.15\%}$$
Because $\text{FP} = \text{FN} = 244$, Precision and Recall are algebraically identical, which necessarily yields an identical F1-Score of $58.15\%$.

---

## 6. Scientific Terminology & Conceptual Corrections

To adhere strictly to biomedical machine learning reporting guidelines (such as TRIPOD-AI and STARD-AI), the following language and conceptual calibrations are mandated across all Phase 4.2 documentation:

### 1. Incommensurability with Historical Baseline (Numerical Proximity Only):
- **Prohibited Phrasing:** Stating that the DINOv2 linear probe "matches," "equals," or "outperforms" the historical `GastricCancerRNN` baseline.
- **Audited Factual Position:** 
  - The historical `GastricCancerRNN` baseline achieved an accuracy of $65.92\%$ ($65.66\%$ strictly unexposed) on its **historical test cohort of $N=3,110$ patches**.
  - The frozen DINOv2 linear probe achieved an accuracy of $65.08\%$ on the **Phase 2 cluster-isolated test cohort of $N=4,665$ patches**.
  - These two evaluations were conducted on **differing test sample populations with different duplicate quarantine constraints and different sample sizes**.
  - Direct statistical equivalence cannot be claimed between disparate test cohorts. The report must state only that the frozen foundation model linear probe exhibits **numerical proximity** ($65.08\%$ vs. $65.92\%$) across these separately constituted benchmarks.

### 2. Patch-Level Scope of Confidence Intervals:
- All reported confidence intervals (Wilson score intervals for accuracy and sensitivity; empirical bootstrap percentiles for Balanced Accuracy and Macro-F1) represent **patch-level / image-level statistical bounds**.
- Because the GCHTID dataset does not provide patient, slide, or block identifiers, it is impossible to perform clustered patient-level bootstrapping or hierarchical mixed-effects modeling.
- The documentation must explicitly state that all intervals reflect patch-sampling variance rather than patient-level diagnostic confidence.

### 3. Absolute Prohibition of Clinical Generalization Claims:
- Downstream linear probing on frozen embeddings is a **methodological feasibility assessment** of latent feature structure, not a clinical diagnostic validation.
- No claim of patient-level generalizability, prospective diagnostic efficacy, or clinical readiness is scientifically permissible.

### 4. Framing Histological Error Explanations as Scientific Hypotheses:
- Interpretations of model confusion modes (e.g., `TUM` $\rightarrow$ `NOR`, `TUM` $\rightarrow$ `STR`, `MUC` $\leftrightarrow$ `ADI`) are based on histological morphology observed in sampled patches.
- These explanations must be explicitly framed as **plausible biological hypotheses** rather than established mechanistic facts, acknowledging that self-supervised vision transformer attention may also respond to non-biological imaging artifacts, stain variations, or background textures.

---

## 7. Checkpoint & System Integrity Audit

The integrity of all existing core system assets was re-verified at the conclusion of this audit pass:

| Asset | Expected Value / State | Observed Value | Integrity Status |
|:---|:---|:---|:---:|
| `models/checkpoints/gastric_rnn_classifier.pt` | SHA-256: `65369ef6...d8b8` | SHA-256: `65369ef6...d8b8` | **Bitwise Preserved** |
| SQLite Database `app.db` | Exactly 4 historical records | Exactly 4 historical records | **Preserved** |
| DINOv2 Cache Manifest (`manifest.csv`) | 31,096 rows | 31,096 rows | **Preserved** |
| Frozen CLS Embeddings (`cls_embeddings.npy`) | Shape: $(31096, 384)$, float32 | Shape: $(31096, 384)$, float32 | **Preserved** |
| Frozen Patch Embeddings (`mean_patch_embeddings.npy`) | Shape: $(31096, 384)$, float32 | Shape: $(31096, 384)$, float32 | **Preserved** |
| Test Predictions CSV Files | Exactly 4,665 rows each | Exactly 4,665 rows each | **Preserved** |

---

## 8. Audit Certification

This audit confirms that Phase 4.2 downstream linear probe evaluations were conducted with rigorous split isolation, deterministic optimization, and verifiable metric fidelity. All required corrections to reporting tables and scientific phrasing have been incorporated into the primary research reports.

*Certified by GastricPath AI Research Engine. Work ceases per Phase 4.2.1 audit boundary.*

