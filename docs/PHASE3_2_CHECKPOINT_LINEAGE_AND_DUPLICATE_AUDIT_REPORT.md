# GastricPath AI — Phase 3.2 Forensic Audit Report
## Checkpoint Training-Lineage & Duplicate-Aware Benchmark Audit

**Document ID:** `GP-REPORT-PHASE3.2-LINEAGE-AUDIT-2026-V1`  
**Date:** November 2026  
**Status:** Complete — Forensic Audit Verified  
**Branch:** `gastricpath-ai-evolution`  
**Test Suite Pass Rate:** 100% (54/54 tests passed in 28.72s)  
**Audit Output Directory:** [`research/audits/phase3_2/`](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/audits/phase3_2/)  
**Benchmark Integrity Status:** **CONTAMINATED** (80.56% of Phase 3.1 test images appeared in checkpoint gradient training)  

---

## 1. Executive Summary

Phase 3.2 executed an exhaustive forensic investigation to determine whether the Phase 3.1 baseline benchmark results (71.04% test accuracy on 4,665 images) represent genuine out-of-sample generalization, and to quantify the statistical influence of duplicate images and cluster correlation on reported metrics.

### Key Forensic Discoveries
1. **100% Original Training Lineage Recovered:** The exact dataset manifest used to train the historical baseline checkpoint (`models/checkpoints/gastric_rnn_classifier.pt`) was recovered from [`model_train/runs/dataset_manifest.csv`](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/model_train/runs/dataset_manifest.csv). Checkpoint metadata confirmed that the weights were saved at **Epoch 15** (`val_accuracy = 0.648553`), exactly matching [`model_train/logs/training_summary.json`](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/model_train/logs/training_summary.json). All 31,096 historical image assignments were mapped with 100% path-level and bitwise certainty.
2. **Benchmark Integrity Status: CONTAMINATED:**  
   Because Phase 2 partitioned the 31,096 images independently into a new 70% / 15% / 15% split without referencing the historical training split, **3,758 of the 4,665 test images (80.56%) were part of the original model's gradient-based training set**, and **465 images (9.97%) were in its validation tuning set**. Only **442 images (9.47%)** were genuinely unseen by the historical checkpoint during training.
3. **Empirical Evidence of Memorization / Train-Test Contamination:**  
   - Accuracy on the 3,758 historical **TRAIN** images: **72.99%** (Macro-F1: 73.10%, AUROC: 0.9645).
   - Accuracy on the 465 historical **VAL** images: **60.22%** (Macro-F1: 60.35%, AUROC: 0.9294).
   - Accuracy on the 442 genuinely **UNSEEN TEST** images: **65.84%** (Macro-F1: 65.14%, AUROC: 0.9434).  
   The baseline model performs **7.15 percentage points higher** on historical training images (72.99%) than on genuinely unseen test images (65.84%). The Phase 3.1 overall accuracy of 71.04% was predominantly driven by training set memorization.
4. **Duplicate Cohort Analysis:**  
   Within the 4,665 test images, there are **4,496 exact-unique SHA-256 hashes** and **169 duplicate files** belonging to 136 duplicate groups. All 136 duplicate groups have 100% consistent ground truth labels and received 100% identical predictions from the model.
5. **Deduplicated vs. File-Weighted Performance:**  
   Evaluating the model on the 4,496 exact-unique images yields **70.93% accuracy** (vs. 71.04% file-weighted, $\Delta = -0.11\%$) and **71.01% macro-F1** (vs. 71.14%, $\Delta = -0.13\%$). Intra-test duplicates exerted negligible metric inflation ($\Delta \approx 0.1\%$).
6. **Cluster-Bootstrap Uncertainty:**  
   Resampling over 4,496 independent clusters (136 multi-member groups + 4,360 singletons) across $B=1000$ replicates yields an **Accuracy 95% Cluster CI of $[69.69\%, 72.45\%]$** and **Macro-F1 95% Cluster CI of $[69.73\%, 72.50\%]$**, properly accounting for intra-cluster correlation.

---

## 2. Original Checkpoint Lineage Forensic Reconstruction

### 1. Checkpoint State-Dict Inspection
- **File:** `models/checkpoints/gastric_rnn_classifier.pt`
- **File Size:** 3,664,577 bytes (3.49 MB)
- **SHA-256 Hash:** `65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8`
- **Payload Dictionary Keys:** `['model_state_dict', 'class_names', 'val_accuracy', 'settings']`
- **Recorded Validation Accuracy:** `0.6485530546623794` (64.86%)
- **Recorded Architectural Settings:** `image_size: 224, patch_size: 56, sequence_length: 16`
- **Classes:** `['ADI', 'DEB', 'LYM', 'MUC', 'MUS', 'NOR', 'STR', 'TUM']` (8 classes)

### 2. Historical Training Log Cross-Referencing
Cross-referencing `val_accuracy: 0.6485530546623794` against `model_train/logs/training_summary.json` revealed an exact bitwise match at **Epoch 15**:

```
Epoch 13: train_loss = 0.8070, train_acc = 0.6863, val_loss = 1.2405, val_acc = 0.576849
Epoch 14: train_loss = 0.7741, train_acc = 0.6976, val_loss = 1.2739, val_acc = 0.540514
Epoch 15: train_loss = 0.7349, train_acc = 0.7118, val_loss = 0.9327, val_acc = 0.648553 <=== SAVED CHECKPOINT
Epoch 16: train_loss = 0.7011, train_acc = 0.7264, val_loss = 2.0835, val_acc = 0.517042
```

In `training/train.py`, the training loop evaluates `val_accuracy` at the end of each epoch and executes:
```python
if val_accuracy > best_val_accuracy:
    best_val_accuracy = val_accuracy
    torch.save(checkpoint_payload, settings.model_checkpoint)
```
Epoch 15 was the global peak validation accuracy. At Epoch 16, severe overfitting occurred (`val_loss` spiked to 2.0835, `val_acc` dropped to 51.70%), so Epoch 15 was preserved as the final model checkpoint.

### 3. Historical Training Dataset Split Reconstruction
The training script `training/train.py` utilized `--manifest model_train/runs/dataset_manifest.csv`, generated by `training/prepare_manifest.py` using `train_test_split(..., test_size=0.2, random_state=42)` and a subsequent 50/50 split of the 20% holdout (`test_size=0.5, random_state=42`).

The historical manifest partitions the 31,096 images as:
- **Historical Train Split:** **24,876 images (80.0%)** — utilized for gradient-based backpropagation.
- **Historical Validation Split:** **3,110 images (10.0%)** — utilized for early stopping and checkpoint selection.
- **Historical Test Split:** **3,110 images (10.0%)** — held out as the original test cohort.

All 31,096 image assignments are stored with absolute filesystem paths and class codes. **Lineage recovery is 100.0% complete (0 unresolved images).**

---

## 3. Original Training vs. Phase 3.1 Test Overlap Audit

In Phase 2, `research/data/split_generator.py` independently partitioned the 31,096 images into a new 70% / 15% / 15% split (`manifest_train.csv`: 21,767; `manifest_val.csv`: 4,664; `manifest_test.csv`: 4,665) using dHash cluster indivisibility and a different random seed (`seed=42 + hash(class)`).

Because the Phase 2 split was generated independently from the full dataset without conditioning on the historical split, the 4,665 images in `manifest_test.csv` drew randomly from the entire dataset:

### Lineage Membership of Phase 3.1 Test Images

```
+-------------------------------------------------------------------------------------------------------+
|                                Phase 3.1 Test Cohort Lineage Breakdown                                |
+------------------------------+---------------+---------------+--------------------+-------------------+
| Historical Training Role     | Sample Count  | Percentage    | Lineage Nature     | Baseline Accuracy |
+------------------------------+---------------+---------------+--------------------+-------------------+
| Historical TRAIN (Gradients) | 3,758         | 80.56%        | Direct Leakage     | 72.99%            |
| Historical VAL (Selection)   | 465           | 9.97%         | Indirect Leakage   | 60.22%            |
| Historical TEST (Unseen)     | 442           | 9.47%         | Truly Independent  | 65.84%            |
| Unresolved Lineage           | 0             | 0.00%         | None               | —                 |
+------------------------------+---------------+---------------+--------------------+-------------------+
| Total Phase 3.1 Test Cohort  | 4,665         | 100.00%       | Mixed Cohort       | 71.04%            |
+------------------------------+---------------+---------------+--------------------+-------------------+
```

### Empirical Overlap Metrics Table

| Metric | Historical TRAIN Subset | Historical VAL Subset | Historical TEST Subset (Unseen) | Full Test Set (Phase 3.1) |
|:---|:---:|:---:|:---:|:---:|
| **Sample Count ($N$)** | 3,758 (80.56%) | 465 (9.97%) | 442 (9.47%) | 4,665 (100.0%) |
| **Top-1 Accuracy** | **72.99%** | **60.22%** | **65.84%** | **71.04%** |
| **Balanced Accuracy** | **73.06%** | 59.75% | **65.13%** | 71.04% |
| **Macro F1-Score** | **73.10%** | 60.35% | **65.14%** | 71.14% |
| **Macro AUROC (OvR)** | **0.9645** | 0.9294 | **0.9434** | 0.9593 |
| **Expected Calib Error (ECE)**| 0.0336 | 0.0748 | 0.0443 | 0.0167 |
| **Brier Score** | **0.3689** | 0.5009 | 0.4319 | 0.3880 |

### Scientific Finding
The baseline model achieves **72.99% accuracy on images it was trained on**, but only **65.84% accuracy on images that were genuinely unseen** during its original training. The overall accuracy of 71.04% reported in Phase 3.1 was an artifact of evaluating a trained checkpoint on a test set composed of 80.56% training images.

The complete image-by-image overlap audit is exported to:  
[`research/audits/phase3_2/checkpoint_train_test_overlap.csv`](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/audits/phase3_2/checkpoint_train_test_overlap.csv).

---

## 4. Duplicate-Aware Test Cohort Analysis

An investigation of duplicate image files and perceptual clusters inside `research/splits/manifest_test.csv` was conducted using `research/reports/exact_duplicates_report.json` and `research/reports/near_duplicates_report.json`:

```
+-----------------------------------------------------------------------------------------------+
|                               Test Set Duplicate Characteristics                              |
+-----------------------------------------------------------+-----------------------------------+
| Metric                                                    | Value                             |
+-----------------------------------------------------------+-----------------------------------+
| Total Image Files in Test Set                             | 4,665                             |
| Total Unique SHA-256 Checksums in Test Set                | 4,496                             |
| Exact Duplicate Image Files in Test Set                   | 169 (3.62% of test set)           |
| Exact Duplicate Groups Represented in Test Set            | 136 groups                        |
| Multi-Member Duplicate Groups in Test Set                 | 136 groups (all co-located)       |
| Total Images in Multi-Member Duplicate Groups             | 305 images                        |
| Contradictory Ground Truth Labels in Test Groups          | 0 (100% label consistency)        |
| Inconsistent Predictions by Baseline Model in Test Groups | 0 (100% prediction consistency)   |
| Total Independent Perceptual / Duplicate Clusters in Test | 4,496 clusters                    |
+-----------------------------------------------------------+-----------------------------------+
```

### Key Duplicate Findings
1. **Zero Split Leakage for Duplicates:** All 136 duplicate groups in the test set have all their duplicate copies co-located within the test set. Zero duplicate copies leak across train, val, or test in the Phase 2 split.
2. **Perfect Prediction Consistency:** For every one of the 136 duplicate groups (comprising 305 images), the model produced **100% identical class predictions** for all identical images in the group.
3. **Zero Label Contradictions:** No duplicate group in the test set contains conflicting histological labels.

The group summary is exported to:  
[`research/audits/phase3_2/test_duplicate_group_summary.csv`](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/audits/phase3_2/test_duplicate_group_summary.csv).

---

## 5. File-Weighted vs. Exact-Unique-Image Metrics

To determine whether the 169 duplicate files in the test set artificially inflated benchmark metrics, the evaluation was deduplicated by retaining exactly one exemplar per unique SHA-256 hash ($N = 4,496$).

### Deduplication Rule
- Images grouped by exact SHA-256 hash.
- Ground truth labels verified to be identical across all copies (0 conflicts).
- Deterministically select the first image path in alphabetical order as the group exemplar.
- Exported to [`research/audits/phase3_2/deduplicated_predictions.csv`](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/audits/phase3_2/deduplicated_predictions.csv).

### Performance Comparison Table

| Metric | File-Weighted (4,665 Files) | Exact-Unique (4,496 Unique Images) | Absolute Delta ($\Delta$) | Impact Assessment |
|:---|:---:|:---:|:---:|:---|
| **Top-1 Accuracy** | **71.04%** | **70.93%** | $-0.11\%$ | Negligible |
| **Balanced Accuracy** | **71.04%** | **70.98%** | $-0.06\%$ | Negligible |
| **Macro F1-Score** | **71.14%** | **71.01%** | $-0.13\%$ | Negligible |
| **Weighted F1-Score** | **71.14%** | **71.03%** | $-0.11\%$ | Negligible |
| **Macro Precision** | **72.28%** | **72.13%** | $-0.15\%$ | Negligible |
| **Macro Recall (Sensitivity)** | **71.04%** | **70.98%** | $-0.06\%$ | Negligible |
| **Cohen's Kappa ($\kappa$)** | **0.6690** | **0.6678** | $-0.0012$ | Substantial agreement preserved |
| **Macro AUROC (OvR)** | **0.9593** | **0.9587** | $-0.0006$ | Outstanding discrimination preserved |
| **Weighted AUROC (OvR)** | **0.9593** | **0.9587** | $-0.0006$ | Negligible |
| **Expected Calibration Error**| **0.0167** | **0.0156** | $-0.0011$ | Slightly better calibrated |
| **Multiclass Brier Score** | **0.3880** | **0.3902** | $+0.0022$ | Negligible |

### Scientific Conclusion on Duplicates
The presence of 169 duplicate files in the test split altered accuracy by merely **0.11 percentage points** (71.04% vs. 70.93%). Therefore, intra-split duplicate inflation is **not** a significant source of error in this benchmark. The primary confounder is training-set overlap, not test-set duplicate duplication.

---

## 6. Cluster-Bootstrap Uncertainty Analysis

In standard i.i.d. bootstrap resampling, each image file is sampled independently. However, if duplicate or near-duplicate images exist, sampling them independently violates the exchangeability assumption and can underestimate variance.

To resolve this, a **Cluster Bootstrap** was implemented:
- **Resampling Unit:** Indivisible duplicate/dHash cluster ($C = 4,496$ clusters: 136 multi-member groups + 4,360 singletons).
- **Sampling Scheme:** In each of $B = 1000$ iterations, $4,496$ clusters are sampled with replacement using deterministic seed `42`.
- **Indivisibility Rule:** Whenever a cluster is selected, **all** member images of that cluster are included in the bootstrap replicate.

### Uncertainty Comparison Table

```
+-------------------------------------------------------------------------------------------------------+
|                                    Confidence Interval Comparison                                     |
+----------------------+--------------------+--------------------+--------------------+-----------------+
| Metric               | Point Estimate     | Wilson Binomial 95%| i.i.d. Bootstrap   | Cluster Bootstrap|
+----------------------+--------------------+--------------------+--------------------+-----------------+
| Test Accuracy        | 71.04%             | [69.72%, 72.32%]   | [69.72%, 72.32%]   | [69.69%, 72.45%]|
| Balanced Accuracy    | 71.04%             | —                  | [69.80%, 72.25%]   | [69.63%, 72.39%]|
| Macro F1-Score       | 71.14%             | —                  | [69.84%, 72.35%]   | [69.73%, 72.50%]|
+----------------------+--------------------+--------------------+--------------------+-----------------+
```

### Observation
The cluster bootstrap yields slightly wider intervals ($\approx \pm 0.13\%$ wider), correctly incorporating the intra-cluster correlation induced by co-located duplicates.

Full bootstrap results are exported to:  
[`research/audits/phase3_2/cluster_bootstrap_results.json`](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/audits/phase3_2/cluster_bootstrap_results.json).

---

## 7. Benchmark Integrity Classification

Under the established taxonomy, the Phase 3.1 baseline benchmark is officially classified as:

# **CONTAMINATED**

### Concrete Justification
1. **Direct Training Overlap:** 3,758 of the 4,665 test images (80.56%) were used during gradient-based training of the baseline model checkpoint.
2. **Performance Disparity:** The checkpoint achieves **72.99% accuracy on historical training images**, but drops to **65.84% accuracy on genuinely unseen test images**.
3. **Misleading Baseline Benchmark:** Reporting 71.04% as the "test accuracy" of the baseline model is scientifically invalid because the test set is primarily evaluating training-set memorization rather than generalization.

---

## 8. Scientific Limitations & Disclosures

1. **True Generalization of Baseline Checkpoint:** The genuine out-of-sample accuracy of `models/checkpoints/gastric_rnn_classifier.pt` is **65.84%** (measured on the 442 unseen test images), not 71.04%.
2. **Non-Patient-Level Uncertainty:** All confidence intervals (Wilson, i.i.d. bootstrap, and cluster bootstrap) measure sampling variance over image patches and perceptual clusters within the GCHTID dataset. True patient identifiers are not present in GCHTID filenames, so patient-level generalization cannot be claimed.
3. **No Anatomical Area Inference:** The classification probability (`TUM: 94%`) indicates patch-level classification confidence, not anatomical tumor area percentage.
4. **Investigational Research Platform:** GastricPath AI is an experimental computational pathology platform and is not cleared for clinical diagnostic use.

---

## 9. Recommended Corrected Benchmark Protocol

To establish clean, scientifically unassailable benchmarks for Phase 4, the following two protocols are recommended:

### Protocol A: Uncontaminated Baseline Evaluation (Immediate, No Training)
- Evaluate the existing baseline checkpoint strictly on the **3,110 genuinely unseen images from its own historical test split** (`model_train/runs/dataset_manifest.csv` where `split == 'test'`).
- This measures the true generalization of the existing checkpoint without data leakage.

### Protocol B: Clean Retraining on Phase 2 Splits (Recommended for Fair Architecture Comparison)
- Train Model A (`GastricCancerRNN`), Model B (`ResNet50`), Model C (`EfficientNet-B0`), and Phase 4 Foundation Model classifiers from scratch strictly on `research/splits/manifest_train.csv` (21,767 images).
- Use `research/splits/manifest_val.csv` (4,664 images) strictly for validation monitoring and checkpoint selection.
- Evaluate all models strictly on `research/splits/manifest_test.csv` (4,665 images).
- Because all models will be trained on the exact same clean training split and evaluated on the exact same unseen test split, this provides a **100% leak-proof, scientifically rigorous, fair comparison**.

> [!CAUTION]
> **Explicit Authorization Mandate:**  
> Retraining models requires significant compute time and user authorization. No retraining will be initiated without explicit user approval.

---

## 10. Machine-Readable Audit Inventory

All audit artifacts have been created and verified under [`research/audits/phase3_2/`](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/audits/phase3_2/):

| Artifact Filename | Size | Purpose / Description |
|:---|:---:|:---|
| `checkpoint_train_test_overlap.csv` | 623 KB | Complete 4,665-image mapping of test samples to historical train/val/test splits with prediction correctness |
| `test_duplicate_group_summary.csv` | 11.9 KB | Summary of 136 duplicate groups in test with label and prediction consistency verification |
| `deduplicated_predictions.csv` | 779 KB | 4,496 exact-unique SHA-256 test predictions with multi-class probability vectors |
| `duplicate_aware_metrics.json` | 13.5 KB | Comprehensive JSON comparing file-weighted, exact-unique, and lineage-subset metrics |
| `cluster_bootstrap_results.json` | 1.4 KB | 1000-replicate cluster-bootstrap confidence intervals for accuracy, balanced accuracy, and Macro-F1 |

---

## 11. Reproducibility Instructions

To reproduce the Phase 3.2 audit programmatically:

```powershell
# 1. Run the Phase 3.2 Audit Engine
python -m research.audits.phase3_2_audit

# 2. Run the audit test suite
python -m pytest tests/test_audit_phase3_2.py -v

# 3. Run the complete platform test suite
python -m pytest tests -v
```

---
*Report certified by GastricPath AI Forensic Audit Engine. Awaiting explicit user authorization before initiating any retraining or proceeding to Phase 4.*
