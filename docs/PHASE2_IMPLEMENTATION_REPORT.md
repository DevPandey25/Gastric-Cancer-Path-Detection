# GastricPath AI — Phase 2 Implementation Report
### Dataset Integrity Audit, Duplicate & Leakage Detection, and Reproducible Dataset Splits

**Document ID:** `GP-REPORT-PHASE2-2026-V1`  
**Execution Phase:** Phase 2 (Dataset Integrity, Leakage Detection, and Reproducible Dataset Splits)  
**Git Branch:** `gastricpath-ai-evolution`  
**Date:** November 2026  
**Status:** Completed & Empirically Verified (34/34 Automated Tests Passing)  

---

## 1. Executive Summary

Phase 2 establishes the data foundation and scientific integrity controls for **GastricPath AI**, conducting a comprehensive, non-destructive audit of the complete Gastric Cancer Histopathology Tissue Image Dataset (GCHTID) comprising **31,096 high-resolution images** across eight tissue classes.

In accordance with the **GastricPath AI Research and Implementation Blueprint** and Phase 2 authorization requirements, this phase delivered:
1. **Dataset Integrity Scanner** (`research/data/integrity_scanner.py`): Streamed and verified all 31,096 images for readability, corruption, dimension uniformity ($224 \times 224$), channel depth, file size distribution, and RGB channel intensity profiles.
2. **Exact Duplicate Detection Engine** (`research/data/duplicate_detector.py`): Computed streaming cryptographic SHA-256 checksums across all images, auditing intra-class duplicates and confirming **zero cross-class label contradictions**.
3. **Perceptual Leakage & Near-Duplicate Detector** (`research/data/leakage_detector.py`): Implemented difference hashing (dHash) with multi-index inverted chunk lookup and Disjoint Set Union (DSU) graph clustering to identify morphologically identical patch clusters.
4. **Canonical Dataset Manifest** (`research/data/manifest_generator.py`): Produced an immutable, deterministically sorted master manifest (`research/manifests/manifest_all.csv`) recording relative paths, tissue classes, class indices, dimensions, file sizes, SHA-256 hashes, and dHash values.
5. **Leak-Proof Group-Stratified Dataset Splits** (`research/data/split_generator.py`): Partitioned the dataset into **70% Training (21,767), 15% Validation (4,664), and 15% Testing (4,665)** with strict group indivisibility, mathematically preventing duplicate or near-duplicate patches from crossing between training and evaluation splits.
6. **Scientific Disclosures**: Fully disclosed that all splits are **image-level**, not patient-level, due to the lack of patient/slide identifiers in raw GCHTID filenames.
7. **Automated Testing**: Created 13 new unit and integration tests across 5 test suites. All **34 tests in the repository pass** in 12.44 seconds.

---

## 2. Empirical Dataset Audit Findings (31,096 Images)

The unified scan was executed across the entire raw dataset directory (`data/raw/HMU-GC-HE-30K/all_image/`):

### 2.1 File Integrity, Dimensions, and Formatting
- **Total Images Scanned:** 31,096
- **Corrupt or Unreadable Files:** 0 (100% readability rate)
- **Image Format Distribution:** 100% `PNG` (31,096 / 31,096)
- **Spatial Resolution:** 100% exactly $224 \times 224$ pixels (0 dimension anomalies detected)
- **File Size Distribution:**
  - Minimum File Size: 17,219 bytes (~16.8 KB)
  - Maximum File Size: 154,611 bytes (~151.0 KB)
  - Mean File Size: 96,739 bytes (~94.5 KB)
  - Median File Size: 99,856 bytes (~97.5 KB)
  - Total Dataset Storage Volume: 2,868.74 MB (~2.80 GB)

### 2.2 Critical Empirical Discovery: RGBA Channel Encoding
- **Finding:** Every single PNG image in the GCHTID dataset (31,096 / 31,096) is stored on disk with **mode RGBA (4 channels: Red, Green, Blue, Alpha)**.
- **Verification:** Inspection of the 4th channel revealed that the alpha channel is **identically 255 (fully opaque) across all pixels and all classes**.
- **Significance:** Standard pathology deep learning models require 3-channel RGB tensors. Naive loading without alpha conversion causes runtime shape mismatches ($[4, 224, 224]$ vs $[3, 224, 224]$). The pipeline now formally validates that every image cleanly and losslessly converts to 3-channel RGB (`img.convert("RGB")`).

### 2.3 Class Balance
The GCHTID dataset is verified to be **perfectly balanced**, with exactly 3,887 images per class:

| Class Code | Class Name | Image Count | Percentage |
|:---|:---|:---:|:---:|
| `ADI` | Adipose Tissue | 3,887 | 12.50% |
| `DEB` | Debris & Mucus | 3,887 | 12.50% |
| `LYM` | Normal Lymphoid Tissue | 3,887 | 12.50% |
| `MUC` | Mucus | 3,887 | 12.50% |
| `MUS` | Muscle | 3,887 | 12.50% |
| `NOR` | Normal Gastric Mucosa | 3,887 | 12.50% |
| `STR` | Cancer-Associated Stroma | 3,887 | 12.50% |
| `TUM` | Gastric Adenocarcinoma Epithelium | 3,887 | 12.50% |
| **Total** | **8 Histopathology Classes** | **31,096** | **100.00%** |

### 2.4 H&E Stain Intensity Statistics
RGB intensity statistics were calculated across $>310,000,000$ sampled pixels:

| Channel | Mean Intensity (0–255) | Std Deviation | Min Intensity | Max Intensity |
|:---|:---:|:---:|:---:|:---:|
| **Red (R)** | 193.55 | 45.18 | 0.0 | 255.0 |
| **Green (G)** | 154.15 | 52.09 | 0.0 | 255.0 |
| **Blue (B)** | 179.04 | 44.76 | 0.0 | 255.0 |

*Interpretation:* The observed channel ordering ($\bar{R} > \bar{B} > \bar{G}$) quantitatively captures the standard histochemical signature of Hematoxylin & Eosin (H&E): eosinophilic cytoplasm/stroma (pink-red) combined with basophilic cell nuclei (blue-purple).

---

## 3. Duplicate and Leakage Audit Findings

### 3.1 Exact Duplicate Audit (SHA-256)
- **Total Files Audited:** 31,096
- **Unique SHA-256 Hashes:** 29,910
- **Total Duplicate Files:** 2,118 images (files that share a bitwise-identical counterpart)
- **Total Duplicate Groups:** 932 groups
- **Cross-Class Duplicate Groups:** **0 (Zero)**
- **Intra-Class Duplicate Groups:** 932 groups

> [!NOTE]
> **Zero Label Contradictions:** There are zero instances of an identical image appearing in multiple different class folders. All bitwise duplicates reside strictly within their own respective class, confirming label integrity at the patch level.

### 3.2 Perceptual Near-Duplicate Audit (dHash, Hamming Distance $\le 2$)
Using 64-bit difference hashing (dHash) and multi-index inverted chunk retrieval:
- **Total Candidate Edges Verified:** 1,751
- **Near-Duplicate Clusters (Size $\ge 2$):** 932 clusters
- **Cross-Class Clusters:** **0 (Zero)**
- **Intra-Class Clusters:** 932 clusters

> [!IMPORTANT]
> **Scientific Integrity Limitation:** Perceptual similarity reflects textural and morphological alignment (e.g. uniform background stroma or empty lipid vacuoles in adipose tissue), but **does NOT prove** that patches originate from the same patient or slide. Histologically uniform tissues from entirely distinct patients naturally generate low Hamming distances.

---

## 4. Canonical Manifest & Reproducible Dataset Splits

### 4.1 Master Canonical Manifest
- **Path:** `research/manifests/manifest_all.csv`
- **Total Entries:** 31,096
- **SHA-256 Checksum:** `8d9c38a9dde58f53728f83475644cca3ca817ed396d2d8e1d5bf078f280c53b4`
- **Properties:** Invariant to operating system file ordering; deterministically sorted by POSIX relative path.

### 4.2 Group-Stratified Split Allocation (70% / 15% / 15%)
To prevent data leakage, each exact duplicate set and near-duplicate cluster was treated as an **indivisible atomic group**. All images in a group were assigned exclusively to either Train, Validation, or Test:

| Split Partition | Manifest File | Sample Count | Actual % | Target % | SHA-256 Checksum |
|:---|:---|:---:|:---:|:---:|:---|
| **Training** | `manifest_train.csv` | 21,767 | 70.00% | 70.0% | `f120813da8561039e1bbaef53b45d2ca18306e93a153d2da575ad768cb49b123` |
| **Validation** | `manifest_val.csv` | 4,664 | 15.00% | 15.0% | `5688205ee837dabb772fb31e899a774d8ae7e98a8c242f47b898d4aa9f6301a7` |
| **Testing** | `manifest_test.csv` | 4,665 | 15.00% | 15.0% | `3d22de382f70e7c3878a3c60bf39b993254dc59b1282efceb4102fdcef67d157` |
| **Total** | — | **31,096** | **100.00%** | **100.0%** | — |

### 4.3 Class Stratification Consistency Across Splits

| Tissue Class | Total | Training (70%) | Validation (15%) | Testing (15%) | Class Balance Check |
|:---|:---:|:---:|:---:|:---:|:---:|
| `ADI` | 3,887 | 2,721 (70.00%) | 583 (15.00%) | 583 (15.00%) | Exact |
| `DEB` | 3,887 | 2,721 (70.00%) | 583 (15.00%) | 583 (15.00%) | Exact |
| `LYM` | 3,887 | 2,721 (70.00%) | 583 (15.00%) | 583 (15.00%) | Exact |
| `MUC` | 3,887 | 2,720 (69.98%) | 583 (15.00%) | 584 (15.02%) | Group Indivisibility |
| `MUS` | 3,887 | 2,721 (70.00%) | 583 (15.00%) | 583 (15.00%) | Exact |
| `NOR` | 3,887 | 2,721 (70.00%) | 583 (15.00%) | 583 (15.00%) | Exact |
| `STR` | 3,887 | 2,721 (70.00%) | 583 (15.00%) | 583 (15.00%) | Exact |
| `TUM` | 3,887 | 2,721 (70.00%) | 583 (15.00%) | 583 (15.00%) | Exact |

### 4.4 Leakage Prevention Invariants Verified
1. **Path Mutual Exclusivity:**
   - $\text{Train} \cap \text{Val} = \emptyset$ (0 overlapping paths)
   - $\text{Train} \cap \text{Test} = \emptyset$ (0 overlapping paths)
   - $\text{Val} \cap \text{Test} = \emptyset$ (0 overlapping paths)
2. **Duplicate Isolation:**
   - $\text{SHA-256}(\text{Train}) \cap \text{SHA-256}(\text{Val}) = \emptyset$
   - $\text{SHA-256}(\text{Train}) \cap \text{SHA-256}(\text{Test}) = \emptyset$
   - $\text{SHA-256}(\text{Val}) \cap \text{SHA-256}(\text{Test}) = \emptyset$
3. **Partition Completeness:**
   - $|\text{Train}| + |\text{Val}| + |\text{Test}| = 31,096 = |\text{Dataset}|$

---

## 5. Automated Test Suite Execution

All unit, regression, and dataset audit tests were executed via `python -m pytest tests -v`:

```
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\KIIT0001\OneDrive\Desktop\gastric-cancer-histopathology-dashboard-main
configfile: pytest.ini
plugins: anyio-4.12.0
collected 34 items

tests/test_api_regression.py::test_dashboard_page PASSED                 [  2%]
tests/test_api_regression.py::test_api_health PASSED                     [  5%]
tests/test_api_regression.py::test_api_dashboard_payload PASSED          [  8%]
tests/test_api_regression.py::test_api_predictions_list PASSED           [ 11%]
tests/test_api_regression.py::test_api_predict_workflow PASSED           [ 14%]
tests/test_api_regression.py::test_api_predict_missing_filename PASSED   [ 17%]
tests/test_config.py::test_default_config_loading PASSED                 [ 20%]
tests/test_config.py::test_config_overrides PASSED                       [ 23%]
tests/test_config.py::test_config_path_resolution PASSED                 [ 26%]
tests/test_database.py::test_database_crud_operations PASSED             [ 29%]
tests/test_database.py::test_database_multiple_records_stats PASSED      [ 32%]
tests/test_dataset_integrity.py::test_integrity_scanner_valid_dataset PASSED [ 35%]
tests/test_dataset_integrity.py::test_integrity_scanner_corrupt_files PASSED [ 38%]
tests/test_dataset_integrity.py::test_integrity_scanner_dimension_and_channel_anomalies PASSED [ 41%]
tests/test_duplicate_detection.py::test_exact_duplicate_intra_class PASSED [ 44%]
tests/test_duplicate_detection.py::test_exact_duplicate_cross_class_conflict PASSED [ 47%]
tests/test_duplicate_detection.py::test_duplicate_report_export PASSED   [ 50%]
tests/test_leakage_detection.py::test_dhash_computation_and_distance PASSED [ 52%]
tests/test_leakage_detection.py::test_near_duplicate_clustering PASSED   [ 55%]
tests/test_leakage_detection.py::test_near_duplicate_cross_class_leakage PASSED [ 58%]
tests/test_ml_pipeline.py::test_image_preprocessing PASSED               [ 61%]
tests/test_ml_pipeline.py::test_softmax_utility PASSED                   [ 64%]
tests/test_ml_pipeline.py::test_gastric_rnn_forward_pass PASSED          [ 67%]
tests/test_registry.py::test_sha256_computation PASSED                   [ 70%]
tests/test_registry.py::test_model_registry_crud PASSED                  [ 73%]
tests/test_registry.py::test_register_actual_baseline_checkpoint PASSED  [ 76%]
tests/test_reproducibility.py::test_seed_determinism PASSED              [ 79%]
tests/test_reproducibility.py::test_environment_provenance PASSED        [ 82%]
tests/test_reproducibility.py::test_nondeterministic_disclosures PASSED  [ 85%]
tests/test_split_generator.py::test_manifest_generator PASSED            [ 88%]
tests/test_split_generator.py::test_split_generator_reproducibility_and_exclusivity PASSED [ 91%]
tests/test_split_generator.py::test_split_leakage_duplicate_grouping PASSED [ 94%]
tests/test_split_manifests_integrity.py::test_generated_manifests_integrity_and_exclusivity PASSED [ 97%]
tests/test_tracker.py::test_experiment_tracker_workflow PASSED           [100%]

======================= 34 passed, 4 warnings in 12.44s =======================
```

- **Pass Rate:** 100% (34 / 34 passed, 0 failed, 0 skipped).
- **Execution Speed:** 12.44 seconds total.

---

## 6. Generated Deliverables and File Locations

| Category | Relative File Path | Description |
|:---|:---|:---|
| **Code** | `research/data/integrity_scanner.py` | Image readability, dimension, channel, and RGB stats scanner |
| **Code** | `research/data/duplicate_detector.py` | SHA-256 exact duplicate detection & cross-class conflict auditor |
| **Code** | `research/data/leakage_detector.py` | Perceptual dHash candidate indexing & clustering engine |
| **Code** | `research/data/manifest_generator.py` | Canonical reproducible manifest writer and loader |
| **Code** | `research/data/split_generator.py` | Group-stratified leak-proof train/val/test split generator |
| **Code** | `research/data/run_phase2_audit.py` | Master driver script executing the full audit pipeline |
| **Report** | `research/reports/dataset_integrity_report.json` | Detailed JSON integrity audit report |
| **Report** | `research/reports/dataset_integrity_records.csv` | Full CSV record table for all 31,096 images |
| **Report** | `research/reports/exact_duplicates_report.json` | Exact duplicate group records and statistics |
| **Report** | `research/reports/exact_duplicates_report.csv` | CSV summary of duplicate hashes and file paths |
| **Report** | `research/reports/near_duplicates_report.json` | Perceptual cluster records and distance statistics |
| **Report** | `research/reports/near_duplicates_report.csv` | CSV summary of near-duplicate clusters |
| **Manifest** | `research/manifests/manifest_all.csv` | Canonical 31,096-row dataset manifest with SHA-256 and dHash |
| **Split** | `research/splits/manifest_train.csv` | 21,767-row training split manifest |
| **Split** | `research/splits/manifest_val.csv` | 4,664-row validation split manifest |
| **Split** | `research/splits/manifest_test.csv` | 4,665-row testing split manifest |
| **Split** | `research/splits/split_summary.json` | Split metadata, class distributions, file hashes, and disclosures |
| **Tests** | `tests/test_dataset_integrity.py` | Automated tests for image corruption and dimension anomalies |
| **Tests** | `tests/test_duplicate_detection.py` | Automated tests for exact duplicates and label conflicts |
| **Tests** | `tests/test_leakage_detection.py` | Automated tests for perceptual hashing and clustering |
| **Tests** | `tests/test_split_generator.py` | Automated tests for split reproducibility and group indivisibility |
| **Tests** | `tests/test_split_manifests_integrity.py` | Integration test verifying generated split files and mutual exclusivity |

---

## 7. Preservation of Existing Application

- **Live Service:** FastAPI dashboard server continues running at `http://127.0.0.1:8000`.
- **Database (`app.db`):** 4 historical user predictions remain untouched.
- **Trained Model Checkpoint:** `models/checkpoints/gastric_rnn_classifier.pt` verified with exact baseline SHA-256 `65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8`.
- **Raw Dataset:** Zero files were modified, moved, renamed, or deleted. All source images remain identical to their original states.

---

## 8. Recommendations for Phase 3: Classification Research Engine

With clean, verified, and leak-proof dataset manifests established, the platform is prepared for **Phase 3 (Classification Research Engine)**:
1. **Multi-Backbone Pathology Classification Suite:**
   - Maintain existing `GastricCancerRNN` (CNN-BiLSTM-attention) as verified baseline.
   - Implement standard 2D pathology classification backbones: ResNet-50 and lightweight modern architectures (ConvNeXt-Tiny / EfficientNet-V2) suited for CPU inference.
2. **Comprehensive Clinical Pathology Evaluation Metrics:**
   - Expand beyond simple accuracy to include Macro-F1, Weighted-F1, Balanced Accuracy, Per-Class Sensitivity (Recall), Per-Class Specificity, AUROC (One-vs-Rest), and Cohen's Kappa.
3. **Formal Decoupling of Cancer Probability from Tumor Area:**
   - Update API and dashboard reporting logic to clearly separate model confidence scores from physical anatomical tumor area fractions.
