# GastricPath AI — Phase 4.3.1 Artifact Provenance Audit
### Lineage, Integrity & Cohort Specification Audit

**Document ID:** `GP-AUDIT-PHASE4-3-1-PROVENANCE-2026-V1`  
**Execution Phase:** Phase 4.3.1 (Metric Reconciliation, Nonlinear Probe Audit & Statistical Certification)  
**Target Repository:** `gastric-cancer-histopathology-dashboard`  
**Git Branch:** `gastricpath-ai-evolution`  
**Base Commit:** `8dbdce6441c779f3eeddc75687c4ed829ca4233b` (`phase4.3: frozen DINOv2 nonlinear probing`)  
**Status:** Certified & Verified  

---

## 1. Audit Mandate & Scope

The mandate of this provenance audit is to establish an exhaustive, forensic record of all model artifacts, checkpoints, feature caches, test splits, prediction files, and probability outputs across the GastricPath AI computational pathology research platform.

To eliminate incommensurable comparisons, this document explicitly distinguishes between disparate evaluation cohorts:
1. **Historical CNN-RNN Baseline on Historical Test Split ($N = 3,110$)**
2. **Historical CNN-RNN Baseline on Phase 2 Test Split ($N = 4,665$)** (and its unexposed subset $N=442$)
3. **DINOv2 ViT-S/14 CLS Linear Probe ($N = 4,665$)**
4. **DINOv2 ViT-S/14 Mean Spatial Patch Linear Probe ($N = 4,665$)**
5. **DINOv2 ViT-S/14 CLS Nonlinear MLP Probe ($N = 4,665$)**
6. **DINOv2 ViT-S/14 Mean Spatial Patch Nonlinear MLP Probe ($N = 4,665$)**

These cohorts differ in sample count, duplicate quarantine boundaries, and patient distributions, and must never be conflated as equivalent evaluation cohorts.

---

## 2. Complete Model & Artifact Provenance Matrix

The following table provides the exact cryptographic and lineage provenance for every evaluated model:

| Evaluation Paradigm | Model Checkpoint / Artifact | Checkpoint SHA-256 | Feature Representation | Dataset Split & Manifest | Sample Count ($N$) | Prediction / Probability Artifact | Validation Selection Criterion | Test Invocation & Task ID | Git Commit Lineage |
|:---|:---|:---|:---|:---|:---:|:---|:---|:---|:---|
| **1. Historical CNN-RNN (Historical Test)** | `models/checkpoints/gastric_rnn_classifier.pt` | `65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8` | Raw $224 \times 224$ RGBA Images | Historical Test Split (`research/audits/phase3_3/historical_test_manifest.csv`) | 3,110 patches | `research/audits/phase3_3/historical_test_predictions.csv` (`917398f4...`) | Historical Epoch 15 (Repository Baseline) | Phase 3.3 Evaluation (`task-1927`) | `3636f2f` (Phase 3.3) / `53a5788` (Phase 3.3.1) |
| **2. Historical CNN-RNN (Strictly Unexposed)** | `models/checkpoints/gastric_rnn_classifier.pt` | `65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8` | Raw $224 \times 224$ RGBA Images | Duplicate-Filtered Historical Test (`historical_split_duplicate_overlap.csv`) | 2,918 patches | `research/audits/phase3_3/historical_test_unique_predictions.csv` (`32e1d685...`) | Duplicate Quarantine Filter | Phase 3.3 Audit | `3636f2f` (Phase 3.3) / `53a5788` (Phase 3.3.1) |
| **3. Historical CNN-RNN (Phase 2 Test Split)** | `models/checkpoints/gastric_rnn_classifier.pt` | `65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8` | Raw $224 \times 224$ RGBA Images | Phase 2 Test Split (`research/splits/manifest_test.csv`) | 4,665 patches *(80.56% Train Contaminated)* | `research/experiments/exp_20261105_221027_2463a2de/predictions_test.csv` (`ddc97c08...`) | Phase 3.1 Initial Benchmark | Phase 3.1 Task | `0ec2872` (Phase 3.1) / `fabdae2` (Phase 3.2) |
| **4. Historical CNN-RNN (Genuinely Unseen inside Phase 2)** | `models/checkpoints/gastric_rnn_classifier.pt` | `65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8` | Raw $224 \times 224$ RGBA Images | Phase 2 Test Subset with Historical Test Lineage | 442 patches | `research/audits/phase3_2/checkpoint_train_test_overlap.csv` (Filtered) | Lineage Membership Filter | Phase 3.2 Audit | `fabdae2` (Phase 3.2) |
| **5. DINOv2 CLS Linear Probe** | `research/foundation_models/checkpoints/linear_probe_cls.joblib` | `42a0c46277cc6ff527f86ac5558e322040d0236545040ca946a9842edb9b7d1c` | Frozen CLS Token ($d=384$) | Phase 2 Test Split (`research/splits/manifest_test.csv`) | 4,665 patches | `research/foundation_models/evaluation/linear_probe_cls_test_predictions.csv` (`ae54d559...`) | Peak Validation Macro-F1 ($C^*=20.0$, Val F1 = $65.32\%$) | `python -m research.foundation_models.linear_probe` (`task-2621`) | `d930e3d` (Phase 4.2) / `7330e8f` (Phase 4.2.1) |
| **6. DINOv2 Mean Patch Linear Probe** | `research/foundation_models/checkpoints/linear_probe_mean_patch.joblib` | `55c54ba01f294b74bcf21beee0ee3e3247ea8b7a90c72732c93adf54e94e9ddc` | Frozen Mean Patch Token ($d=384$) | Phase 2 Test Split (`research/splits/manifest_test.csv`) | 4,665 patches | `research/foundation_models/evaluation/linear_probe_mean_patch_test_predictions.csv` (`b3f0d806...`) | Peak Validation Macro-F1 ($C^*=200.0$, Val F1 = $66.06\%$) | `python -m research.foundation_models.linear_probe` (`task-2621`) | `d930e3d` (Phase 4.2) / `7330e8f` (Phase 4.2.1) |
| **7. DINOv2 CLS Nonlinear MLP Probe** | `research/foundation_models/checkpoints/nonlinear_probe_cls.pt` | `0cc6c4b8424fcad12f6d1f35d19347205f3364d41538c47f030dd16d966b2e77` | Frozen CLS Token ($d=384$) | Phase 2 Test Split (`research/splits/manifest_test.csv`) | 4,665 patches | `research/foundation_models/evaluation/nonlinear_probe_cls_test_predictions.csv` (`3574250f...`) | Peak Validation Macro-F1 (Config 3, Epoch 24, Val F1 = $66.32\%$) | `python -m research.foundation_models.nonlinear_probe` (`task-2947`) | `8dbdce6` (Phase 4.3) |
| **8. DINOv2 Mean Patch Nonlinear MLP Probe** | `research/foundation_models/checkpoints/nonlinear_probe_mean_patch.pt` | `353f04544f88869d96cec030aeca3293d0d9839ad64dc5675a33fd25528399c1` | Frozen Mean Patch Token ($d=384$) | Phase 2 Test Split (`research/splits/manifest_test.csv`) | 4,665 patches | `research/foundation_models/evaluation/nonlinear_probe_mean_patch_test_predictions.csv` (`4f60f7a6...`) | Peak Validation Macro-F1 (Config 3, Epoch 19, Val F1 = $66.81\%$) | `python -m research.foundation_models.nonlinear_probe` (`task-2947`) | `8dbdce6` (Phase 4.3) |

---

## 3. Embedding Cache Provenance

The frozen feature extraction pipeline (Phase 4.1, commit `ced00db`) extracted representations for all 31,096 images in the GCHTID dataset without backpropagation:

- **Cache Directory:** `research/foundation_models/cache/full/dinov2_vits14/`
- **Encoder Architecture:** Meta AI DINOv2 ViT-S/14 (`dinov2_vits14`)
- **Pretrained Weights Checksum:** PyTorch Hub release (`b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`)
- **CLS Token Cache File:** `cls_embeddings.npy`
  - Shape: $(31096, 384)$, `float32`
  - Size: $47,763,456$ bytes
  - SHA-256: `dc25ae822f2accd0df07ec2c4ab58424e817a2d6bb216b9be611e3d2880ffce5`
- **Mean Spatial Patch Cache File:** `mean_patch_embeddings.npy`
  - Shape: $(31096, 384)$, `float32`
  - Size: $47,763,456$ bytes
  - SHA-256: `1cab3ac533678f28de3b592529f63aeb1ab0981c9c93bfe38da4cb4ec483867c`
- **Manifest Alignment File:** `manifest.csv`
  - Total records: $31,096$ images
  - Canonical Class Order: `['ADI', 'DEB', 'LYM', 'MUC', 'MUS', 'NOR', 'STR', 'TUM']` (indices $0$ to $7$)

---

## 4. Separation of Evaluation Cohorts

A core finding of Phase 3.2 and Phase 3.3 was that direct comparisons between the historical checkpoint and downstream foundation model probes must strictly respect evaluation split boundaries:

### Cohort 1: The Historical Test Split ($N = 3,110$)
- **Formulation:** Formulated by original dataset authors prior to cluster isolation.
- **Evaluated Checkpoint:** `gastric_rnn_classifier.pt` (Epoch 15).
- **Observed Metrics:** Accuracy: $65.92\%$, Macro-F1: $66.11\%$.
- **Duplicate Exposure:** Contains 192 duplicate instances across 136 groups; strictly unexposed subset ($N=2,918$) achieves $65.66\%$ accuracy.
- **Applicability:** Benchmarks the historical model under its intended training/test partitioning.

### Cohort 2: The Phase 2 Test Split ($N = 4,665$)
- **Formulation:** Created in Phase 2 using image-level perceptual hash cluster isolation (dHash Hamming distance $\le 2$).
- **Evaluated Probes:** DINOv2 Linear and Nonlinear MLP probes.
- **Sample Distribution:** ADI: 583, DEB: 583, LYM: 583, MUC: 584, MUS: 583, NOR: 583, STR: 583, TUM: 583 (Total: $4,665$).
- **Historical Baseline Performance on this Cohort:**
  - Full cohort ($N=4,665$): $71.04\%$ accuracy, but **80.56% ($3,758$ images) were in the checkpoint's training split** (direct gradient leakage).
  - Truly unseen subset ($N=442$ images): Achieves **$65.84\%$ accuracy**.

---

## 5. Certification Sign-off

- [x] All 6 evaluation models possess persistent, serialized prediction CSV files on disk.
- [x] All prediction CSV files contain true labels, predicted labels, confidence scores, and full 8-class probability distributions.
- [x] Every prediction row in the Phase 2 test cohort ($N=4,665$) aligns path-by-path and index-by-index with `research/splits/manifest_test.csv`.
- [x] Checkpoint SHA-256 for `gastric_rnn_classifier.pt` matches the reference bit-for-bit.
- [x] DINOv2 embedding cache files match their reference SHA-256 bit-for-bit.

