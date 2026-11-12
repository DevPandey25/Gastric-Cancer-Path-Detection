# GastricPath AI — Research Release Readiness Checklist

This checklist provides a structured, verifiable audit of all technical, scientific, security, and reproducibility criteria required for the formal research release of GastricPath AI.

---

## 1. Model Preservation & Baseline Invariants
- [x] **Checkpoint Bitwise Integrity:** `models/checkpoints/gastric_rnn_classifier.pt` has an exact SHA-256 hash of `65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8`.
- [x] **No Retraining of Baseline:** Checkpoint weights and architecture parameters (869,640) are strictly intact. Checkpoint internal `val_accuracy` is 64.86% (0.6486); registry records val 64.86%, test 56.82%; historical Phase 3.2 test set evaluation achieved 65.92% (65.66% unexposed). The earlier draft figure of 88.22% is formally retracted.
- [x] **Model Registry Alignment:** `research/registry/models.json` accurately catalogues the baseline model, Owkin Phikon probes, Meta DINOv2 probes, and U-Net architecture.

---

## 2. Scientific Consistency & Terminology Standards
- [x] **Standardized Tumor Probability Wording:** All patch-level tumor metrics are labeled as **`Predicted TUM class probability (patch-level model output)`**.
- [x] **No Patient Cancer Diagnoses:** Patch classifications are explicitly disclaimed as patch-level tissue categorization, not anatomical tumor area percentage or clinical diagnoses.
- [x] **HER-2 Terminology:** HER-2 IHC categorization is labeled as **`IHC 2+/3+ exploratory category — not confirmed HER2-positive status`** ($24.67\%$, 74/300 patients) versus negative/baseline ($75.33\%$, 226/300 patients), noting absence of confirmatory FISH gene-amplification assays.
- [x] **In-Silico miRNA Terminology:** MicroRNA target interactions are formally designated as **`In-silico miRNA target ranking`** for mature miRNA `hsa-miR-497-5p` across 3,711 interaction rows in `data/support/Gastric.xlsx`, identifying 11 high-consensus target genes ($\ge 7/8$ algorithms) with explicit disclosure of computational algorithms and lack of wet-lab validation.
- [x] **Spatial Graph Configuration:** Accurately documented as **Symmetric $k$-NN ($k=6$, Euclidean distance, bidirectional with self-loops) over 16-D normalized node features**.
- [x] **Spatial Graph Metrics:** Accurately distinguishes **Balanced Accuracy (MLP: 23.75%, GCN: 21.25%)** from **Accuracy (MLP: 22.05%, GCN: 20.51%)**, confirming that the non-graph Morphology MLP baseline outperformed all GNNs on the 64-patch pilot.
- [x] **Foundation Model Evaluation Partition:** Accurately documented as evaluated on **$N = 467$ held-out test patches** (from the 3,110 10% stratified subset) where Phikon Mean-Patch achieved **66.38% test accuracy, 0.9450 AUROC, 66.54% Macro F1**. Distinct from Phase 4.2 full-dataset DINOv2 protocol (65.08%, $N=4,665$).
- [x] **Segmentation Blocker Disclosure:** U-Net explicitly cataloged as **Untrained / Blocked by Data** due to zero pixel-level masks or GeoJSON tumor annotations in GCHTID. No fabricated Dice or IoU scores.
- [x] **Cellular Detection Caveat:** Classical Ruifrok HED-watershed detections explicitly designated as **candidate objects**, not manually validated nuclei. Deep learning models disclosed as blocked due to missing pretrained weights.
- [x] **Whole-Slide Processing Disclosure:** Pure-Python WSI engine validated on coordinate grids; zero clinical WSIs in GCHTID honestly documented.

---

## 3. Data Governance & Privacy Compliance
- [x] **Individual Identifiers Withheld:** Patient hospital admission numbers present in raw clinical CSV are strictly withheld from application outputs, templates, and API payloads.
- [x] **Clinical Cohort Invariant:** HMU-GC-Clinical contains **$N = 300$ patient records** (0 excluded rows; empty trailing columns dropped; missing values median-imputed with indicators).
- [x] **Unlinked Cohort Provenance:** Explicitly discloses that HMU-GC-Clinical records are independent and unlinked from GCHTID image tiles.

---

## 4. Hardware Telemetry & Benchmarking
- [x] **Real-Time CPU Benchmarking:** Standalone benchmark script executed with empirical measurements stored in `research/benchmarks/cpu_benchmark_results.json`:
  - Model Load Latency: $53.85\ \text{ms}$
  - Single-Patch Preprocessing: $1.51 \pm 1.31\ \text{ms}$
  - Warm Inference Latency: $38.56\ \text{ms}$
  - Throughput: $25.94\ \text{patches/sec}$
  - Peak Process RSS: $237.63\ \text{MB}$
- [x] **GPU Honest Documentation:** NVIDIA CUDA absence honestly documented as not applicable; no fabricated GPU speedups reported.

---

## 5. Security & Production Hardening
- [x] **Elimination of Indiscriminate Static Mounting:** Removed `app.mount("/research-artifacts")` that previously exposed the full `research/` directory.
- [x] **Allowlisted Visual Artifact Endpoint:** Implemented `/api/research/artifacts/{category}/{filename}` restricted to 10 approved research categories and visual image extensions (`.png`, `.jpg`, `.jpeg`). Excluded `.svg` to prevent potential script execution vulnerabilities.
- [x] **Path Traversal Guards:** Implemented strict traversal rejection on report retrieval (`/api/research/reports/{filename}`) and visual artifacts (`/api/research/artifacts/{category}/{filename}`).
- [x] **HTTP Security Headers Middleware:** Added headers for `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `X-XSS-Protection: 1; mode=block`, `Referrer-Policy`, and `Content-Security-Policy`.
- [x] **File Upload Validation:** Implemented multi-layer validation in `/api/predict`: file extension allowlisting (`.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`), **10 MB maximum size limit**, PIL image content decoding and integrity check (`img.verify()`), dimension bounding ($\le 4096 \times 4096$), and decompression-bomb cap (`Image.MAX_IMAGE_PIXELS = 16_000_000`).

---

## 6. Deployment & Environment Artifacts
- [x] **Environment Configuration Template:** `.env.example` created with default production values and zero credentials.
- [x] **Pinned Requirements:** `requirements.txt` updated with pinned dependencies compatible with Python 3.10–3.13 on CPU.
- [x] **Container Definition:** `Dockerfile` created with multi-stage build, non-root user (`gastricpath`), and CPU PyTorch wheels.
- [x] **Container Exclusions:** `.dockerignore` created excluding large raw datasets, virtual environments, caches, `.env`, and local secrets.
- [x] **Docker Compose Configuration:** `docker-compose.yml` created with persistent volume mounts and health checks.
- [x] **Docker Host Status Documented:** Formally disclosed as "configuration authored, build/runtime verification unavailable" due to absence of Docker daemon/CLI on the Windows host.

---

## 7. Documentation & Cards Suite
- [x] **`docs/REPRODUCIBILITY.md`:** Complete step-by-step reproduction instructions for all research pipelines and benchmarks.
- [x] **`docs/VALIDATION_AND_LIMITATIONS.md`:** Comprehensive disclosure of clinical boundaries, offline sandbox constraints, and data blockers.
- [x] **`docs/PHASE12_TESTING_DEPLOYMENT_RESEARCH_RELEASE_REPORT.md`:** Authoritative Phase 12 completion report.
- [x] **Dataset Cards Suite:**
  - `docs/datasets/GCHTID_DATASET_CARD.md`
  - `docs/datasets/HMU_GC_CLINICAL_DATA_CARD.md` ($N=300$, 0 dropped rows)
  - `docs/datasets/GASTRIC_MIRNA_TARGET_DATA_CARD.md` (11 high-consensus targets for `hsa-miR-497-5p`)
- [x] **Model Cards Suite:**
  - `docs/models/GASTRIC_RNN_MODEL_CARD.md`
  - `docs/models/PHIKON_FOUNDATION_MODEL_CARD.md`
  - `docs/models/DINOV2_FOUNDATION_MODEL_CARD.md` (ViT-S/14 evaluated downstream, ViT-B/14 surveyed only)
  - `docs/models/UNET_SEGMENTATION_MODEL_CARD.md`
  - `docs/models/CELLULAR_HED_WATERSHED_CARD.md`
  - `docs/models/SPATIAL_GNN_MODEL_CARD.md`

---

## 8. Verification & Test Suite
- [x] **Phase 12 Automated Test Suite:** `tests/test_original_phase12.py` authored (16/16 tests passing).
- [x] **Full Repository Regression:** Complete test suite passes with 100% pass rate (334/334 tests passing).
- [x] **Zero Remote Network Calls:** All tests and operations run strictly offline.
- [x] **Repository Cleanliness:** 0 files staged or committed to Git.

