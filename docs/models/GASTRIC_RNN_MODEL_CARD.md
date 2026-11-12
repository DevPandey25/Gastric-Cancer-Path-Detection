# Model Card: GastricCancerRNN (Baseline Classification Model)

## 1. Model Overview
- **Model Name:** `GastricCancerRNN`
- **Model Type:** Hybrid Convolutional-Recurrent Neural Network with Temporal Attention Pooling
- **Checkpoint Location:** `models/checkpoints/gastric_rnn_classifier.pt`
- **Checkpoint File Size:** 3,485,348 bytes (3.32 MB)
- **Parameter Count:** 869,640 trainable parameters
- **Authoritative SHA-256 Digest:**
  `65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8`
- **Execution Target:** Optimized for standalone CPU inference; GPU-compatible when CUDA is present.

---

## 2. Architecture & Design

The `GastricCancerRNN` processes a $224 \times 224 \times 3$ histopathology patch by partitioning it into an ordered grid sequence of $16$ non-overlapping sub-patches of dimension $56 \times 56 \times 3$:

```
Input Patch (224x224x3)
  │
  ▼
Patch Tiling Sequence (16 x [56x56x3])
  │
  ▼
Shared CNN Backbone (Conv2D -> BatchNorm -> ReLU -> MaxPool2D)
  │ (Extracts 128-D spatial feature vector per sub-patch)
  ▼
Bidirectional GRU / LSTM Sequence Layer (Hidden dim = 64)
  │ (Models inter-tile morphological spatial transitions)
  ▼
Attention Pooling Layer (Learned dynamic tile weighting)
  │
  ▼
Linear Classifier Head (Dense 128 -> 8 Logits) -> Softmax
  │
  ▼
Class Probabilities (8 GCHTID Tissue Classes)
```

---

## 3. Training & Validation Provenance (Traceable Artifact Audit)
- **Dataset:** GCHTID 8-Class Partition (Historical dataset splits & Phase 3 audit)
- **Checkpoint Internal State Dict Metric:** `val_accuracy: 0.6486` ($64.86\%$, recorded at checkpoint serialization in `models/checkpoints/gastric_rnn_classifier.pt`).
- **Model Registry Baseline Record:** Cataloged in `research/registry/models.json` with validation accuracy of **$64.86\%$**, test accuracy of **$56.82\%$**, and test loss of **$1.2732$** (historical notebook benchmark).
- **Historical Test Cohort ($N=3,110$ patches, Phase 3.2 Audit):**
  - **Overall Test Accuracy:** **$65.92\%$**
  - **Strictly Unexposed Partition Accuracy:** **$65.66\%$**
  - Detailed in `docs/PHASE3_2_EXPERIMENT_AND_DATA_LEAKAGE_AUDIT.md`.
- **Erroneous 88.22% Metric Retraction:** Previous draft documentation cited an $88.22\%$ validation accuracy. Thorough inspection of the checkpoint binary state dict confirmed `val_accuracy: 0.6485530546623794`. The $88.22\%$ figure has been formally retracted as an unverified draft metric.
- **Model Role:** Fast baseline inference engine for real-time web UI patch evaluation.

---

## 4. Benchmark Performance (CPU Telemetry & Scope)
*Audited on Windows 11, AMD Ryzen 5 5500U (6 cores / 12 threads), 16 GB RAM, Python 3.13.5, PyTorch 2.9.1+cpu (batch size = 1, 100 warm repetitions, single-worker local execution):*
- **Model Cold-Start Initialization:** $53.85\ \text{ms}$
- **Preprocessing Latency (224x224 crop & sequence):** $1.51 \pm 1.31\ \text{ms}$
- **Inference Latency (Warm Mean):** **$38.56\ \text{ms}$** per patch
- **Median Latency (p50):** $36.25\ \text{ms}$
- **95th Percentile Latency (p95):** $50.08\ \text{ms}$
- **Inference Throughput:** **$25.94\ \text{patches/sec}$** (single worker, batch size = 1)
- **Peak Process Memory RSS:** $237.63\ \text{MB}$
- **Throughput & Capacity Disclosure:** These single-worker benchmark results do **not** establish concurrent-user capacity, multi-threaded throughput, or production server scalability under network load.
- **GPU Benchmark Status:** CUDA hardware absent; GPU benchmarking honestly recorded as not applicable.
- **GPU Benchmark Status:** CUDA hardware absent; GPU benchmarking honestly recorded as not applicable.

---

## 5. Usage & Scientific Safeguards
1. **Patch-Level Classification:** Predictions represent patch-level categorical estimates.
2. **Standardized TUM Output:** The predicted tumor probability is formally designated as:
   > **`Predicted TUM class probability (patch-level model output)`**
3. **Clinical Disclaimer:** Not certified or approved by regulatory bodies (FDA, CE-IVD, NMPA) for clinical diagnosis, patient staging, or therapeutic guidance.

