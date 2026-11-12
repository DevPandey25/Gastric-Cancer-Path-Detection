# GastricPath AI — Model Checkpoint & Architecture Guide

This document details the machine learning architectures, certified baseline weights, lineage verification, and model evaluation protocols utilized in GastricPath AI.

---

## 1. Baseline Architecture: Lightweight CNN + BiLSTM + Attention

GastricPath AI intentionally utilizes a lightweight custom architecture designed for efficient CPU execution and interpretable feature extraction without the computational overhead of heavy ResNet-50 backbones:
- **Spatial Encoder:** 4 convolutional blocks with batch normalization, ReLU activation, and max pooling, extracting 128-dimensional spatial feature maps.
- **Sequential Recurrent Aggregator:** Bidirectional Long Short-Term Memory (BiLSTM) network (hidden dimension: 64, 2 layers), modeling spatial sequence transitions across flattened tile rows.
- **Self-Attention Pooling:** Additive attention layer producing attention weights over sequential hidden states.
- **Classification Head:** Linear projection into the 8 tissue class logits (`ADI`, `DEB`, `LYM`, `MUC`, `MUS`, `NOR`, `STR`, `TUM`).
- **Total Parameters:** ~850,000 parameters (3.49 MB file footprint).

---

## 2. Certified Baseline Checkpoint Verification

The certified baseline model checkpoint is stored at:
```
models/checkpoints/gastric_rnn_classifier.pt
```

### Cryptographic Checksum
To guarantee reproducible science and protect against bitrot or tampering, the checkpoint is verified against its certified SHA-256 hash:
```
65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8
```

### Verification Command (PowerShell)
```powershell
Get-FileHash models/checkpoints/gastric_rnn_classifier.pt -Algorithm SHA256
```

### Verification Command (Python)
```python
import hashlib

def verify_checkpoint(path="models/checkpoints/gastric_rnn_classifier.pt"):
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest().lower()

assert verify_checkpoint() == "65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8"
print("Checkpoint verified successfully.")
```

---

## 3. Retraining Procedure

To train a new model checkpoint using the full dataset:
```powershell
python -m training.train --epochs 10 --batch-size 32 --lr 0.001 --output-dir models/checkpoints/
```

Training telemetry and loss curves are automatically logged to `model_train/logs/` and `model_train/runs/`.

---

## 4. Foundation Model Comparisons (Phikon & DINOv2)

For research benchmarking, GastricPath AI includes comparison modules evaluating self-supervised foundation models:
- **Phikon:** Pathology-specific foundation model pretrained on 40M+ histology tiles from The Cancer Genome Atlas (TCGA).
- **DINOv2 (ViT-S/14):** Vision transformer pretrained by Meta AI on 142M general visual images.
- Linear probe and nonlinear multi-layer perceptron (MLP) evaluation scripts are located in `research/foundation_models/`.
