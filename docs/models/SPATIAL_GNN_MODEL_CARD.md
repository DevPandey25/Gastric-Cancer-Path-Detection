# Model Card: Spatial Graph and GNN Engine

## 1. Module Overview
- **Module Name:** Spatial Graph Neural Network (GNN) Engine & Morphological Baselines
- **Research Phase:** Original Phase 8 (Spatial Graph and GNN Engine)
- **Role:** Graph-level representation learning and classification of histopathology patches from cellular candidate networks
- **Pilot Scope:** 64 image patches from GCHTID (8 balanced samples per class, 3,895 total candidate nodes)
- **Graph Topology:** Symmetric $k$-Nearest-Neighbors ($k = 6$, Euclidean distance, bidirectional edges with self-loops)
- **Node Features:** 16-dimensional normalized morphological and spatial coordinate vectors

---

## 2. Evaluated Architectures & Benchmark Results

All models were evaluated using 5-fold cross-validation on the 64-patch pilot dataset:

| Model Architecture | Graph Type / Aggregation | Trainable Parameters | Balanced Accuracy | Accuracy | Macro F1 |
|:---|:---|:---:|:---:|:---:|:---:|
| **Morphology MLP Baseline (Winning)** | **Non-graph Global Pooling** (Mean/Std/Max over 16-D features) | **6,408** | **23.75%** | **22.05%** | **20.48%** |
| Graph Convolutional Network (GCN) | First-order spectral Chebyshev / Symmetric degree normalization | 7,560 | 21.25% | 20.51% | 19.34% |
| Graph Attention Network (GAT) | Multi-head self-attention weighting (4 heads) | 8,128 | 18.75% | 17.95% | 16.82% |
| GraphSAGE | Inductive mean neighborhood sampling & aggregation | 9,216 | 17.50% | 16.67% | 15.44% |

---

## 3. Major Empirical Findings
1. **Morphology MLP Superiority:** The non-graph Morphology MLP baseline, which simply pools global summary statistics (mean, standard deviation, max) of the 16-D morphological features, outperformed all three spatial graph architectures across both Balanced Accuracy ($23.75\%$ vs $21.25\%$) and Overall Accuracy ($22.05\%$ vs $20.51\%$).
2. **Spatial Message-Passing Overhead:** On this small 64-patch pilot, spatial graph message passing introduced over-smoothing and excess variance without providing discriminative advantage over global cellular morphology distributions.
3. **Top Graph Architecture:** Among the message-passing graph architectures, GCN demonstrated superior stability and generalization compared to parameterized attention (GAT) or sampling-based aggregation (GraphSAGE).

---

## 4. Methodological & Biological Disclosures
1. **Geometric Proximity Only:** Graph edges connect spatial candidate centroids based purely on Euclidean geometric proximity ($k=6$). They do **not** represent verified molecular signaling, paracrine cellular communication, or receptor-ligand interactions.
2. **Candidate-Object Nodes:** Graph nodes originate from the automated classical Ruifrok HED-watershed candidate object segmenter. They are candidate structures, not pathologist-verified nuclei.

