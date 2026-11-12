# Model Card: Meta DINOv2 (General Vision Foundation Model)

## 1. Model Overview & Architectural Distinction
- **Model Name:** `facebookresearch/dinov2`
- **Architectural Distinction:**
  - **`dinov2_vits14` (Evaluated & Cached):** Vision Transformer Small, $14 \times 14$ patch, $384$-D embedding dimension, $22.05\ \text{M}$ frozen parameters. This is the **sole DINOv2 model cached and evaluated with linear probes** in both Phase 4.2 (full dataset) and Phase 4.4 (10% stratified benchmark).
  - **`dinov2_vitb14` (Surveyed Only):** Vision Transformer Base, $14 \times 14$ patch, $768$-D embedding dimension, $86.6\ \text{M}$ parameters. Evaluated during the Phase 4.0 architectural feasibility survey (`docs/PHASE4_0_FOUNDATION_MODEL_FEASIBILITY_REPORT.md`); **no feature cache or downstream probing checkpoints exist** in the repository due to offline CPU compute constraints.
- **Pretraining Methodology:** DINOv2 self-supervised learning with discriminative feature learning, centering, and sinkhorn-knopp normalization.
- **Pretraining Dataset:** LVD-142M (curated dataset of 142 million natural general-domain images).

---

## 2. Evaluation Results on GCHTID (Traceable Artifact Audit)

### Phase 4.4 Reduced 10% Stratified Benchmark ($N = 467$ Held-Out Test Partition)
Source: `research/foundation_models/cache/subset10pct/evaluation/evaluation_summary.json`

| Model Configuration | Embedding Dim | Pooling Method | Optimal $C^*$ | Test Accuracy (95% CI) | Macro F1 (95% CI) | Macro OvR AUROC | ECE | Brier Score |
|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **DINOv2 ViT-S/14 Linear Probe** | **384-D** | **Mean-Patch** | **10.0** | **59.31%** [54.80%, 63.68%] | **59.24%** [54.54%, 63.69%] | **0.9196** | **0.0485** | **0.5198** |
| DINOv2 ViT-S/14 Linear Probe | 384-D | CLS Token | 100.0 | 56.96% [52.43%, 61.38%] | 56.87% [52.31%, 61.13%] | 0.9108 | 0.1165 | 0.5562 |

### Phase 4.2 Full-Dataset Benchmark ($N = 4,665$ Full Test Split)
Source: `docs/PHASE4_2_IMPLEMENTATION_REPORT.md` (trained on $N=21,767$ full training split)

| Model Configuration | Embedding Dim | Pooling Method | Test Accuracy (95% CI) | Macro F1 (95% CI) | Macro AUROC | ECE |
|:---|:---:|:---|:---:|:---:|:---:|:---:|
| **DINOv2 ViT-S/14 (Full Dataset)** | **384-D** | **Mean-Patch** | **65.08%** [63.70%, 66.44%] | **65.03%** [63.75%, 66.38%] | **0.9316** | **0.0368** |
| DINOv2 ViT-S/14 (Full Dataset) | 384-D | CLS Token | 64.35% [62.97%, 65.71%] | 64.21% [62.93%, 65.64%] | 0.9293 | 0.0220 |

---

## 3. Comparative Analysis: Natural Images vs. Pathology Pretraining
1. **Domain Gap Penalty on Identical Samples:** On the Phase 4.4 test split ($N=467$), DINOv2 ViT-S/14 lagged Owkin Phikon ViT-B/16 by **$7.07\text{ pp}$ in accuracy** ($59.31\%$ vs $66.38\%$) and $0.0254$ in Macro OvR AUROC ($0.9196$ vs $0.9450$). This paired difference was statistically significant (McNemar $p = 0.00218$, paired bootstrap 95% CI $[+2.78, +11.56]\text{ pp}$), quantifying the representation gap between natural photography and histopathology structures.
2. **Evaluation Population Distinction:** DINOv2's 65.08% result required full-dataset training ($N=21,767$ samples), whereas Phikon achieved 66.38% with only 10% training data ($N=2,177$). Metrics from $N=4,665$ and $N=467$ represent separate evaluation cohorts and must not be conflated.

