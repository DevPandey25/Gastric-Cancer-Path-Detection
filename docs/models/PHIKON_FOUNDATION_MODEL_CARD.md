# Model Card: Owkin Phikon (Computational Pathology Foundation Model)

## 1. Model Overview
- **Model Name:** `owkin/phikon`
- **Architecture:** Vision Transformer Base (`ViT-B/16`), 12 transformer encoder blocks, 12 attention heads
- **Patch Resolution:** $16 \times 16$ pixel input patch projection ($14 \times 14 = 196$ spatial tokens per $224 \times 224$ image)
- **Pretraining Methodology:** iBOT self-supervised masked image modeling with online tokenizer
- **Pretraining Dataset:** The Cancer Genome Atlas (TCGA) pan-cancer collection (~400 million histological tiles)
- **Feature Embedding Dimension:** $768$-D normalized continuous vector representation
- **Evaluation Partition:** Phase 4.4 Reduced 10% Stratified Test Subset ($N = 467$ held-out patches)

---

## 2. Evaluation Results on GCHTID (Traceable Artifact Audit: $N = 467$ Test Partition)

In the Phase 4.4 comparative benchmark (`research/foundation_models/cache/subset10pct/evaluation/evaluation_summary.json`), frozen Phikon feature representations were evaluated across downstream probing heads on the held-out test split ($N = 467$):

| Probing Configuration | Optimal $C^*$ | Pooling Method | Accuracy (95% CI) | Macro F1 (95% CI) | Macro OvR AUROC | ECE | Brier Score |
|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **Phikon Linear Probe (Winning)** | **10.0** | **Mean-Patch Pooling** | **66.38%** [61.98%, 70.52%] | **66.54%** [61.58%, 70.85%] | **0.9450** | **0.0524** | **0.4487** |
| Phikon Linear Probe | 1.0 | CLS Token Only | 63.81% [59.36%, 68.04%] | 63.75% [58.95%, 67.78%] | 0.9326 | 0.1154 | 0.4947 |

*Multiclass AUROC Definition:* Macro One-vs-Rest (OvR) unweighted arithmetic average of binary class AUROC curves across all 8 canonical tissue classes.

---

## 3. Key Findings & Cross-Phase Reconciliation
1. **Mean-Patch Superiority:** Spatial average pooling over all 196 patch tokens statistically significantly outperforms CLS token readout alone ($+2.57\text{ pp}$ accuracy gain; McNemar paired test $p = 0.00218$ on 467 test samples), demonstrating that histological diagnostic cues in gastric tissue are spatially dispersed across glandular and stromal regions rather than concentrated at a single summary token.
2. **Pathology Pretraining Advantage:** Phikon Mean Patch ($66.38\%$) significantly outperformed general-domain vision models (Meta DINOv2 ViT-S/14 Mean Patch achieved $59.31\%$ on the identical 467 test samples; $+7.07\text{ pp}$ difference, paired bootstrap 95% CI $[+2.78, +11.56]\text{ pp}$), verifying the domain-specific value of histopathology pretraining on TCGA.
3. **Reconciliation with Phase 4.2 Full-Dataset DINOv2 Result:**
   - In Phase 4.2 (`docs/PHASE4_2_IMPLEMENTATION_REPORT.md`), DINOv2 was trained on the full training set ($N=21,767$) and tested on the full test set ($N=4,665$), achieving $65.08\%$ accuracy.
   - The Phase 4.4 Phikon benchmark was conducted on the 10% stratified subset (training $N=2,177$, testing $N=467$). With only 10% training data, Phikon achieved $66.38\%$, surpassing DINOv2's full-dataset linear probe. These two evaluation populations differ in sample size ($N=467$ vs $N=4,665$) and training regime ($N=2,177$ vs $N=21,767$) and must not be directly equated.
4. **Similarity Search Retrieval:** The 768-D mean-patch embeddings form the retrieval index for the GastricPath AI content-based image retrieval (CBIR) engine (`research/foundation_models/cache/subset10pct/phikon/mean_patch_embeddings.npy`, 3,110 samples, normalized cosine similarity).

