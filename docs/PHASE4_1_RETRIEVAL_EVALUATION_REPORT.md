# GastricPath AI — Phase 4.1 Retrieval Evaluation Report
### Leakage-Aware Test-to-Train Nearest Neighbor Retrieval Benchmarking

**Document ID:** `GP-REPORT-PHASE4-1-RETRIEVAL-2026-V1`  
**Execution Phase:** Phase 4.1 (Full Dataset Retrieval Evaluation)  
**Target Checkpoint:** `models/checkpoints/gastric_rnn_classifier.pt`  
**Checkpoint SHA-256:** `65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8` (Bitwise Preserved)  
**Git Branch:** `gastricpath-ai-evolution`  
**Date:** November 2026  
**Status:** Completed & Empirically Verified (86/86 Automated Tests Passing)  

---

## 1. Executive Summary

Phase 4.1 evaluated the morphological representation capabilities of the frozen **DINOv2 ViT-S/14** encoder through a rigorous, leakage-aware test-to-train retrieval benchmark across all **4,665 test queries** searching against the **21,767 training reference patches**.

Operating with strict split isolation, exact-duplicate exclusion, and perceptual-cluster exclusion, the benchmark assessed two self-supervised representation strategies:
1. **CLS Token Embeddings (`[31096, 384]`)**
2. **Mean Spatial Patch-Token Embeddings (`[31096, 384]`)**

### Key Empirical Findings:
- **Top-1 Class Agreement (Hit@1):** **47.31%** (CLS) vs. **47.33%** (Mean Patch). Pure random chance across 8 balanced classes is $12.50\%$, demonstrating that the frozen self-supervised encoder projects histologically related patches into proximate neighborhoods without any supervised fine-tuning or domain adaptation.
- **Majority Agreement@10:** When aggregating the top-10 retrieved neighbors by majority vote, class consistency rises to **54.98%** (CLS) and **55.13%** (Mean Patch).
- **Sub-Cellular vs. Architectural Phenotypes:** The model excels at identifying distinct sub-cellular patterns such as **Adipose tissue (`ADI`: 66.90% P@1)** and **Lymphocytes (`LYM`: 61.92% P@1)**, while exhibiting expected histological confusion between structurally similar tissue compartments (e.g. smooth muscle `MUS` vs. fibrous stroma `STR`, and non-neoplastic mucosa `NOR` vs. well-differentiated tumor epithelium `TUM`).
- **High Gallery Utilization:** **14,484 out of 21,767 training patches (66.54%)** were retrieved in top-10 candidate sets, confirming the absence of pathological hubness or representation collapse.
- **Verified Zero Leakage:** An exhaustive post-retrieval audit verified that **0.0% of retrieved neighbors** shared exact SHA-256 hashes or perceptual dHash cluster identities with query images.

---

## 2. Mathematical Formulation & Retrieval Architecture

Given an L2-normalized query representation $\mathbf{q} \in \mathbb{R}^D$ ($\|\mathbf{q}\|_2 = 1.0$) and a reference gallery matrix $\mathbf{X}_{\text{train}} \in \mathbb{R}^{N_{\text{train}} \times D}$ ($\|\mathbf{x}_j\|_2 = 1.0 \;\; \forall j$):

### 1. Cosine Similarity Equivalence:
Because all vectors are unit-normalized, cosine similarity reduces exactly to the matrix-vector dot product:
$$\cos(\mathbf{q}, \mathbf{x}_j) = \frac{\mathbf{q} \cdot \mathbf{x}_j}{\|\mathbf{q}\|_2 \|\mathbf{x}_j\|_2} = \mathbf{q} \cdot \mathbf{x}_j$$
Batch similarity across all $M = 4,665$ test queries and $N = 21,767$ training patches is computed as:
$$\mathbf{S} = \mathbf{Q}_{\text{test}} \mathbf{X}_{\text{train}}^T \in \mathbb{R}^{4665 \times 21767}$$
This computation requires only $\sim 406\text{ MB}$ of transient RAM and executes in **1.56 seconds** on CPU ($\sim 3,000$ queries/sec).

### 2. Leakage-Aware Exclusion Rules:
For each query $\mathbf{q}_i$ with file path $p_i$, SHA-256 $h_i$, and duplicate cluster ID $c_i$, a gallery candidate $j$ with properties $(p_j, h_j, c_j)$ is filtered prior to ranking:
1. **Self-Exclusion Rule:** If $p_j == p_i$, exclude.
2. **Exact Duplicate Rule:** If $h_j == h_i$, exclude.
3. **Near-Duplicate Cluster Rule:** If $c_j == c_i$ and $c_i \neq \text{None}$, exclude.
4. **Reference Split Rule:** Only gallery items satisfying $\text{split}_j == \text{"train"}$ are permissible. Test images are permanently quarantined from the reference index.

---

## 3. Retrieval Performance Comparison: CLS vs. Mean Patch Tokens

We evaluated both representations across $k \in \{1, 5, 10\}$ nearest neighbors:

| Representation Type | Top-k ($k$) | Class Consistency ($P@k$) | Majority Agreement@k | Mean Cosine Sim | Median Cosine Sim | 25th Pct Sim | 75th Pct Sim |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CLS Token (384-dim)** | **k = 1** | **47.31%** ($\pm 49.9\%$) | **47.31%** | 0.9234 | 0.9334 | 0.9022 | 0.9541 |
| | **k = 5** | **45.00%** ($\pm 29.8\%$) | **52.95%** | 0.9165 | 0.9276 | 0.8935 | 0.9482 |
| | **k = 10** | **43.42%** ($\pm 24.6\%$) | **54.98%** | 0.9124 | 0.9242 | 0.8885 | 0.9450 |
| **Mean Patch Token (384-dim)** | **k = 1** | **47.33%** ($\pm 49.9\%$) | **47.33%** | 0.9413 | 0.9490 | 0.9248 | 0.9654 |
| | **k = 5** | **45.28%** ($\pm 29.8\%$) | **52.33%** | 0.9359 | 0.9446 | 0.9174 | 0.9613 |
| | **k = 10** | **43.82%** ($\pm 24.7\%$) | **55.13%** | 0.9327 | 0.9419 | 0.9130 | 0.9589 |

### Observations:
1. **Equivalence of Top-1 Accuracy:** CLS ($47.31\%$) and Mean Spatial Patch ($47.33\%$) yield virtually identical top-1 retrieval agreement ($\Delta = +0.02\%$).
2. **Higher Cosine Density in Spatial Tokens:** Mean patch tokens exhibit higher absolute cosine similarity (Mean: $0.9413$ vs. $0.9234$ for CLS) due to spatial smoothing across 256 tokens.
3. **Majority Voting Lift:** Aggregating top-10 neighbors via majority vote improves agreement from $47.3\%$ to **$\sim 55.0\%$**, demonstrating that true class signal reinforces across multiple neighbors while spurious background matches disperse.

---

## 4. Per-Class Retrieval Performance Breakdown

Below is the detailed per-class retrieval consistency ($P@k$) for the primary CLS token representation across the 8 canonical classes ($N = 583$ or $584$ queries per class):

| Canonical Tissue Class | Description | P@1 (Top-1 Match) | P@5 (Top-5 Consistency) | P@10 (Top-10 Consistency) | Majority Agreement@10 | Easiest / Hardest Rank |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **ADI** | Adipose Tissue | **66.90%** | **63.43%** | **60.36%** | **70.67%** | Rank 1 (Easiest) |
| **LYM** | Lymphocytes | **61.92%** | **57.77%** | **55.42%** | **68.27%** | Rank 2 |
| **TUM** | Tumor Epithelium | **48.20%** | **47.41%** | **46.71%** | **58.66%** | Rank 3 |
| **MUS** | Smooth Muscle | **48.03%** | **44.97%** | **43.76%** | **56.95%** | Rank 4 |
| **STR** | Cancer-Associated Stroma | **45.28%** | **45.42%** | **44.92%** | **56.60%** | Rank 5 |
| **DEB** | Necrotic Debris | **39.62%** | **37.84%** | **36.57%** | **48.03%** | Rank 6 |
| **NOR** | Normal Gastric Mucosa | **34.48%** | **32.86%** | **31.39%** | **43.74%** | Rank 7 |
| **MUC** | Mucus | **34.08%** | **30.34%** | **28.27%** | **36.82%** | Rank 8 (Hardest) |
| **Overall Macro Mean** | *Unweighted Class Average* | **47.31%** | **45.00%** | **43.42%** | **54.98%** | — |

---

## 5. Top-1 Retrieval Confusion Matrix & Histological Analysis

Below is the complete count matrix comparing the true test query class against the class of the top-1 retrieved training neighbor:

| True Class \ Retrieved | ADI | DEB | LYM | MUC | MUS | NOR | STR | TUM | Total Queries | Recall / P@1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ADI** | **390** | 21 | 3 | 91 | 42 | 20 | 6 | 10 | 583 | **66.90%** |
| **DEB** | 9 | **231** | 83 | 29 | 42 | 43 | 65 | 81 | 583 | **39.62%** |
| **LYM** | 4 | 77 | **361** | 8 | 21 | 28 | 51 | 33 | 583 | **61.92%** |
| **MUC** | 96 | 66 | 10 | **199** | 51 | 63 | 43 | 56 | 584 | **34.08%** |
| **MUS** | 25 | 58 | 28 | 30 | **280** | 33 | 112 | 17 | 583 | **48.03%** |
| **NOR** | 37 | 56 | 54 | 42 | 19 | **201** | 69 | 105 | 583 | **34.48%** |
| **STR** | 4 | 53 | 58 | 16 | 76 | 43 | **264** | 69 | 583 | **45.28%** |
| **TUM** | 6 | 76 | 37 | 18 | 8 | 74 | 83 | **281** | 583 | **48.20%** |

### Pathological Explanation of Primary Error Modes:
1. **Adipose (`ADI`) $\leftrightarrow$ Mucus (`MUC`) [91 cases]:** Both tissue types present as pale, optically clear spaces with sparse nuclear material (lipid clearing in adipocytes vs. extracellular mucin lakes).
2. **Smooth Muscle (`MUS`) $\leftrightarrow$ Stroma (`STR`) [112 cases]:** Both compartments consist of elongated spindle cells embedded in eosinophilic collagenous matrices. Distinguishing muscularis propria from dense fibrous stroma without immunohistochemical stains (e.g. SMA/Desmin) is challenging for general visual representations.
3. **Tumor (`TUM`) $\leftrightarrow$ Normal Mucosa (`NOR`) [74 cases] and Stroma (`STR`) [83 cases]:** Well-differentiated adenocarcinoma retains glandular morphology similar to normal gastric glands (`NOR`). In poorly differentiated or scirrhous carcinoma, malignant cells infiltrate stroma as single cells or cords, causing strong stromal feature overlap (`STR`).
4. **Debris (`DEB`) $\leftrightarrow$ Lymphocytes (`LYM`) [83 cases]:** Karyorrhectic apoptotic fragments in necrosis mimic pyknotic dark lymphocyte nuclei.

---

## 6. Audit & Leakage Verification

An independent programmatic audit was executed during retrieval:

- **Self-Matches Blocked:** **0** (Query paths were completely isolated).
- **Exact SHA-256 Duplicates Blocked:** **0** (Confirming Phase 2 split successfully quarantined all exact duplicate groups).
- **dHash Perceptual Clusters Blocked:** **0** (Confirming Phase 2 split successfully grouped all near-duplicate clusters).
- **Cross-Split Leakage Check:** **100% of queries were `test`** ($N=4,665$), and **100% of retrieved neighbors were `train`** ($N=21,767$).
- **Gallery Coverage:** **14,484 unique training patches** were returned in top-10 lists across all test queries (**66.54% gallery coverage**), demonstrating high representational diversity.

---

## 7. Research Visualizations Produced

Four high-resolution research figures were generated in `research/foundation_models/visualizations/`:

1. **`similarity_distribution.png`:** Histogram illustrating cosine similarity distributions across Top-1 (Mean: $0.923$), Top-5 (Mean: $0.917$), and Top-10 (Mean: $0.912$) retrieval ranks.
2. **`retrieval_at_k_by_class.png`:** Grouped bar chart comparing $P@1$, $P@5$, and $P@10$ across all 8 canonical tissue classes.
3. **`retrieval_confusion_matrix.png`:** Heatmap showing percentage neighbor agreement matrix for all class pairs.
4. **`embedding_pca_umap.png`:** 2D Principal Component Analysis (PCA) projection of 2,000 sampled patch embeddings (250 per class) explaining $18.4\%$ (PC1) and $11.2\%$ (PC2) of variance, demonstrating natural clustering of adipose, lymphoid, and muscular tissues.

---

## 8. Mandatory Scientific & Clinical Disclosures

> [!IMPORTANT]
> **SCIENTIFIC & CLINICAL DISCLOSURE STATEMENT:**
> 1. **Retrieval Precision $\neq$ Supervised Classification Accuracy:** The reported $47.31\%$ top-1 retrieval consistency reflects unsupervised nearest-neighbor clustering in a frozen representation space. It must not be directly compared with supervised classifier accuracy without noting that the foundation model was never trained or fine-tuned on the GCHTID dataset.
> 2. **No Clinical Diagnostic Generalization:** Retrieved nearest neighbors demonstrate visual morphological proximity. They do NOT constitute automated histologic diagnosis, cancer grading, staging, or clinical decision support.
> 3. **Image-Level Limitation:** Because GCHTID filenames do not contain patient or whole-slide identifiers, retrieval neighbors reflect patch-level visual similarity and cannot prove patient-level generalization.

---

*Report certified by GastricPath AI Research Engine.*
