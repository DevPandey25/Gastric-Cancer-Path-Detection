# GASTRICPATH AI: PUBLICATION EXPERIMENT ROADMAP & FUTURE RESEARCH EXTENSIONS

**Document Identifier:** `docs/GASTRICPATH_AI_PUBLICATION_EXPERIMENT_ROADMAP.md`  
**Author:** Multidisciplinary Scientific Review Panel  
**Target Repository:** GastricPath AI  
**Date of Audit & Literature Cutoff:** November 12, 2026  
**Status:** Actionable Experimental Roadmap  

---

## 1. Executive Summary & Phased Experimental Philosophy

To ensure that the recommended publication (*"Benchmarking Foundation Models, Attribution Faithfulness, and Calibrated Selective Classification in Gastric Cancer Histopathology: An Audited Evaluation of HMU-GC-HE-30K"*) withstands the highest levels of peer review, future experimental work is organized into three distinct tiers:

1. **Priority 1: Immediate Pre-Submission Enhancements (Zero Financial Cost / Native Repository Data):**  
   Experiments executable directly on the current hardware and dataset without requiring external GPU clusters or newly acquired clinical cohorts.
2. **Priority 2: High-Value Methodological Refinements (Low Friction / Fast Turnaround):**  
   Ablation studies and cross-dataset evaluations that substantially fortify the manuscript against skeptical reviewers.
3. **Priority 3: Data-Dependent Clinical Expansions (Long-Term Horizons):**  
   Longitudinal extensions conditional upon acquiring patient-to-slide crosswalks, whole-slide image cohorts (e.g., TCGA-STAD), and expert nuclear segmentations.

```
+----------------------------------------------------------------------------------------------------+
|                               EXPERIMENTAL ROADMAP TIMELINE & RESOURCE MATRIX                      |
+--------------------------+-----------------------+------------------------+------------------------+
| Experimental Priority    | Compute / Resource    | Estimated Execution    | Target Manuscript Goal |
+--------------------------+-----------------------+------------------------+------------------------+
| PRIORITY 1:              | Native CPU / Python   | 1 to 2 Weeks           | Submission Readiness   |
| Pre-Submission Core      | Existing Repositories | (Zero Cost)            | (Closes all empirical  |
|                          | N = 4,665 Test Data   |                        | gaps for peer review)  |
+--------------------------+-----------------------+------------------------+------------------------+
| PRIORITY 2:              | Native CPU / PyTorch  | 2 to 4 Weeks           | Revision & Rebuttal    |
| Methodological Ablations | Public Benchmark Data | (Zero Cost)            | (Preempts reviewer     |
|                          | (GasHisSDB / TCGA)    |                        | edge-case objections)  |
+--------------------------+-----------------------+------------------------+------------------------+
| PRIORITY 3:              | Cloud GPU Cluster     | 3 to 6 Months          | Follow-Up Journal /    |
| Multi-Modal & WSI Scale  | Institutional WSIs    | (Grant / Partner Data) | Clinical Trial Phase   |
|                          | Clinical Crosswalk    |                        |                        |
+--------------------------+-----------------------+------------------------+------------------------+
```

---

## 2. Priority 1: Essential Pre-Submission Experiments (Immediate)

These three experiments require zero external software, zero paid cloud compute, and zero additional data collection. They execute entirely within the existing GastricPath AI virtual environment.

### Experiment 1.1: Full Test-Set Evaluation of Phikon and DINOv2 ($N = 4,665$)
* **Objective:** Expand the foundation model linear probing benchmark from the stratified pilot subset ($N=467$ test partition) to the complete, duplicate-group-separated test partition of **4,665 images**.
* **Current State in Repository:** Phase 4.4 evaluated Phikon and DINOv2 on 3,110 tiles (467 test). Phase 4.2 extracted full 31,096 embeddings for DINOv2 (achieving 65.08% accuracy on 4,665 test). The Phikon full-dataset extraction was paused at 10,400 images (33.4%) in Phase 4.4.2 to accelerate workbench delivery.
* **Protocol:**
  1. Complete the background extraction of frozen 768-D Phikon embeddings across the remaining test partition images.
  2. Fit the validation-tuned multinomial logistic probe ($C$ parameter sweep: $[0.001, 0.01, 0.1, 1.0, 10.0]$) strictly on the 21,767 training embeddings.
  3. Evaluate both `[CLS]` token and Mean-Patch pooling across all 4,665 certified test patches.
* **Target Output:** A comprehensive 4,665-patch metric table with exact Accuracy, Balanced Accuracy, Macro-F1, OvR AUROC, and ECE.

### Experiment 1.2: Rigorous Statistical Significance Testing (McNemar & DeLong Tests)
* **Objective:** Provide mathematical proof that the observed performance divergences between `GastricCancerRNN` (baseline), `Phikon Mean-Patch`, `Phikon CLS`, and `DINOv2 Mean-Patch` are statistically significant across the full cohort.
* **Protocol:**
  1. **McNemar's Paired Test:** Construct $2 \times 2$ contingency tables comparing binary correctness between paired models on identical test instances. Compute two-tailed asymptotic $p$-values with continuity correction:
     $$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$
     *Audited Baseline:* On the $N=467$ test subset, McNemar significance was established between Phikon Mean-Patch (66.38%) and DINOv2 Mean-Patch (59.31%) with $p = 0.00218$ ($\chi^2 = 9.3945$). The full test run will evaluate whether the $+2.57\text{ pp}$ gap between Phikon Mean-Patch and Phikon CLS reaches strict significance at $N=4,665$.
  2. **DeLong's Non-Parametric Test:** Test statistical differences between paired One-vs-Rest ROC curves using DeLong's algorithm to obtain variance estimates and $z$-scores for multi-class macro AUROC.
  3. **Paired Cluster-Bootstrap ($B=1,000$):** Resample unique duplicate clusters to construct empirical 95% confidence intervals for paired accuracy differences ($\Delta \text{Acc}$).
* **Target Output:** Publication-ready Table 3 with exact contingency matrices, $\chi^2$ values, and $p$-values.

### Experiment 1.3: Temperature Scaling Calibration Optimization across All Encoders
* **Objective:** Demonstrate that post-hoc temperature scaling reliably reduces Expected Calibration Error (ECE) across all candidate architectures without altering classification accuracy.
* **Protocol:**
  1. For each model (`GastricCancerRNN`, `Phikon Mean-Patch`, `Phikon CLS`, `DINOv2`), optimize scalar temperature $T^* > 0$ strictly on validation logits ($N=4,664$) by minimizing negative log-likelihood (NLL) via L-BFGS-B:
     $$\min_{T} -\sum_{i=1}^{N_{\text{val}}} \log \left( \frac{\exp(z_{i, y_i} / T)}{\sum_{j=1}^K \exp(z_{i, j} / T)} \right)$$
  2. Evaluate calibrated probabilities on the test set using 10 equal-width bins.
  3. Measure the reduction in ECE (e.g., from 0.0524 down to $<0.0200$) and Brier score.
* **Target Output:** Figure 4 dual-panel reliability diagrams illustrating pre- and post-calibration confidence curves.

---

## 3. Priority 2: High-Value Methodological Refinements (Short-Term)

These experiments directly preempt anticipated reviewer criticisms regarding generalizability and representation depth.

### Experiment 2.1: Cross-Dataset Domain Shift Evaluation on GasHisSDB
* **Objective:** Quantify the robustness of the trained GastricPath AI classifiers under hospital and scanner domain shifts by testing on the independent *GasHisSDB* dataset (Hu et al., 2022).
* **Protocol:**
  1. Ingest normal and abnormal gastric patches from GasHisSDB (Harbin Medical University 160x160 cohort).
  2. Map the 8-class GastricPath predictions to binary status (NOR vs. TUM/DEB).
  3. Measure accuracy degradation ($\Delta \text{Acc}$) between native HMU-GC-HE-30K and external GasHisSDB patches.
  4. Evaluate whether Mahalanobis distance OOD screening successfully flags the domain-shifted GasHisSDB tiles.
* **Value to Paper:** Proves that feature representations generalize beyond a single scanning protocol.

### Experiment 2.2: Layer-Wise Token Pooling Dynamics Across Transformer Depths
* **Objective:** Investigate at which layer depth Mean-Patch pooling diverges from `[CLS]` token routing within the Vision Transformer.
* **Protocol:**
  1. Extract intermediate representations from Phikon ViT-B/16 at Blocks 3, 6, 9, and 12.
  2. Compute cosine similarity $\cos(\text{CLS}_l, \text{MeanPatch}_l)$ at each layer $l$.
  3. Train linear probes on each layer's representations.
* **Context:** In Phase 8, final-layer cosine similarity collapsed to $\cos \theta \approx 0.060$, demonstrating mutually orthogonal feature subspaces. Layer-wise tracking will clarify where this divergence emerges.
* **Value to Paper:** Provides profound theoretical insight into token specialization in pathology foundation models.

### Experiment 2.3: Sequence Order Permutation Ablation on `GastricCancerRNN`
* **Objective:** Verify whether the Bidirectional LSTM in `GastricCancerRNN` exploits spatial sequential adjacency across the 16 sub-tiles ($56 \times 56$), or acts merely as an unordered feature aggregator.
* **Protocol:**
  1. Evaluate the certified checkpoint on normal sequential raster scan order (sub-tiles 1 to 16).
  2. Evaluate under random spatial permutation of the 16 sub-tiles.
  3. Evaluate under reverse scan order (sub-tiles 16 down to 1).
* **Hypothesis:** A significant drop under random shuffling validates spatial sequential context as an active inductive bias.
* **Value to Paper:** Rigorously validates the architectural hypothesis of the sequence baseline.

### Experiment 2.4: Expanded Perturbation Suite: Additive Gaussian Noise
* **Objective:** Expand the Phase 6 robustness audit (which evaluated Gaussian blur, contrast/brightness scaling, JPEG Q50, and HED optical density shift across 2,335 predictions) by adding additive zero-mean Gaussian noise ($\sigma \in \{0.05, 0.10, 0.20\}$).
* **Value to Paper:** Preempts reviewer queries regarding sensor thermal noise and sensor artifacts.

---

## 4. Priority 3: Data-Dependent Clinical Expansions (Long-Term Horizons)

These long-term extensions outline the future research trajectory for follow-up grants, institutional collaborations, and clinical translation.

### Experiment 3.1: Validated Nuclear Segmentation via PanNuke / NuCLS
* **Current Limitation:** GastricPath AI Phase 7 and Phase 8 utilized classical Ruifrok HED color deconvolution + marker-controlled watershed detection (3,895 candidate objects across 64 patches). The Tabular Morphology MLP baseline ($23.75\% \pm 9.27\%$) performed on par with GCN ($21.25\% \pm 10.46\%$) with high fold variance.
* **Roadmap:**
  1. Fine-tune a deep nuclear instance segmentation model (HoVer-Net, Graham et al., 2019; or StarDist 2D) on the public *PanNuke* or *CoNSeP* multi-tissue histology benchmarks.
  2. Deploy the fine-tuned instance segmenter to extract verified nuclear boundaries and cell phenotypes (neoplastic epithelial, inflammatory, stromal, necrotic) on gastric patches.
  3. Reconstruct spatial cell graphs using biologically typed nodes.
* **Impact:** Upgrades the spatial graph engine from candidate geometric proximity to true biological cell-cell interaction modeling.

### Experiment 3.2: Slide-Level Weakly Supervised Multiple Instance Learning on TCGA-STAD
* **Current Limitation:** Public HMU-GC-HE-30K consists strictly of isolated $224 \times 224$ patches; zero clinical WSIs are downloadable from the public Figshare repository. GastricPath AI's WSI engine (`research/wsi/`) is implemented with Level-0 coordinate handling and GatedAttentionMIL, but evaluated on synthetic test fixtures.
* **2026 Prior Art Context:** Ding et al. (2026, *Nature Communications*) evaluated recurrence on 2,840 clinical WSIs using UNI and CONCH, using HMU-GC-HE-30K as an external patch test set.
* **Roadmap:**
  1. Download the public TCGA-STAD cohort ($\approx 440$ diagnostic FFPE whole slide images in `.svs` format).
  2. Process all WSIs through GastricPath AI's Otsu HSV tissue detector and Level-0 coordinate tiling pipeline.
  3. Extract 768-D Phikon Mean-Patch embeddings for all tissue tiles.
  4. Train GatedAttentionMIL, CLAM-SB, and TransMIL to predict slide-level molecular subtypes (EBV, MSI-H, GS, CIN) and 5-year overall survival.
* **Impact:** Establishes GastricPath AI as a complete, clinical-grade whole-slide diagnostic platform.

### Experiment 3.3: Multimodal Vision-Tabular Fusion with Genuine Patient Crosswalk
* **Current Limitation:** The repository contains 300 clinical records in `HMU-GC-Clinical.csv` and extensive biomarker data in `Gastric.xlsx`, but lacks a verified patient-to-patch crosswalk manifest and longitudinal survival time (`OS_months`).
* **Roadmap:**
  1. Obtain institutional clinical re-identification crosswalk linking the 300 WSIs from Harbin Medical University Cancer Hospital to specific patient IDs.
  2. Extract patient survival duration and recurrence status from oncology electronic health records.
  3. Unlock GastricPath AI's multimodal fusion modules (`ClinicalMLPEncoder`, `EarlyFusionClassifier`, `LateFusionClassifier`) to train joint vision-tabular survival models using multivariable Cox proportional hazards loss.
* **Impact:** Delivers the gold standard of multimodal precision oncology: integrating routine H&E morphology, clinical staging, and microRNA regulatory networks to predict individual patient chemotherapy response and prognosis.

---

## 5. Execution Summary for Authors

| Milestone | Action Item | Code Artifact / Script | Expected Time |
|:---|:---|:---|:---:|
| **Step 1** | Run McNemar & DeLong tests on existing baseline outputs | `research/benchmarks/statistical_tests.py` | 1 Day |
| **Step 2** | Optimize Temperature Scaling across validation logits | `research/explainability/temperature_scaling.py` | 1 Day |
| **Step 3** | Complete full test partition Phikon extraction ($N=4,665$) | `research/foundation_models/extract_phikon.py` | 3 Days |
| **Step 4** | Generate Publication Figures 1 through 6 | `research/benchmarks/generate_figures.py` | 3 Days |
| **Step 5** | Assemble LaTeX manuscript with `references.bib` | `manuscript/main.tex` | 5 Days |

**Final Recommendation:** Proceed directly with Step 1 and Step 2. The existing GastricPath AI codebase, model checkpoints, embedding caches, and test suite provide everything required to submit a top-tier peer-reviewed manuscript immediately.