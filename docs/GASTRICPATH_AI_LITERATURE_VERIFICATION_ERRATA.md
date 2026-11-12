# GASTRICPATH AI: SCIENTIFIC LITERATURE REVIEW & BENCHMARK AUDIT ERRATA

**Document Identifier:** `docs/GASTRICPATH_AI_LITERATURE_VERIFICATION_ERRATA.md`  
**Author:** Multidisciplinary Scientific Review Panel  
**Date of Audit:** November 12, 2026  
**Status:** Authoritative Scientific Correction Log  
**Scope:** Material Corrections to the Initial Literature Review and Novelty Assessment  

---

## 1. Executive Summary & Purpose of Audit

Following an exhaustive scientific and bibliographic audit conducted on November 12, 2026, several material errors, citation mismatches, overconfident empirical interpretations, and misattributed metrics in the initial literature review were identified and formally corrected. 

This errata document records every material modification with complete intellectual honesty. In rigorous academic research, identifying and correcting discrepancies prior to peer-reviewed manuscript submission is the hallmark of trustworthy science.

---

## 2. Itemized Errata & Material Corrections

### Erratum 1: Bibliographic Citation Mismatches (P28, P29, P30, P05)
* **P28 (SEMA6D in Gastric Cancer):**  
  * *Initial Record:* Cited as `Li, P., et al. (2021). Semaphorin 6D promotes gastric cancer progression and angiogenesis via ERK and PI3K/AKT pathways. Oncology Reports, 45, 52. DOI: 10.3892/or.2021.8003`.  
  * *Discrepancy:* DOI `10.3892/or.2021.8003` resolves to a completely different paper: **Wang et al. (2021), "BCAT1 overexpression regulates proliferation and c-Myc/GLUT1 signaling in head and neck squamous cell carcinoma"**, *Oncology Reports*, 45(5), Article 52.  
  * *Correction:* Replaced with authentic peer-reviewed studies on SEMA6D in gastric cancer: **Zhou, H., & Chen, G. (2024)**, *"Biological role of Semaphorin 6D in the proliferation, migration and invasion of gastric cancer"*, *African Health Sciences*, 24(3), 21 (DOI: [`10.4314/ahs.v24i3.21`](https://doi.org/10.4314/ahs.v24i3.21), PMID: `40777970`), along with **Lu et al. (2016)**, *Oncology Letters*, 12(4), 2821–2827 (DOI: [`10.3892/ol.2016.5182`](https://doi.org/10.3892/ol.2016.5182)) and **Zhao et al. (2006)**, *World Journal of Gastroenterology*, 12(45), 7388–7392 (DOI: [`10.3748/wjg.v12.i45.7388`](https://doi.org/10.3748/wjg.v12.i45.7388)).
* **P29 (miR-497-5p in Gastric Cancer):**  
  * *Initial Record:* Cited as `Wang, Y., et al. (2020). MicroRNA-497-5p inhibits cell proliferation and invasion in gastric cancer by targeting PDK3. Pathology - Research and Practice, 216, 153024. DOI: 10.1016/j.prp.2020.153024`.  
  * *Discrepancy:* DOI `10.1016/j.prp.2020.153024` resolves to a case report: **Parente et al. (2020), "Primary effusion lymphoma metachronous to multicentric Castleman disease in an immunocompetent patient"**, *Pathology - Research and Practice*, 216(7), 153024.  
  * *Correction:* Replaced with the authentic peer-reviewed publications establishing the miR-497-5p / PDK3 tumor suppressor axis in gastric cancer: **Feng, L., Cheng, K., Zang, R., Wang, Q., & Wang, J. (2019)**, *"miR-497-5p inhibits gastric cancer cell proliferation and growth through targeting PDK3"*, *Bioscience Reports*, 39(11), BSR20190654 (DOI: [`10.1042/BSR20190654`](https://doi.org/10.1042/BSR20190654), PMID: `31693723`), and **Chen, X., Zhou, L., Han, Y., et al. (2024)**, *"miR-497-5p Expression and Biological Activity in Gastric Cancer"*, *Journal of Cancer*, 15(7), 1930–1941 (DOI: [`10.7150/jca.90087`](https://doi.org/10.7150/jca.90087), PMID: `38495475`).
* **P30 (CCNE1 Amplification & Therapeutic Resistance):**  
  * *Initial Record:* Cited as `Ayala-Aguilera, C. C., et al. (2022). CCNE1 amplification is a predictive biomarker of resistance to targeted therapies in gastric and gastroesophageal junction adenocarcinoma. Clinical Cancer Research, 28, 4710–4721. DOI: 10.1158/1078-0432.CCR-22-1247`.  
  * *Discrepancy:* Author Ayala-Aguilera has no publication in *Clinical Cancer Research* on CCNE1 amplification in gastric cancer; the DOI suffix was mismatched.  
  * *Correction:* Replaced with the definitive translational studies on CCNE1 amplification and anti-HER2 resistance in gastric cancer: **Kim, S. T., et al. (2014)**, *"Preexisting oncogenic events impact trastuzumab sensitivity in ERBB2-amplified gastroesophageal adenocarcinoma"*, *Journal of Clinical Investigation*, 124(12), 5145–5154 (DOI: [`10.1172/JCI74346`](https://doi.org/10.1172/JCI74346), PMID: `25401476`), and *Cancer Research Communications* (2024, DOI: [`10.1158/2767-9764.CRC-23-0496`](https://doi.org/10.1158/2767-9764.CRC-23-0496)).
* **P05 (Kather et al. PLOS Medicine 2019):**  
  * *Correction:* Clarified that this foundational study (`10.1371/journal.pmed.1002730`) evaluated **colorectal cancer histology** (NCT-CRC-HE-100K), not gastric cancer. It established the 8-to-9 tissue-class microenvironment decomposition paradigm subsequently adopted by HMU-GC-HE-30K.

---

### Erratum 2: Expansion to Recent 2026 Literature
Two high-impact 2026 studies directly utilizing or benchmarking HMU-GC-HE-30K were incorporated into the master database and related-work analysis:
1. **Ding, P., Chen, S., Guo, H., Yang, S., et al. (2026)**  
   *Title:* *"A deep learning-based digital biopsy for predicting early recurrence in gastric cancer"*  
   *Journal:* ***Nature Communications***, DOI: [`10.1038/s41467-026-71347-6`](https://doi.org/10.1038/s41467-026-71347-6).  
   *Significance:* Evaluated multimodal recurrence risk in locally advanced gastric cancer (LAGC) across multicenter WSIs; utilized pathology foundation models (UNI, CONCH) and **utilized HMU-GC-HE-30K as an independent external test benchmark** for tumor microenvironment patch representation.
2. **Debsarkar, S. S., & Prasath, V. B. S. (February 2026)**  
   *Title:* *"A multi-expert deep learning framework with LLM-guided arbitration for multimodal histopathology prediction"*  
   *Journal:* ***Computerized Medical Imaging and Graphics***, 128, 102704, DOI: [`10.1016/j.compmedimag.2026.102704`](https://doi.org/10.1016/j.compmedimag.2026.102704).  
   *Significance:* Directly benchmarked **HMU-GC-HE-30K** alongside BCNB for a multi-expert framework (MELLMA) combining ResNet-50, ViT-B/16, EfficientNet-B3, and SwinV2-S with LLM-guided arbitration.

---

### Erratum 3: HMU-GC-HE-30K Public Data Release Truth
An audit of the official Figshare package (DOI: `10.6084/m9.figshare.25954813`) resolved the exact data reality:
- **No SVS Files Downloadable:** The public archive contains `HMU-GC-HE-30K.zip` (3.03 GB) and `HMU-GC-Clinical.csv` (22.72 KB). The 300 gigapixel SVS whole-slide images (>150 GB) are **not included** in the public release.
- **No Slide or Patch Identifiers:** Patch filenames are serialized class-stems (`ADI_1.png` to `ADI_3887.png`). There are **no slide IDs, no patient IDs, and no spatial pixel coordinates** in the patch files.
- **No Crosswalk Manifest:** `HMU-GC-Clinical.csv` lists 300 patient admission numbers (e.g. `1600097`), but provides zero linkage to patch filenames.
- **Scientific Implication:** In the public data release, **it is impossible to construct a true patient-level or slide-level split without unverified guessing**. Prior studies claiming patient-level splits either used private institutional crosswalks or only performed patient-level splitting on other datasets.

---

### Erratum 4: Correction of Duplicate Leakage & Novelty Claims
- **Reconstruction Context:** The reported **80.56% test contamination** and **+7.15 percentage-point accuracy inflation** were discovered during an **internal audit of a simulated split collision** between the historical baseline training manifest (`model_train/runs/dataset_manifest.csv`) and the newly generated Phase 2 split (`manifest_test.csv`), where 3,758 of 4,665 test tiles had been exposed during training (72.99% on trained tiles vs. 65.84% on clean tiles).
- **Critical Clarification:** This internal split collision **does NOT constitute empirical proof that the original 2025 Nature Scientific Data paper by Lou et al. was contaminated**, because Lou et al.'s specific 10-fold cross-validation split assignments were not published or reconstructed.
- **Mandatory Terminology:** The term *"certified leak-proof benchmark"* has been formally retracted and replaced with:
  > **"Duplicate-group-separated patch-level benchmark."**
  It is explicitly disclosed that this partition separates bitwise and perceptual duplicate clusters, but **does not guarantee patient-independent or slide-independent generalization**, which is fundamentally unverifiable without the non-public slide crosswalk.

---

### Erratum 5: Foundation Model Statistical Comparisons & P-Values
- **The $p = 0.00218$ Misattribution:** The initial report erroneously claimed that Phikon Mean-Patch statistically significantly outperformed Phikon `[CLS]` token with $p = 0.00218$.  
  *Audit Truth:* In `docs/PHASE4_4_REDUCED_COMPARISON_REPORT.md`, the McNemar test with $\chi^2 = 9.3945$ and $p = 0.00218$ was computed **strictly between Phikon Mean-Patch (66.38%) and DINOv2 Mean-Patch (59.31%)** ($\Delta = +7.07\text{ pp}$ on $N=467$).
- **Mean-Patch vs. CLS Difference:** On the 467 test tiles, Phikon Mean-Patch achieved **66.38%** vs. Phikon CLS **63.81%** ($\Delta = +2.57\text{ pp}$); DINOv2 Mean-Patch achieved **59.31%** vs. DINOv2 CLS **56.96%** ($\Delta = +2.35\text{ pp}$).
- **Correction of Biological Hypotheses:** Low cosine similarity ($\cos \theta \approx 0.060$) between `[CLS]` and Mean-Patch vectors proves mathematical distinction in representation space, but does **not** prove that Mean-Patch captures "diffuse microenvironmental signals" while `[CLS]` "dilutes" them. Biological assertions must remain framed as hypotheses.
- **CPU Latency Misattribution:** The recorded CPU warm latency of **38.56 ms** (25.94 samples/s) and peak RSS of **237.63 MB** belongs strictly to `GastricCancerRNN` (869,640 parameters), as documented in `research/benchmarks/cpu_benchmark_results.json`. It must **never** be attributed to Phikon (86.4M parameters).

---

### Erratum 6: Explainability and Robustness Terminology
- **Attribution Faithfulness vs. Causal Proof:** RISE deletion and insertion curves quantify **perturbation-based attribution faithfulness** (how well saliency maps reflect the model's internal feature reliance under input ablation), **not** causal biological proof or tumor area boundaries.
- **Sample Counts:** Baseline CNN explainability (Grad-CAM, Integrated Gradients, Occlusion, MC Dropout) was evaluated on **8 class exemplars** (1 per class). Token-level Leave-One-Out feature ablation was evaluated on a **32-tile gallery**.
- **The Four Perturbations Evaluated:** Phase 6 evaluated four specific synthetic perturbation families:
  1. *Mild Optical Gaussian Blur* ($5\times 5$, $\sigma=1.0$).
  2. *Brightness & Contrast Enhancement* ($1.1\times$ brightness, $1.1\times$ contrast).
  3. *Lossy JPEG Compression* (Quality 50).
  4. *H&E Stain Optical Density Variation* (+10% in Beer-Lambert optical density space).
- **False Claim Retracted:** The claim that "Gaussian noise" was evaluated has been **retracted**. Gaussian noise was not part of Phase 6.

---

### Erratum 7: Cellular & Spatial Graph Scope
- **Candidate Objects, Not Nuclei:** The 3,895 segmented objects extracted from 64 patches via Ruifrok HED color deconvolution and distance-transform watershed are **candidate cellular objects**, not pathologist-verified nuclei.
- **Spatial Proximity, Not Biological Signaling:** Graph edges generated via symmetric $k$-NN ($k=6$) represent **Euclidean spatial proximity**, not biological or paracrine signaling.
- **5-Fold Cross-Validation on $N=64$:** The performance of the non-graph Morphology MLP baseline ($23.75\% \pm 9.27\%$) versus GCN ($21.25\% \pm 10.46\%$), GraphSAGE ($18.75\% \pm 11.69\%$), and GAT ($17.50\% \pm 11.18\%$) exhibited substantial fold variance on 64 patches. Over-smoothing and topological noise are **plausible theoretical interpretations**, not experimentally isolated proofs.

---

### Erratum 8: Molecular and Biomarker Data Provenance
- **Computational Predictions, Not Clinical Assays:** All 3,711 interaction rows in `data/support/Gastric.xlsx` represent **in-silico bioinformatic predictions** across 8 computational databases, not patient-level expression assays.
- **SEMA6D & CCNE1 Evidence Separation:** SEMA6D is the single 8/8 multi-tool consensus target, and CCNE1 is a 7/8 consensus target. However, computational consensus **does not constitute experimental validation**. Experimental validation comes strictly from independent wet-lab literature (e.g. Zhou & Chen 2024 for SEMA6D; Kim et al. 2014 for CCNE1).

---

### Erratum 9: Review Methodology Classification
- **Classification:** The literature review is formally classified as a **structured scoping and narrative literature review with a systematic multi-database search methodology**, rather than a full PRISMA-compliant clinical systematic review.
- **Preprint Distinction:** Preprints (e.g., Filiot et al., 2023 for Phikon) are explicitly demarcated from peer-reviewed journal publications.

---

## 3. Summary of Status

All 8 core research artifacts in `docs/` and `research/literature_review/` have been updated to reflect these verified, audited corrections. The scientific foundation of GastricPath AI is now completely aligned with verifiable empirical evidence.
