# GastricPath AI — Original Phase 10: Multimodal Clinical Research Report

**Document Version:** 1.0.0  
**Phase Identifier:** Original Phase 10 (Multimodal Clinical Research)  
**Status:** Clinical Tabular & In-Silico Molecular Engines Operational / Multimodal Fusion and Survival Modeling Strictly Blocked by Data Linkage & Outcome Absence  
**Author:** GastricPath AI Research Team  
**Date:** November 2026  
**Audited Cohort:** 
- GCHTID Histopathology: 31,096 isolated $224 \times 224$ PNG image tiles across 8 tissue categories.
- Harbin Medical University Clinical Cohort (`HMU-GC-Clinical.csv`): 300 gastric cancer patient records across 15 standardized clinical variables.
- Molecular Target Workbook (`Gastric.xlsx`): 3,711 in-silico predicted microRNA-target interactions for `hsa-miR-497-5p` across 3,497 unique human genes.
**Baseline Model Invariant:** `models/checkpoints/gastric_rnn_classifier.pt` (SHA-256: `65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8`, verified intact).

---

## 1. Executive Summary and Operational Classification

Original Phase 10 executes the Multimodal Clinical Research program for the GastricPath AI platform, adhering to the authoritative specification in `GastricPath_AI_Research_and_Implementation_Blueprint.docx`. This phase addresses the integration, modeling, and rigorous provenance auditing of three distinct data modalities in the repository:
1. **Histopathology:** The Gastric Cancer Histopathology Tile Image Dataset (GCHTID; 31,096 tiles).
2. **Clinical Phenotypes:** The Harbin Medical University Gastric Cancer Clinical Cohort (`HMU-GC-Clinical.csv`; 300 patients).
3. **Molecular MicroRNA Predictions:** The computationally predicted microRNA target interaction workbook (`Gastric.xlsx`; 3,711 records).

```
+----------------------------------------------------------------------------------------------------+
|                                ORIGINAL PHASE 10 ARCHITECTURAL AUDIT                               |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|    [ GCHTID Histopathology Tiles ]          [ HMU-GC-Clinical Cohort ]         [ Gastric.xlsx ]    |
|             (N = 31,096)                             (N = 300)                    (N = 3,711)      |
|        Format: ADI-001.png, etc.               IDs: 1600097, 1502891, etc.         miRNA: miR-497  |
|         Patient IDs: ABSENT                     Patient IDs: PRESENT             Patient IDs: ABSENT|
|                  \                                   /                                     /       |
|                   \                                 /                                     /        |
|                    \_______ [ LINKAGE AUDIT ] _____/                                     /         |
|                                    |                                                    /          |
|                       Pairwise Linkage: UNAVAILABLE                                    /           |
|                       Crosswalk Lookup Table: ABSENT                                  /            |
|                                    |                                                 /             |
|                                    v                                                v              |
|                   +-----------------------------------+             +-----------------------------+|
|                   |   MULTIMODAL FUSION GATEKEEPER    |             |  SURVIVAL ELIGIBILITY AUDIT ||
|                   |  MultimodalLinkageBlockedError:   |             | SurvivalAnalysisBlockedError||
|                   | Real Multimodal Training BLOCKED  |             |  Follow-up / Vital ABSENT   ||
|                   +-----------------------------------+             +-----------------------------+|
|                                    |                                                |              |
|                                    v                                                v              |
|                   +-----------------------------------+             +-----------------------------+|
|                   |   UNIMODAL CLINICAL BENCHMARK     |             | IN-SILICO REGULATORY NETWORK||
|                   | HER-2 Prediction (5-Fold CV)      |             | NetworkX Graph Construction ||
|                   | Logistic Reg: AUROC 0.421 ± 0.071 |             | Consensus >= 7: 11 Targets  ||
|                   | Random Forest: AUROC 0.490 ± 0.106|             | CCNE1, FBXW7, SEMA6D, etc.  ||
|                   +-----------------------------------+             +-----------------------------+|
+----------------------------------------------------------------------------------------------------+
```

> [!IMPORTANT]
> **Strict Scientific Integrity Protocol & Operational Classification:**
> Following an exhaustive audit of all identifiers, headers, and metadata across the repository:
> 1. **No Shared Identifiers:** GCHTID image tiles use sequential class-stem labels (`ADI-001.png`, `TUM-124.png`) with zero patient identifiers; `HMU-GC-Clinical.csv` contains 300 integer admission numbers; `Gastric.xlsx` contains Ensembl gene IDs and microRNA accessions.
> 2. **Prohibition of Fabricated Joins:** Merging datasets by row index, sorting, arbitrary partitioning, or disease category assumption was strictly rejected.
> 3. **Prohibition of Synthetic Crosswalks:** No artificial lookup tables or synthetic patient IDs were introduced into production pipelines.
> 4. **Prohibition of Label Pollution:** Patch-level tissue categories (e.g., ADI, TUM) were not transferred to patient clinical records.
> 5. **Scientific Clarification of Biomarker Data:** All 3,711 records in `Gastric.xlsx` were confirmed to be *in-silico* bioinformatic target predictions for `hsa-miR-497-5p`, **not** patient-level laboratory measurements or clinical expression assays.
> 6. **Survival Analysis Blocked:** Follow-up duration (time-to-event) and vital status (dead/alive) are absent from `HMU-GC-Clinical.csv`. Fitting Kaplan-Meier or Cox models to non-survival cross-sectional data was blocked to prevent scientific invalidity.
>
> Original Phase 10 is formally classified as:  
> **Clinical Tabular Preprocessing, Unimodal Baseline Modeling, and In-Silico Molecular Network Infrastructure Implemented and Software-Verified / Multimodal Fusion Training and Survival Modeling Strictly Blocked Awaiting Genuine Clinical Crosswalk and Longitudinal Follow-Up Data.**

---

## 2. Dataset Audits and Cross-Dataset Provenance Analysis

### 2.1 The Three Evaluated Datasets

| Dataset Identifier | File Path | Record Count | Unique Entities | Identifier Nature | Patient IDs Present? |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **GCHTID Histopathology** | `research/manifests/manifest_all.csv` | 31,096 | 31,096 image tiles | Class sequence stems (`ADI-001.png` ... `TUM-3887.png`) | **No** |
| **HMU-GC-Clinical** | `data/support/HMU-GC-Clinical.csv` | 300 | 300 patients | 7-digit admission IDs (`1600097`, `1502891`, etc.) | **Yes** |
| **Gastric.xlsx Workbook** | `data/support/Gastric.xlsx` (`Table s2`) | 3,711 | 3,497 unique gene targets | Ensembl gene IDs (`ENSG...`) & microRNA accessions | **No** |

### 2.2 Pairwise Linkage Audit Results

The `PatientLinkageAuditor` executed automated compatibility checks across all pairwise combinations of these datasets:

1. **GCHTID Tiles $\longleftrightarrow$ HMU-GC-Clinical Records:**
   - **Linkage Status:** `UNAVAILABLE`
   - **Compatible Identifiers:** 0
   - **Matched Records:** 0 / 31,096 (0.00%)
   - **Blocker Description:** GCHTID image tiles were extracted and distributed as anonymized, isolated $224 \times 224$ crops labeled solely by morphological class and integer indices. The repository contains no accession manifest mapping tile sequences or original whole-slide scans to the 300 Harbin Medical University clinical patient records.
   - **Unblocking Requirement:** An institutional crosswalk associating individual slide accessions or image filenames with HMU patient IDs.

2. **HMU-GC-Clinical Records $\longleftrightarrow$ Gastric.xlsx In-Silico Predictions:**
   - **Linkage Status:** `UNAVAILABLE`
   - **Compatible Identifiers:** 0
   - **Matched Records:** 0 / 300 (0.00%)
   - **Blocker Description:** `Gastric.xlsx` contains in-silico computational predictions of microRNA-gene targeting. It possesses no patient identification numbers, clinical specimen identifiers, or experimental expression quantifications. The clinical cohort contains no molecular microRNA profiling.
   - **Unblocking Requirement:** Specimen-matched molecular assays (e.g., RT-qPCR or miRNA-Seq) measuring `hsa-miR-497-5p` expression across the 300 HMU patients.

3. **GCHTID Tiles $\longleftrightarrow$ Gastric.xlsx In-Silico Predictions:**
   - **Linkage Status:** `UNAVAILABLE`
   - **Compatible Identifiers:** 0
   - **Matched Records:** 0 / 31,096 (0.00%)
   - **Blocker Description:** Zero specimen or spatial transcriptomic bridges exist between the pixel-level H&E tiles and the bioinformatic miRNA target predictions.
   - **Unblocking Requirement:** Spatial transcriptomics (e.g., 10x Visium) or localized in-situ hybridization on matched tissue sections.

---

## 3. Clinical Cohort Schema, Missingness, and Descriptive Analysis

### 3.1 Schema Cleaning and Field Harmonization

The raw `data/support/HMU-GC-Clinical.csv` file contained 17 column headers. An automated audit identified and dropped two trailing empty artifact columns (`Unnamed: 15` and `Unnamed: 16`), retaining 15 standardized clinical variables for 300 unique patient admissions.

All 300 patient identifiers (`Patient`) were verified to be strictly unique ($100\%$ uniqueness, zero duplicate records).

### 3.2 Formal Clinical Data Dictionary

| Standardized Column Name | Raw CSV Header | Variable Type | Missing Count (%) | Valid Value Range / Categories | Preprocessing & Modeling Role |
| :--- | :--- | :--- | :---: | :--- | :--- |
| `patient_id` | `Patient` | Identifier | 0 (0.0%) | 300 unique 7-digit IDs | Excluded from feature matrix; grouping key |
| `sex` | `Sex` | Binary Nominal | 0 (0.0%) | Male ($N=236$), Female ($N=64$) | Binary indicator (`Male=1, Female=0`) |
| `age` | `Age` | Numerical | 0 (0.0%) | 33–79 years ($60.77 \pm 9.09$) | StandardScaler normalization |
| `t_stage` | `T staging` | Ordinal | 0 (0.0%) | T1a, T1b, T2, T3, T4a, T4b | One-hot encoding |
| `n_stage` | `N staging` | Ordinal | 0 (0.0%) | N0, N1, N2, N3a, N3b | One-hot encoding |
| `m_stage` | `M staging` | Binary Nominal | 0 (0.0%) | M0 ($N=276$), M1 ($N=24$) | Binary indicator (`M1=1, M0=0`) |
| `comprehensive_stage` | `Comprehensive Staging` | Ordinal | 0 (0.0%) | ⅠA, ⅠB, ⅡA, ⅡB, ⅢA, ⅢB, ⅢC, Ⅳ | Ordinal grouping / One-hot encoding |
| `histological_type` | `Histological Type` | Nominal | 0 (0.0%) | Tubular, Mucinous, Poorly diff, etc. | One-hot encoding |
| `lauren_classification` | `Lauren classification(...)` | Nominal | 55 (18.33%) | Intestinal (0), Mixed (1), Diffuse (2) | One-hot encoding + Missing indicator flag |
| `lymphovascular_invasion`| `Lymphovascular Invasion(...)` | Binary | 50 (16.67%) | Negative (0), Positive (1) | Binary imputation + Missing indicator flag |
| `venous_invasion` | `Venous Invasion(...)` | Binary | 83 (27.67%) | Negative (0), Positive (1) | Binary imputation + Missing indicator flag |
| `perineural_invasion` | `Perineural Invasion(...)` | Binary | 52 (17.33%) | Negative (0), Positive (1) | Binary imputation + Missing indicator flag |
| `stroma_quantity` | `Stroma Quantity(...)` | Ordinal | 18 (6.00%) | Medullary (0), Intermediate (1), Scirrhous (2)| Ordinal imputation + Missing indicator flag |
| `tumor_infiltration_pattern`| `Tumor Infiltration Pattern` | Nominal | 20 (6.67%) | INFa (Expanding), INFb, INFc (Infiltrative) | One-hot encoding + Missing indicator flag |
| `her2_status` | `HER-2(Negative=0,...)` | Ordinal | 0 (0.0%) | Negative (0), 1+, 2+, 3+ | Primary classification target |

### 3.3 Missingness Quantification and Demographic Distributions

Missingness is concentrated in pathological microinvasion parameters:
- **Venous Invasion:** $27.67\%$ missing ($N=83$).
- **Lauren Classification:** $18.33\%$ missing ($N=55$).
- **Perineural Invasion:** $17.33\%$ missing ($N=52$).
- **Lymphovascular Invasion:** $16.67\%$ missing ($N=50$).
- **Tumor Infiltration Pattern:** $6.67\%$ missing ($N=20$).
- **Stroma Quantity:** $6.00\%$ missing ($N=18$).
- All core macro-staging variables (`Age`, `Sex`, `T staging`, `N staging`, `M staging`, `Comprehensive Staging`, `HER-2`) exhibited $0.00\%$ missingness.

```
Missingness Rate by Feature:
[██████████████] Venous Invasion (27.67%)
[█████████     ] Lauren Classification (18.33%)
[████████      ] Perineural Invasion (17.33%)
[████████      ] Lymphovascular Invasion (16.67%)
[███           ] Tumor Infiltration Pattern (6.67%)
[███           ] Stroma Quantity (6.00%)
[              ] Age, Sex, TNM, Comprehensive Stage, HER-2 (0.00%)
```

### 3.4 Clinical Visualizations

The clinical audit generated publication-grade figures saved in `research/multimodal/visualizations/`:

1. **Clinical Missingness Heatmap:**  
   [Clinical Missingness Heatmap](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/multimodal/visualizations/clinical_missingness_heatmap.png)  
   Illustrates patient-level sparsity patterns across clinical variables. Shows co-occurring missingness across microinvasion metrics.

2. **Cohort Demographics and Staging Distribution:**  
   [Clinical Cohort Demographics](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/multimodal/visualizations/clinical_cohort_demographics.png)  
   Multi-panel visualization depicting age distribution by biological sex, comprehensive AJCC pathological staging proportions, Lauren histopathology distribution, and immunohistochemical HER-2 expression breakdown.

---

## 4. Unimodal Clinical Tabular Modeling Benchmark

### 4.1 Benchmark Target Definition and Rationale

In clinical oncology, human epidermal growth factor receptor 2 (HER2 / *ERBB2*) overexpression dictates eligibility for targeted trastuzumab therapy in advanced gastric adenocarcinoma. In `HMU-GC-Clinical.csv`, HER-2 status is scored by immunohistochemistry (IHC) on a 4-tier clinical scale:
- IHC 0: Negative ($N=140$, $46.67\%$)
- IHC 1+: Negative/Equivocal ($N=86$, $28.67\%$)
- IHC 2+: Equivocal/Overexpressing ($N=48$, $16.00\%$)
- IHC 3+: Strong Positive Overexpression ($N=26$, $8.67\%$)

Following standard ASCO/CAP clinical guidelines:
- **HER-2 Overexpression (Positive, Class 1):** IHC 2+ or 3+ ($N = 74$, prevalence $24.67\%$).
- **HER-2 Non-Overexpression (Negative, Class 0):** IHC 0 or 1+ ($N = 226$, prevalence $75.33\%$).

### 4.2 Leakage-Free Preprocessing and Cross-Validation Protocol

To avoid data leakage:
1. **Train-Only Parameter Fitting:** Scaling parameters (means, standard deviations) and categorical encodings were fitted strictly on training folds within `ClinicalTabularPreprocessor` and applied out-of-fold to validation sets.
2. **Missingness Indicator Flags:** Missing values in invasion parameters were assigned dedicated binary indicator columns (`missing_venous_invasion`, etc.) alongside median/mode imputation.
3. **Stratified 5-Fold Cross-Validation:** Folds preserved the $24.67\%$ positive class prevalence across all splits with strict patient isolation.

### 4.3 Unimodal Clinical Performance Results

| Model Architecture | Accuracy (Mean ± SD) | Balanced Accuracy (Mean ± SD) | Macro F1 (Mean ± SD) | AUROC (Mean ± SD) | Pooled AUROC | Brier Score (Mean ± SD) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (L2)** | $0.4400 \pm 0.0346$ | $0.4274 \pm 0.0666$ | $0.2576 \pm 0.0806$ | $0.4212 \pm 0.0710$ | $0.4194$ | $0.2763 \pm 0.0239$ |
| **Random Forest (100 Trees)** | $0.6200 \pm 0.0938$ | $0.4930 \pm 0.0704$ | $0.2366 \pm 0.1125$ | $0.4895 \pm 0.1058$ | $0.4654$ | $0.2309 \pm 0.0197$ |

### 4.4 Scientific Interpretation

The unimodal clinical models demonstrate near-chance or sub-chance discriminative performance (Balanced Accuracy $\approx 0.43 - 0.49$, AUROC $\approx 0.42 - 0.49$). 
- **Biological Conclusion:** Standard clinicopathological variables (age, sex, TNM stage, histological subtype, and microvascular invasion) do **not** reliably predict molecular HER-2 receptor overexpression.
- **Translational Relevance:** HER-2 amplification represents a distinct molecular alteration occurring independently of macroscopic TNM stage or depth of invasion. This highlights the theoretical necessity for multimodal models: histopathological morphology or genomic assays capture cellular phenomena that macro-clinical records alone fail to resolve.

---

## 5. In-Silico Molecular Regulatory Network Audit (`Gastric.xlsx`)

### 5.1 Workbook Provenance and Content Disclosures

The molecular workbook `data/support/Gastric.xlsx` contains a single sheet (`Table s2`) comprising 3,711 interaction records.

> [!CAUTION]
> **Scientific Disclosure on Molecular Predictions:**
> All 3,711 interaction rows in `Gastric.xlsx` represent **in-silico computational target predictions**, generated by bioinformatics algorithmic toolkits for a single human microRNA:
> - **MicroRNA Symbol:** `hsa-miR-497-5p`
> - **miRBase Accession:** `MIMAT0002820`
> 
> This workbook contains **zero** wet-lab patient expression measurements, zero RT-qPCR Cq values, zero RNA-Seq counts, and zero patient IDs. High prediction scores indicate algorithmic consensus, **not** verified clinical biomarkers in gastric cancer patients.

### 5.2 Algorithmic Consensus Distribution

The workbook aggregates predictions across eight bioinformatics databases and algorithms:
1. DIANA-microT
2. ElMMo
3. MicroCosm
4. miRanda
5. miRDB
6. PicTar
7. PITA
8. TargetScan

Across the 3,497 unique predicted target gene symbols, the consensus count (number of predicting algorithms agreeing on a target) exhibits a steep power-law decay:

| Algorithmic Consensus Count (Databases Agreeing) | Number of Target Genes | Percentage of Predicted Targets |
| :---: | :---: | :---: |
| **1 database** | 2,149 | 61.45% |
| **2 databases** | 693 | 19.82% |
| **3 databases** | 399 | 11.41% |
| **4 databases** | 219 | 6.26% |
| **5 databases** | 158 | 4.52% |
| **6 databases** | 82 | 2.35% |
| **7 databases** | 10 | 0.29% |
| **8 databases (Universal Consensus)** | 1 | 0.03% |

```
Consensus Decay Curve:
Consensus 1: [████████████████████████████████████████] 2,149 genes
Consensus 2: [█████████████                           ]   693 genes
Consensus 3: [███████                                 ]   399 genes
Consensus 4: [████                                    ]   219 genes
Consensus 5: [███                                     ]   158 genes
Consensus 6: [█                                       ]    82 genes
Consensus 7: [                                        ]    10 genes
Consensus 8: [                                        ]     1 gene
```

### 5.3 High-Consensus Target Genes and Gastric Cancer Biology

> [!NOTE]
> **Dated Addendum & Erratum (November 2026 — Final Release Integrity Audit Closeout):**
>
> In the initial draft of Table 5.3 below, the consensus counts for **`SEMA6D`** and **`CCNE1`** were accidentally transposed (listing `CCNE1` with 8 and `SEMA6D` with 7). Re-inspection and programmatic audit of the authoritative workbook `data/support/Gastric.xlsx` (`Table s2`) confirms:
>
> 1. **Unanimous 8/8 Consensus Gene:** **`SEMA6D`** (Row 0, Entrez: 80031, Ensembl: `ENSG00000137872`) is the **sole gene** in the entire 3,711-row dataset that achieved consensus across all 8 prediction tools (`diana_microt: 1`, `elmmo: 4`, `microcosm: 1`, `miranda: 2`, `mirdb: 1`, `pictar: 1`, `pita: 1`, `targetscan: 1`).
> 2. **7/8 Consensus Genes (10 Genes):** **`CCNE1`** (Row 2, Entrez: 898, Ensembl: `ENSG00000105173`) was predicted by 7 of the 8 tools (`diana_microt: 2`, `elmmo: 4`, `miranda: 2`, `mirdb: 2`, `pictar: 1`, `pita: 2`, `targetscan: 2`), but scored 0 in `microcosm`. All 10 genes at consensus 7/8 (`AK4`, `CCNE1`, `CD2AP`, `CDK17`, `FBXW7`, `PAFAH1B1`, `PLAG1`, `PRDM4`, `USP15`, `ZBTB34`) share this exact pattern (positive in 7 algorithms, 0 in MicroCosm).
> 3. **Cumulative High Consensus ($\ge 7/8$):** Exactly **11 unique target genes** achieve consensus $\ge 7/8$.
> 4. **Scientific Invariant:** These interactions are strictly **in-silico computational predictions**, not experimentally verified biomarkers. Zero wet-lab assays or patient-level tissue validations exist in this workbook.
>
> The corrected consensus values are shown in the verified table below.

Only 11 target genes achieved high consensus ($\ge 7$ predicting algorithms):

| Target Gene Symbol | Entrez ID | Ensembl ID | Consensus Count | Known Oncological / Gastric Cancer Pathway |
| :--- | :---: | :--- | :---: | :--- |
| **SEMA6D** | 80031 | `ENSG00000137872` | **8** (Unanimous) | **Semaphorin 6D:** Critical regulator of cell motility, axonal guidance, and tumor angiogenesis. |
| **CCNE1** | 898 | `ENSG00000105173` | 7 | **Cyclin E1:** Critical regulator of G1/S phase transition. Amplified in chromosomal instability (CIN) subtype gastric cancer; associated with resistance to trastuzumab. |
| **AK4** | 205 | `ENSG00000162433` | 7 | **Adenylate Kinase 4:** Controls mitochondrial energy homeostasis and metabolic reprogramming. |
| **CD2AP** | 23607 | `ENSG00000198087` | 7 | **CD2 Associated Protein:** Actin cytoskeleton remodeling and receptor endocytosis. |
| **CDK17** | 5128 | `ENSG00000059758` | 7 | **Cyclin Dependent Kinase 17:** Cell division cycle kinase and transcriptional regulator. |
| **FBXW7** | 55294 | `ENSG00000109670` | 7 | **F-Box and WD Repeat Domain 7:** Tumor suppressor ubiquitin ligase promoting degradation of MYC, Cyclin E, and NOTCH. Frequently mutated in gastric cancer. |
| **PAFAH1B1** | 5048 | `ENSG00000007168` | 7 | **Platelet Activating Factor Acetylhydrolase 1b Regulatory Subunit 1 (LIS1):** Dynein-mediated motility and mitotic spindle orientation. |
| **PLAG1** | 5324 | `ENSG00000181690` | 7 | **PLAG1 Zinc Finger:** Oncogenic transcription factor activating growth factor signaling pathways. |
| **PRDM4** | 11108 | `ENSG00000110851` | 7 | **PR/SET Domain 4:** Epigenetic chromatin remodeler and cell cycle modulator. |
| **USP15** | 9958 | `ENSG00000135655` | 7 | **Ubiquitin Specific Peptidase 15:** Deubiquitinating enzyme regulating TGF-$\beta$ signaling and genome stability. |
| **ZBTB34** | 403341| `ENSG00000177125` | 7 | **Zinc Finger And BTB Domain Containing 34:** Transcriptional repression. |

### 5.4 Molecular Regulatory Visualizations

The molecular audit generated two publication figures in `research/multimodal/visualizations/`:

1. **Algorithm Agreement and Consensus Distribution:**  
   [miRNA Prediction Database Distribution](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/multimodal/visualizations/mirna_prediction_database_distribution.png)  
   Bar chart illustrating positive prediction rates across the 8 bioinformatics databases and the exponential consensus decay across human target genes.

2. **Molecular Target Regulatory Network:**  
   [miRNA Target Network Graph](file:///c:/Users/KIIT0001/OneDrive/Desktop/gastric-cancer-histopathology-dashboard-main/research/multimodal/visualizations/mirna_target_network_graph.png)  
   NetworkX circular-layout graph rendering `hsa-miR-497-5p` as the central hub connected to top consensus target genes, color-coded by algorithmic consensus strength.

---

## 6. Multimodal Fusion Engine and Gatekeeper Architecture

### 6.1 Neural Architectures Implemented in `research/multimodal/fusion.py`

To ensure software readiness for future linked datasets, full PyTorch multimodal architectures were engineered:

1. **Clinical Tabular MLP Encoder (`ClinicalMLPEncoder`):**
   - Maps $D_{\text{clinical}}$ tabular features through linear layers, batch normalization, LeakyReLU activation ($\alpha = 0.1$), and dropout ($p = 0.3$) to produce a dense 128-dimensional clinical representation embedding.
2. **Early Fusion Classifier (`EarlyFusionClassifier`):**
   - Concatenates the 128-D clinical embedding with the image feature representation ($D_{\text{image}} = 768$ for Phikon / DINOv2-S, or $D_{\text{image}} = 1024$ for DINOv2-L), followed by a joint multilayer projection head producing task logits.
3. **Late Fusion Classifier (`LateFusionClassifier`):**
   - Evaluates unimodal image logits and clinical tabular logits through a cross-modal gating network, dynamically weighting each modality's decision confidence.

```
Early Fusion Architecture:
[ Image Crop (224x224) ] ----> [ Phikon ViT-B/16 ] ---------> (768-D Image Embedding)    \
                                                                                          ==> [ Concat: 896-D ] ---> [ Fusion Head ] ---> Logits
[ Clinical Features (15) ] --> [ ClinicalMLPEncoder ] -------> (128-D Tabular Embedding) /
```

### 6.2 The Gatekeeper Mechanism (`MultimodalFusionGatekeeper`)

To uphold scientific validity, the `MultimodalFusionGatekeeper` asserts six strict structural prerequisites prior to initiating multimodal training or inference:
1. `crosswalk_manifest_present`: An explicit crosswalk file must exist on disk.
2. `patient_identifiers_aligned`: Both modalities must share a common patient identifier schema.
3. `positive_linkage_count`: The number of matched image-patient pairs must exceed zero.
4. `no_heuristic_sorting_used`: Linkage must not rely on row indices or arbitrary order.
5. `temporal_specimen_validity`: Histological slides must correspond temporally to the clinical biopsy episode.
6. `no_synthetic_data_in_production`: Synthetic linkages must be strictly confined to isolated unit test fixtures.

If any check fails, the gatekeeper raises `MultimodalLinkageBlockedError`, aborting execution with an explanatory scientific audit trail.

---

## 7. Survival Analysis Blocker Audit and Conditional Pipeline

### 7.1 Exhaustive Variable Inspection

The repository was systematically audited for longitudinal outcome parameters required for time-to-event survival analysis (Kaplan-Meier survival curves, Log-Rank tests, Cox Proportional Hazards regression):

| Survival Analysis Requirement | Expected Field Example | Observed in HMU-GC-Clinical.csv | Blocker Impact |
| :--- | :--- | :---: | :--- |
| **Time-to-Event Duration** | `OS_months`, `DFS_days`, `time_to_recurrence` | **ABSENT** | Fatal Blocker |
| **Event Indicator / Vital Status** | `vital_status` (0=Alive, 1=Dead), `recurrence` | **ABSENT** | Fatal Blocker |
| **Observation Origin Date** | `date_of_surgery`, `date_of_diagnosis` | **ABSENT** | Fatal Blocker |
| **Last Follow-up Date** | `date_of_last_followup`, `date_of_death` | **ABSENT** | Fatal Blocker |

### 7.2 The Survival Blocker Protocol

The `SurvivalEligibilityAuditor` verified that `HMU-GC-Clinical.csv` contains exclusively cross-sectional pathological staging and baseline biopsy variables.
- **Enforcement:** The `ConditionalSurvivalModelRunner` evaluates `SurvivalEligibilityAuditor.is_eligible()`. Because all four endpoints are missing, the runner raises `SurvivalAnalysisBlockedError`.
- **Integrity Guarantee:** Synthetic survival durations (e.g., generating pseudo-survival times from exponential distributions) were strictly prohibited in production pipelines.

---

## 8. Requirements for Future Multimodal Unblocking

To transition the GastricPath AI multimodal and survival modules from their current software-ready state to active clinical deployment, external data contributors must provide:

### 8.1 Required Unblocking Specifications

1. **Institutional Patient-Tile Crosswalk Manifest:**
   - Format: `manifest_crosswalk.csv`
   - Schema:
     - `tile_filename` (string): Exact filename of GCHTID tile (e.g., `TUM-124.png`).
     - `wsi_accession_id` (string): Institutional whole-slide accession identifier.
     - `patient_id` (integer): Corresponding admission ID in `HMU-GC-Clinical.csv` (e.g., `1600097`).
     - `biopsy_date` (ISO-8601 string): Date of surgical resection/biopsy.

2. **Patient-Matched Molecular Expression Profiling:**
   - Format: `clinical_mirna_expression.csv`
   - Schema:
     - `patient_id` (integer): Aligned with HMU clinical cohort.
     - `mirna_id` (string): `hsa-miR-497-5p`.
     - `expression_quantification` (float): RT-qPCR $\Delta\Delta C_T$ or RNA-Seq normalized counts (e.g., TPM / FPKM).

3. **Longitudinal Follow-up Outcome Registry:**
   - Format: `clinical_longitudinal_outcomes.csv`
   - Schema:
     - `patient_id` (integer): Aligned with HMU clinical cohort.
     - `overall_survival_months` (float): Follow-up time from surgical resection to event or censoring.
     - `vital_status` (integer): 0 = Alive (Censored), 1 = Deceased (Event).
     - `disease_free_survival_months` (float): Time to local recurrence or distant metastasis.
     - `recurrence_status` (integer): 0 = Disease-free, 1 = Recurred.

---

## 9. Baseline Checkpoint Integrity Verification

In strict compliance with governance rules, the baseline historical classifier checkpoint was verified:
- **File Path:** `models/checkpoints/gastric_rnn_classifier.pt`
- **Expected SHA-256:** `65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8`
- **Observed SHA-256:** `65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8`
- **Integrity Status:** **VALIDATED & UNMODIFIED**

---

## 10. Summary of Deliverables and File Inventory

All artifacts, modules, pipelines, and tests for Original Phase 10 have been authored and verified:

```
research/multimodal/
├── __init__.py                     # Package initialization
├── config.json                     # Declarative configuration of paths, schemas, and blockers
├── linkage_audit.py                # Cross-dataset provenance and linkage auditor
├── clinical_schema.py              # Clinical data dictionary and schema cleaner
├── clinical_preprocessing.py       # Leakage-free preprocessor and patient-level splitting
├── clinical_analysis.py            # Cohort statistics, association tests, and demographic plotting
├── clinical_models.py              # Tabular MLP encoder, Logistic Regression, 5-Fold CV evaluator
├── biomarker_analysis.py           # In-silico workbook auditor for Gastric.xlsx
├── molecular_network.py            # NetworkX graph engine and query API
├── fusion.py                       # Early/Late fusion architectures and gatekeeper
├── survival_analysis.py            # Survival eligibility auditor and conditional runner
├── evaluation.py                   # Master pipeline driver
├── linkage_audit_summary.json      # Provenance audit summary artifact
├── clinical_schema_summary.json    # Clinical schema summary artifact
├── biomarker_audit_summary.json    # In-silico miRNA audit summary artifact
├── clinical_modeling_summary.json  # 5-fold CV unimodal modeling summary artifact
└── visualizations/
    ├── clinical_missingness_heatmap.png            # Missingness pattern visualization
    ├── clinical_cohort_demographics.png            # Demographic & staging multi-panel plot
    ├── mirna_prediction_database_distribution.png  # Database agreement & consensus decay
    └── mirna_target_network_graph.png              # Molecular regulatory network graph

tests/
└── test_original_phase10.py        # 19 comprehensive automated unit and integration tests

docs/
├── PHASE10_MULTIMODAL_CLINICAL_RESEARCH_REPORT.md  # Authoritative research report (this file)
└── IMPLEMENTATION_STATUS.md                        # Updated status tracker
```

