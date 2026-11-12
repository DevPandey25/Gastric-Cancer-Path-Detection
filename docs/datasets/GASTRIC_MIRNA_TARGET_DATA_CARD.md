# Dataset Card: Gastric Carcinoma miRNA Target Network (In-Silico)

## 1. Resource Summary
- **Resource Name:** In-Silico Gastric Adenocarcinoma miRNA Target Interaction Catalog
- **Authoritative Data Source:** `data/support/Gastric.xlsx` (Sheet: `Table s2`)
- **Evaluated microRNA:** Strictly **`hsa-miR-497-5p`** (miRBase Accession: `MIMAT0002820`)
- **Total Interaction Rows:** 3,711 interaction records
- **Unique Predicted Target Genes:** 3,497 unique gene symbols
- **Prediction Toolset:** 8 independent computational algorithms (`diana_microt`, `elmmo`, `microcosm`, `miranda`, `mirdb`, `pictar`, `pita`, `targetscan`)
- **High-Consensus Target Count ($\ge 7$ Algorithms):** **11 unique target genes**
- **Validation Level:** **In-silico computational predictions only — not wet-lab validated**. Zero wet-lab luciferase assays or qRT-PCR experiments were conducted in this release.

---

## 2. In-Silico Consensus Target Catalog ($\ge 7$ Algorithms)

When filtering for strict high-consensus targets predicted by at least 7 of the 8 bioinformatics engines, exactly **11 unique target genes** are identified:

| Target Gene Symbol | Official Gene Name | Entrez Gene ID | Ensembl Gene ID | Algorithm Consensus Count | Supporting Algorithms ($\ge 7 / 8$) |
|:---|:---|:---:|:---|:---:|:---|
| **SEMA6D** | Semaphorin 6D | 80031 | ENSG00000137872 | **8 / 8** (Unanimous) | DIANA, ElMMo, MicroCosm, miRanda, miRDB, PicTar, PITA, TargetScan |
| **AK4** | Adenylate Kinase 4 | 205 | ENSG00000162433 | **7 / 8** | DIANA, ElMMo, miRanda, miRDB, PicTar, PITA, TargetScan (Missing: MicroCosm) |
| **CCNE1** | Cyclin E1 | 898 | ENSG00000105173 | **7 / 8** | DIANA, ElMMo, miRanda, miRDB, PicTar, PITA, TargetScan (Missing: MicroCosm) |
| **CD2AP** | CD2 Associated Protein | 23607 | ENSG00000198087 | **7 / 8** | DIANA, ElMMo, miRanda, miRDB, PicTar, PITA, TargetScan (Missing: MicroCosm) |
| **CDK17** | Cyclin Dependent Kinase 17 | 5128 | ENSG00000059758 | **7 / 8** | DIANA, ElMMo, miRanda, miRDB, PicTar, PITA, TargetScan (Missing: MicroCosm) |
| **FBXW7** | F-Box and WD Repeat Domain Containing 7 | 55294 | ENSG00000109670 | **7 / 8** | DIANA, ElMMo, miRanda, miRDB, PicTar, PITA, TargetScan (Missing: MicroCosm) |
| **PAFAH1B1** | Platelet Activating Factor Acetylhydrolase 1b Regulatory Subunit 1 | 5048 | ENSG00000007168 | **7 / 8** | DIANA, ElMMo, miRanda, miRDB, PicTar, PITA, TargetScan (Missing: MicroCosm) |
| **PLAG1** | PLAG1 Zinc Finger | 5324 | ENSG00000181690 | **7 / 8** | DIANA, ElMMo, miRanda, miRDB, PicTar, PITA, TargetScan (Missing: MicroCosm) |
| **PRDM4** | PR / SET Domain 4 | 11108 | ENSG00000110851 | **7 / 8** | DIANA, ElMMo, miRanda, miRDB, PicTar, PITA, TargetScan (Missing: MicroCosm) |
| **USP15** | Ubiquitin Specific Peptidase 15 | 9958 | ENSG00000135655 | **7 / 8** | DIANA, ElMMo, miRanda, miRDB, PicTar, PITA, TargetScan (Missing: MicroCosm) |
| **ZBTB34** | Zinc Finger And BTB Domain Containing 34 | 403341 | ENSG00000177125 | **7 / 8** | DIANA, ElMMo, miRanda, miRDB, PicTar, PITA, TargetScan (Missing: MicroCosm) |

---

## 3. Consensus Distribution & Counting Unit Reconciliation

The distribution of algorithmic agreement across the 3,711 interaction rows in `data/support/Gastric.xlsx` is:
- **Consensus = 8 (Unanimous):** 1 gene (`SEMA6D`)
- **Consensus = 7:** 10 genes
- **Cumulative $\ge 7$ Consensus:** **11 unique genes** (authoritative high-consensus set)
- **Consensus = 6:** 82 genes (cumulative $\ge 6$: 93 genes)
- **Consensus = 5:** 158 genes (cumulative $\ge 5$: 251 genes)
- **Consensus = 4:** 219 genes (cumulative $\ge 4$: 470 genes)
- **Consensus = 3:** 399 genes (cumulative $\ge 3$: 869 genes)
- **Consensus = 2:** 693 genes (cumulative $\ge 2$: 1,562 genes)
- **Consensus = 1:** 2,149 genes (predicted by only 1 algorithm)

### Reconciliation with Historical Reports & Draft Counts
1. **The 11 vs. 18 Count Reconciliation:**
   - **11 Targets:** The true, verified count of high-consensus target genes predicted by **$\ge 7$ bioinformatics algorithms** for `hsa-miR-497-5p` in `data/support/Gastric.xlsx`.
   - **18 Pairs in Draft Card:** A preliminary draft table had listed 18 illustrative pairs spanning multi-pathway oncogenes (HER2, EGFR, VEGFA, MET, CD274, TP53). That draft table did not correspond to `Gastric.xlsx`, which evaluates `hsa-miR-497-5p` exclusively. The draft table has been formally retracted and replaced by the authoritative 11-gene catalog above.
2. **Computational In-Silico Restriction:**
   - High consensus across algorithmic tools indicates shared sequence complementarity and thermodynamic stability models, **not** biological silencing in human gastric adenocarcinoma tissue. All references throughout the application must remain designated as **`In-silico miRNA target ranking`**.

