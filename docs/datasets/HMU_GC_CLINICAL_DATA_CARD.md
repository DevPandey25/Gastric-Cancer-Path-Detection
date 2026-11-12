# Dataset Card: HMU-GC-Clinical Cohort

## 1. Cohort Summary
- **Dataset Name:** HMU-GC-Clinical Gastric Carcinoma Retrospective Cohort
- **Institution / Origin:** Harbin Medical University Affiliated Hospital Gastric Cancer Series
- **Total Patients Analyzed:** $N = 300$ clinical records (complete usable cohort)
- **Primary Clinical Focus:** Clinicopathological association, pathological staging, and exploratory biomarker target correlation
- **Data Governance & Privacy Protection:** **Individual identifiers withheld from application outputs**. The underlying raw source file (`data/support/HMU-GC-Clinical.csv`) contains 7-digit integer patient admission identifiers (in the 1600000 range), which are pseudonymous sensitive hospital numbers. These identifiers are strictly withheld and redacted from all application API responses, user interfaces, logs, figures, and public documentation.

---

## 2. Cohort Cleaning, Exclusion Accounting & Feature Schema

### Data Cleaning and Cohort Derivation Audit
- **Original Source Records:** 300 patient rows in `data/support/HMU-GC-Clinical.csv`.
- **Records Excluded:** **0 records excluded**. All 300 rows represent valid clinical entries with physiologically plausible age ($33\text{--}79$), biological sex, and AJCC staging.
- **Columns Cleaned:** 2 completely empty trailing spreadsheet columns (`Unnamed: 15`, `Unnamed: 16`) were dropped, retaining 15 clinically standardized variables.
- **Missing Value Handling:** Missing attributes (e.g. Lauren classification in 18.33%, venous invasion in 27.67%, lymphovascular invasion in 16.67%, perineural invasion in 17.33%, stroma quantity in 6.0%, and infiltration pattern in 6.67%) were **not handled by dropping rows**. Instead, preprocessing applies train-split median imputation alongside explicit missingness indicator binary flags, ensuring the complete $N=300$ cohort remains usable for modeling without attrition.

### Clinical Feature Inventory & Distribution ($N = 300$)

| Clinical Feature | Type | Valid Values / Distribution ($N=300$) | Missingness | Clinical Relevance |
|:---|:---|:---|:---:|:---|
| **Age** | Continuous | Range: 33 – 79 years (Mean: $60.77 \pm 9.09$, Median: 61.0) | 0% | Age at surgical resection |
| **Sex** | Binary Categorical | Male: 236 (78.7%), Female: 64 (21.3%) | 0% | Biological sex |
| **Pathological T Stage** | Ordinal | T1a: 6, T1b: 20, T2: 24, T3: 130, T4a: 111, T4b: 9 | 0% | Depth of primary tumor invasion |
| **Pathological N Stage** | Ordinal | N0: 97, N1: 52, N2: 60, N3a: 65, N3b: 26 | 0% | Regional lymph node metastasis |
| **Pathological M Stage** | Binary Categorical | M0: 284 (94.7%), M1: 16 (5.3%) | 0% | Distant metastasis |
| **Comprehensive Staging** | Ordinal | ⅠA: 20, ⅠB: 17, ⅡA: 23, ⅡB: 43, ⅢA: 64, ⅢB: 67, ⅢC: 50, Ⅳ: 16 | 0% | AJCC 8th anatomic stage grouping |
| **Histological Type** | Categorical Nominal | Tubular: 182, Poorly cohesive: 73, Mucinous: 31, Other: 14 | 0% | WHO histological classification |
| **Lauren Classification** | Categorical Ordinal | Intestinal: 140, Mixed: 32, Diffuse: 73 (Missing: 55) | 18.33% | Lauren gastric classification |
| **HER-2 IHC Status** | Exploratory Categorical | IHC 0: 140 (46.7%), IHC 1+: 86 (28.7%) [Total 0/1+: 226, 75.33%]<br>IHC 2+: 48 (16.0%), IHC 3+: 26 (8.67%) [Total 2+/3+: 74, 24.67%] | 0% | Exploratory IHC categorical salience |
| **Overall Survival (OS)** | Time-to-Event | Blocked / Incomplete follow-up | 100% (Unlinked) | Follow-up censored/unlinked from GCHTID tiles |

---

## 3. Critical Methodological Disclosures

### Exploratory HER-2 IHC Categorization
In this research software release, HER-2 categorization is explicitly designated as:
> **`IHC 2+/3+ exploratory category — not confirmed HER2-positive status`**

*Rationale:* In clinical diagnostics (ASCO/CAP guidelines), an IHC 2+ score is equivocal and requires dual-probe Fluorescence In Situ Hybridization (FISH) or Silver In Situ Hybridization (SISH) to assess *ERBB2* gene amplification before a patient is deemed HER-2 positive. Because confirmatory FISH gene-amplification assays were unlinked from this retrospective cohort, treating IHC 2+/3+ as a definitive clinical diagnosis is scientifically impermissible.

### Independent Cohort Provenance (Zero Tile Linkage)
HMU-GC-Clinical records are **not paired on a slide-by-slide or patient-by-patient basis** with GCHTID image tiles. Any multimodal scoring or cross-tabulation in GastricPath AI constitutes a simulated exploratory risk fusion for methodology demonstration, not a validated single-patient diagnostic score.

---

## 3. Critical Methodological Disclosures

### Exploratory HER-2 IHC Categorization
In this research software release, HER-2 categorization is explicitly designated as:
> **`IHC 2+/3+ exploratory category — not confirmed HER2-positive status`**

*Rationale:* In clinical diagnostics (ASCO/CAP guidelines), an IHC 2+ score is equivocal and requires dual-probe Fluorescence In Situ Hybridization (FISH) or Silver In Situ Hybridization (SISH) to assess *ERBB2* gene amplification before a patient is deemed HER-2 positive. Because confirmatory FISH gene-amplification assays were unlinked from this retrospective cohort, treating IHC 2+/3+ as a definitive clinical diagnosis is scientifically impermissible.

### Independent Cohort Provenance (Zero Tile Linkage)
HMU-GC-Clinical records are **not paired on a slide-by-slide or patient-by-patient basis** with GCHTID image tiles. Any multimodal scoring or cross-tabulation in GastricPath AI constitutes a simulated exploratory risk fusion for methodology demonstration, not a validated single-patient diagnostic score.

