# GASTRICPATH AI — CLINICAL DATA DICTIONARY & INFORMATION SCHEMA

**Document Identifier:** `docs/CLINICAL_DATA_DICTIONARY.md`  
**Module:** Clinical Information Schema & Pydantic Data Models  
**Status:** Authoritative Standard  
**Date:** November 2026  

---

## 1. Information Philosophy & Missing Data Standard

GastricPath AI implements strict medical recording invariants:
- **No Implicit Normalization:** An undocumented clinical field is **never** converted into a normal or negative finding. A missing symptom is marked `[Not assessed]`, not `Absent`. An unperformed biomarker test is marked `[Not available]`, not `Negative`.
- **Status Enumeration:** Every `ClinicalField` carries an explicit `verification_status`:
  1. `Documented`: Verified in original signed clinical or laboratory report.
  2. `Clinician verified`: Explicitly certified by an attending physician/pathologist.
  3. `Patient reported`: Subjective patient testimony.
  4. `Imported from source document`: Ingested via verified electronic records.
  5. `Model generated`: Produced by an algorithmic model (research only).
  6. `Unknown`: Cannot be determined from available records.
  7. `Not available`: Record does not contain this information.
  8. `Not assessed`: Diagnostic test or examination was not performed.
  9. `Not applicable`: Field does not apply to this specimen or clinical stage.
  10. `Conflicting information`: Discordant reports between tests or providers.

---

## 2. Domain Data Elements

### 2.1 Patient & Case Metadata (`PatientCase`)
| Field Name | Type / Values | Description / Rules |
|:---|:---|:---|
| `case_id` | String | De-identified identifier (e.g., `GP-GC-2026-001`). Direct identifiers (MRN, SSN, Name, Address) prohibited. |
| `age` | Integer / Years | Patient age in years or age range. |
| `sex` | String | `Male` / `Female` / `Other` / `[Not available]`. |
| `jurisdiction` | Enum | `India`, `United States`, `European Union`, `Japan`, `South Korea`, `Global`. Directs guideline alignment. |
| `ecog_performance_status`| Integer (0–4) | Functional status (0: Fully active, 4: Completely bedridden). |
| `family_cancer_history` | String | Documented gastric or Lynch/HDGC hereditary syndromes. |

### 2.2 Endoscopic Findings (`EndoscopyFindings`)
| Field Name | Type / Units | Description / Rules |
|:---|:---|:---|
| `tumor_location` | String | `Cardia`, `Fundus`, `Body`, `Antrum`, `Pylorus`, `Gastroesophageal Junction (Siewert I/II/III)`. |
| `lesion_size` | Float / cm | Longest diameter measured by endoscopist (NOT inferred from patch crops). |
| `borrmann_morphology` | String | Type 1 (Polypoid), Type 2 (Ulcerated circumscribed), Type 3 (Ulcerated infiltrative), Type 4 (Diffuse linitis plastica). |
| `biopsy_count` | Integer / fragments | Number of biopsy pieces retrieved ($\ge 6\text{--}8$ recommended for biomarker accuracy). |

### 2.3 Surgical Histopathology (`HistopathologyReport`)
| Field Name | Classification Standard | Description / Rules |
|:---|:---|:---|
| `who_histological_type` | WHO 6th Edition (2026) | Tubular, Papillary, Mucinous, Poorly cohesive (including signet-ring), Mixed adenocarcinoma. |
| `lauren_classification` | Lauren Classification | `Intestinal`, `Diffuse`, `Mixed`. |
| `histological_grade` | CAP / WHO | `G1` (Well), `G2` (Moderate), `G3` (Poor), `Undifferentiated`. |
| `depth_of_invasion` | AJCC 8th Edition | T1a (Lamina propria), T1b (Submucosa), T2 (Muscularis propria), T3 (Subserosa), T4a (Serosa), T4b (Adjacent structures). |
| `lymphovascular_invasion` | Binary | Present / Absent / Cannot be determined. |
| `perineural_invasion` | Binary | Present / Absent / Cannot be determined. |
| `lymph_nodes_examined` | Integer | Total regional nodes retrieved ($\ge 16$ required for AJCC N-staging validity). |
| `treatment_effect_trg` | Ryan / CAP Score | Post-neoadjuvant Tumor Regression Grade (0: Complete response, 3: No response). |

### 2.4 Molecular Biomarker Companion Diagnostics (`MolecularBiomarkers`)
| Biomarker | Assay / Testing Method | Positive Criteria | Clinical Implication |
|:---|:---|:---|:---|
| **HER2** | IHC (4B5/HercepTest) + ISH | IHC 3+ OR IHC 2+/ISH+ (ratio $\ge 2.0$) | Eligibility for Trastuzumab (1st line) / T-DXd (2nd line). |
| **PD-L1** | IHC (28-8 / 22C3) CPS | CPS numerical value ($\ge 5$ ESMO; $\ge 1$ or $\ge 10$ ASCO/FDA) | Eligibility for Nivolumab / Pembrolizumab combination. |
| **MMR / MSI** | IHC (4-panel) / PCR / NGS | Loss of $\ge 1$ protein (dMMR) / MSI-High | Marked immunotherapy sensitivity; chemo-monotherapy resistance. |
| **CLDN18.2** | IHC (43-14A assay) | $\ge 75\%$ tumor cells with $\ge 2+$ membranous staining | Eligibility for Zolbetuximab + mFOLFOX6 / CAPOX. |

### 2.5 Staging Classification (`StagingClassification`)
- Separates `cTNM` (pre-treatment clinical), `pTNM` (pathological post-resection), and `ypTNM` (post-neoadjuvant).
- Strict Invariant: If T, N, or M are unavailable, the system outputs:  
  *"TNM stage cannot be established from the currently supplied information."*

### 2.6 Treatment History & Intent (`TreatmentHistory`)
- Mandated Field: `treatment_intent_documented`:
  - `Curative intent documented by treating team`
  - `Disease-control intent documented`
  - `Palliative intent documented`
  - `Not documented in supplied records`
  - `Requires oncology-team clarification`
- Prohibits automated claims that a patient is cured or will be cured.
