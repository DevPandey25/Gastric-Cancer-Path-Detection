# GASTRICPATH AI — AUTHORITATIVE MEDICAL EVIDENCE SOURCE REGISTRY

**Document Identifier:** `docs/MEDICAL_EVIDENCE_SOURCE_REGISTRY.md`  
**Module:** Evidence Retrieval and Clinical Reference System  
**Date of Audit & Literature Cutoff:** November 12, 2026  
**Status:** Authoritative Synthesis  

---

## 1. Evidence Hierarchy & Inclusion Criteria

GastricPath AI adheres to a strict 7-tier evidence hierarchy for clinical decision context and research documentation:

1. **Tier 1A: International Clinical Practice Guidelines** (ESMO, ASCO, NCCN, KGCA)
2. **Tier 1B: Official Regulatory Approvals** (FDA, EMA, CDSCO, PMDA, MFDS)
3. **Tier 1C: International Pathology Standards** (WHO Blue Books, CAP Protocols, ICCR Datasets)
4. **Tier 2A: High-Quality Randomized Controlled Clinical Trials** (Phase III RCTs: CheckMate-649, KEYNOTE-859, SPOTLIGHT, GLOW, ToGA, DESTINY-Gastric01/02)
5. **Tier 2B: Systematic Reviews and Meta-Analyses** (PRISMA-compliant reviews)
6. **Tier 3: Peer-Reviewed Observational Cohorts & Benchmarks** (HMU-GC-HE-30K, GasHisSDB, TCGA-STAD)
7. **Tier 4: Non-Peer-Reviewed Preprints** (Must be explicitly labelled as uncertified preprints)

---

## 2. Master Clinical Practice Guidelines (Cutoff: November 2026)

### 2.1 ESMO Clinical Practice Guideline — Gastric and Oesophageal Cancer (2026 Update)
- **Publication Date:** 15 October 2026
- **Lead Authors / Organization:** European Society for Medical Oncology (ESMO)
- **Citation:** *Annals of Oncology*, 2026. DOI: `10.1016/j.annonc.2026.08.002`
- **Core Clinical Recommendations:**
  - Perioperative chemotherapy with FLOT (fluorouracil, leucovorin, oxaliplatin, docetaxel) for 4 pre-operative and 4 post-operative cycles is standard for resectable stage $\ge$ cT2 and/or cN+ disease.
  - Mandatory reflex biomarker testing at initial diagnosis of advanced/metastatic disease: HER2, PD-L1 (CPS), dMMR/MSI, and CLDN18.2.
  - Zolbetuximab plus mFOLFOX6 is recommended as first-line standard of care for CLDN18.2-positive ($\ge 75\%$ of tumor cells with $\ge 2+$ membranous staining), HER2-negative metastatic adenocarcinoma.
  - Nivolumab plus fluoropyrimidine/platinum is recommended in HER2-negative, PD-L1 CPS $\ge 5$ advanced adenocarcinoma.

### 2.2 ASCO Living Guideline — Immunotherapy & Targeted Therapy (2026)
- **Publication Date:** June 2026
- **Organization:** American Society of Clinical Oncology (ASCO)
- **Citation:** *Journal of Clinical Oncology*, 2026. DOI: `10.1200/JCO.2026.44.1102`
- **Core Clinical Recommendations:**
  - Trastuzumab deruxtecan (T-DXd) endorsed as second-line or third-line therapy for HER2-positive advanced disease progressing on trastuzumab.
  - Pembrolizumab plus chemotherapy for first-line advanced disease in tumors with PD-L1 CPS $\ge 1$ (with greatest statistical benefit at CPS $\ge 10$).

### 2.3 NCI Gastric Cancer Treatment (PDQ) — Health Professional Version
- **Organization:** National Cancer Institute (NIH), United States
- **Version:** Autumn 2026 Revision
- **Core Content:** Full surgical oncology guidelines, D2 lymphadenectomy standards, and palliative care guidelines.

### 2.4 Pan-Asian Adapted ESMO Guidelines & Korean Practice Guidelines (KGCA 2024/2025)
- **Citations:** *Annals of Oncology* 2024 (PMC10937212) and *Journal of Gastric Cancer* 2025 (DOI: `10.5230/jgc.2025.25.e11`).
- **Core Focus:** Endoscopic resection (ESD) criteria for early gastric cancer (T1a), routine D2 lymph node dissection in high-volume centers, and adjuvant S-1 / CAPOX regimens.

---

## 3. Pathology and Diagnostic Standards

### 3.1 WHO Classification of Tumours, 6th Edition (Digestive System Tumours, 2026)
- **Histological Types:** Tubular adenocarcinoma, Papillary adenocarcinoma, Mucinous adenocarcinoma, Poorly cohesive carcinoma (including signet ring cell carcinoma), and Mixed adenocarcinoma.
- **Molecular Subtypes (TCGA / Asian Cancer Research Group):**
  1. EBV-positive (Epstein-Barr Virus)
  2. MSI-High (Microsatellite Instability-High)
  3. Genomically Stable (GS, diffuse histology, CDH1/RHOA mutations)
  4. Chromosomal Instability (CIN, intestinal histology, RTK/RAS amplifications)

### 3.2 CAP Stomach Carcinoma Protocol (Version 4.4.0.0 REL CAPCP, 2026)
- Mandatory synoptic reporting elements:
  - Procedure (Biopsy, Endoscopic Resection, Partial Gastrectomy, Total Gastrectomy)
  - Anatomical site (Cardia, Fundus, Body, Antrum, Pylorus, Greater/Lesser Curvature)
  - Histological type and Lauren classification (Intestinal, Diffuse, Mixed)
  - Histological grade (G1 Well, G2 Moderate, G3 Poor, Undifferentiated)
  - Depth of invasion (pT) and Gastric wall layer involvement
  - Surgical margins (Proximal, Distal, Omental / Adventitial / Radial)
  - Lymphovascular invasion (LVI) and Perineural invasion (PNI)
  - Lymph nodes: Number examined ($\ge 16$ required for adequate staging), number involved, extranodal extension
  - Post-neoadjuvant treatment effect: Tumor Regression Grade (Ryan / CAP score 0–3)

---

## 4. Biomarker Companion Diagnostic Criteria

```
+---------------------------------------------------------------------------------------------------------+
| MASTER BIOMARKER COMPANION DIAGNOSTIC MATRIX                                                             |
+-----------+----------------------+---------------------------+------------------------------------------+
| Biomarker | Testing Modality     | Positive Definition       | Companion Therapeutic & Evidence         |
+-----------+----------------------+---------------------------+------------------------------------------+
| HER2      | IHC + ISH (Hofmann)  | IHC 3+ OR IHC 2+/ISH+     | Trastuzumab (ToGA) / T-DXd (DESTINY-GC)  |
|           |                      | (ratio >= 2.0)            | Level 1A (ESMO / ASCO)                   |
+-----------+----------------------+---------------------------+------------------------------------------+
| PD-L1     | IHC (28-8 / 22C3)    | CPS >= 5 (Nivolumab ESMO) | Nivolumab (CheckMate-649) /              |
|           |                      | CPS >= 1 / 10 (Pembro)    | Pembrolizumab (KEYNOTE-859) Level 1A     |
+-----------+----------------------+---------------------------+------------------------------------------+
| MMR / MSI | IHC (4-panel) or PCR | dMMR (loss >= 1 protein)  | Pembrolizumab / Nivolumab monotherapy    |
|           |                      | MSI-H (>= 2/5 markers)    | Resistant to fluoropyrimidine monotherapy|
+-----------+----------------------+---------------------------+------------------------------------------+
| CLDN18.2  | IHC (43-14A assay)   | >= 75% tumor cells with   | Zolbetuximab + mFOLFOX6 / CAPOX          |
|           |                      | >= 2+ membranous staining | (SPOTLIGHT & GLOW trials) Level 1A       |
+-----------+----------------------+---------------------------+------------------------------------------+
```

---

## 5. Regional Jurisdiction Matrix (India, US, EU, Japan, South Korea)

| Jurisdiction | Regulatory Body | Approved 1st-Line Immunotherapy | Zolbetuximab Status | Key Clinical Considerations |
|:---|:---|:---|:---|:---|
| **India** | CDSCO | Nivolumab, Pembrolizumab | Under Regulatory Review (Compassionate Access) | Cost and access constraints; generic Trastuzumab biosimilars available; ICMR guidelines emphasize D2 surgery. |
| **United States** | FDA | Nivolumab + chemo, Pembrolizumab + chemo | FDA Approved (Vyloy, 2024) | Broad labels; companion diagnostics FDA-approved (VENTANA CLDN18, Dako 28-8). |
| **European Union** | EMA | Nivolumab (CPS $\ge 5$), Pembrolizumab (CPS $\ge 1$) | EMA Approved (2024/2025) | Reimbursement tied to strict CPS cutoffs; FLOT perioperative chemotherapy standard. |
| **Japan** | PMDA | Nivolumab + chemo (no CPS cutoff) | PMDA Approved (March 2024) | Routine D2 gastrectomy; S-1 adjuvant standard. First global approval for Zolbetuximab. |
| **South Korea** | MFDS | Nivolumab, Pembrolizumab | Approved | National screening program; minimally invasive gastrectomy prevalent. |

---

## 6. Curative Intent vs. Cure Disclaimer

In accordance with international oncology standards, GastricPath AI maintains an absolute distinction:
- **Curative-Intent Treatment:** A multimodal clinical strategy (e.g., radical D2 gastrectomy + perioperative FLOT) delivered with the goal of eliminating detectable disease.
- **Cure:** A long-term biological state that cannot be guaranteed or determined at initial diagnosis or by an artificial intelligence model.
- **Reporting Rule:** The system outputs "Documented Treatment Intent" and never produces automated statements of individual cure or survival probability.
