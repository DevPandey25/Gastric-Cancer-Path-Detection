# GASTRICPATH AI — CLINICAL RESEARCH & PDF REPORTING ARCHITECTURE

**Document Identifier:** `docs/CLINICAL_REPORTING_ARCHITECTURE.md`  
**Module:** Gastric Cancer Clinical Research & Reporting Workbench  
**Status:** Authoritative Architectural Specification  
**Date:** November 2026  

---

## 1. System Vision & Purpose

The **Gastric Cancer Clinical Research & Reporting Workbench** extends GastricPath AI from an algorithmic experimentation workspace into an evidence-grounded research documentation and clinical information assistance platform.

The system bridges five foundational capabilities:
1. **Histopathology Image Documentation & Analysis:** Integrates real microscopic images, verified CNN-RNN sequence outputs, foundation model embeddings (Phikon, DINOv2), and attribution saliency maps (Grad-CAM).
2. **Structured Clinical Information Management:** Versioned, Pydantic-validated clinical data schemas capturing patient background, symptoms, endoscopy, surgical histopathology, molecular biomarkers, radiology, staging, and treatment history.
3. **Evidence-Based Literature & Guideline Retrieval:** Direct linkage to international clinical practice guidelines (ESMO 2026, ASCO 2026, WHO 6th Edition, CAP Protocols, ICCR) and a verified 32-publication scientific literature database.
4. **AI-Assisted Research Interpretation:** Localized synthesis tools operating strictly under a local-only privacy barrier—zero protected health information or raw images leave the host machine.
5. **Professional Multi-Mode PDF Generation:** Publication-grade document generation engine rendering high-resolution image panels, structured tables, running headers/footers, dynamic page numbers, and numbered Vancouver citations.

```
+----------------------------------------------------------------------------------------------------+
|                               GASTRICPATH AI REPORTING ARCHITECTURE                                |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|   +-----------------------+     +------------------------+     +-------------------------------+   |
|   | 1. Histopathology     |     | 2. Structured Clinical |     | 3. Evidence & Literature      |   |
|   | Image Evidence        |     | Data Schema (Pydantic) |     | Registry (ESMO, ASCO, WHO)    |   |
|   +-----------+-----------+     +-----------+------------+     +---------------+---------------+   |
|               |                             |                                  |                   |
|               +----------------------+      |      +---------------------------+                   |
|                                      |      |      |                                               |
|                                      v      v      v                                               |
|                          +------------------------------------+                                    |
|                          |    REPORT VALIDATION & SAFETY      |                                    |
|                          |    ENGINE (Audit Invariants)       |                                    |
|                          +-----------------+------------------+                                    |
|                                            |                                                       |
|                                            v                                                       |
|                          +------------------------------------+                                    |
|                          |  THREE REPORT GENERATORS           |                                    |
|                          |  - Mode A: Clinical Case Dossier   |                                    |
|                          |  - Mode B: Image Atlas             |                                    |
|                          |  - Mode C: Research Manuscript     |                                    |
|                          +-----------------+------------------+                                    |
|                                            |                                                       |
|                                            v                                                       |
|                          +------------------------------------+                                    |
|                          |   PROFESSIONAL PDF ENGINE          |                                    |
|                          |   (ReportLab Flowables & Canvas)   |                                    |
|                          +-----------------+------------------+                                    |
|                                            |                                                       |
|                                            v                                                       |
|                          +------------------------------------+                                    |
|                          |   PUBLICATION-GRADE PDF EXPORTS    |                                    |
|                          +------------------------------------+                                    |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Non-Negotiable Medical & Scientific Boundaries

The reporting architecture strictly adheres to established scientific boundaries:
1. **Clinical Information Assistance, Not Autonomous Diagnosis:** The platform is an educational and documentation aid. It does not replace a multidisciplinary tumor board (MDT) or clinical pathologist.
2. **Curative Intent vs. Cure:** The system never claims a patient is cured, guarantees treatment success, or calculates individual cure probabilities. It documents "treatment intent" (curative, disease-control, palliative) as recorded by treating physicians.
3. **Patch-Level Model Boundaries:** Model predictions derive from 224x224 pixel H&E patches. Patch-level TUM probabilities are never represented as patient-level cancer stage, total tumor burden, or survival probability.
4. **Data Realities:** The public HMU-GC-HE-30K dataset lacks whole-slide images, pixel masks, and patient crosswalk keys. The system explicitly discloses these limitations.
5. **Privacy Guard:** All patient data, case files, images, and generated PDFs remain 100% local. No identifiable records are transmitted to external services.

---

## 3. Three Distinct Document Modes

### Mode A: Gastric Cancer Clinical Case Review Dossier
- **Objective:** Organizes clinical records, pathology findings, molecular biomarkers, radiology, and staging for review by a multidisciplinary team.
- **Key Modules:**
  - Executive summary and case provenance.
  - Endoscopic and biopsy findings.
  - CAP-style synoptic pathology reporting.
  - High-resolution histopathology figure gallery.
  - Molecular biomarker panel: HER2, PD-L1 CPS, MMR/MSI, CLDN18.2.
  - Clinical and pathological staging (cTNM, pTNM, ypTNM).
  - Clinical Evidence and Guideline Alignment Matrix.
  - Multidisciplinary Team Review Questions & Missing Information Checklist.
  - Numbered Vancouver references to authoritative guidelines.

### Mode B: Histopathology Image Atlas
- **Objective:** Detailed, image-centered research document showcasing genuine tissue patches and their computational analysis.
- **Key Modules:**
  - Original 224x224 tissue patch with acquisition metadata.
  - Top-1 predicted class and 8-class probability distribution.
  - Certified baseline checkpoint identification (`gastric_rnn_classifier.pt`).
  - Side-by-side Grad-CAM saliency heatmaps with attribution limits disclosed.
  - Candidate cellular object segmentation overlays.
  - Feature-space foundation model embedding coordinates (Phikon, DINOv2).

### Mode C: Gastric Cancer Research Manuscript
- **Objective:** Publication-grade scientific manuscript reporting experimental results from project benchmarks.
- **Key Modules:**
  - Structured abstract, introduction, research questions.
  - Materials and methods with duplicate-group-separated benchmark protocol.
  - Baseline CNN-RNN sequence architecture description (38.56 ms CPU latency).
  - Foundation model representation benchmark (Phikon Mean-Patch vs. DINOv2 vs. CLS).
  - Causal attribution faithfulness evaluation (Deletion/Insertion AUC).
  - Spatial cell graph vs. tabular morphology MLP benchmark (reporting negative result).
  - Full CLAIM 2024 and TRIPOD+AI reporting compliance statements.

---

## 4. Software Component Layout

```
research/clinical_reporting/
    schemas/            # Pydantic data models for clinical cases, staging, biomarkers
    ingestion/          # Secure local document/image ingestion and de-identification
    evidence/           # Guideline registry, literature search, evidence matrix
    report_engine/      # Document composers for Modes A, B, and C
    figures/            # High-resolution figure composition and captioning
    pdf/                # ReportLab canvas, flowables, numbered canvas, styles
    security/           # Local privacy guard, path traversal protection, audit log
```

---

## 5. Security and Privacy Guarantees

- **De-Identification Engine:** Automatically redacts names, MRNs, phone numbers, addresses, and full dates of birth before rendering.
- **Local Isolation:** All file reads, database lookups, and PDF builds operate strictly within the local filesystem.
- **Allowlisted Path Routing:** Restricts artifact access to allowlisted directories, preventing directory traversal.
- **Deterministic Output:** PDF builds are reproducible given the same data snapshot.
