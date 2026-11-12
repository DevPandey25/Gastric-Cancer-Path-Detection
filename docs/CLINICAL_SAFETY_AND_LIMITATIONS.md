# GASTRICPATH AI — CLINICAL SAFETY, ETHICAL BOUNDARIES & LIMITATIONS

**Document Identifier:** `docs/CLINICAL_SAFETY_AND_LIMITATIONS.md`  
**Module:** Clinical Reporting Governance & Safety Framework  
**Status:** Mandatory Compliance Policy  
**Date:** November 2026  

---

## 1. Core Medical Disclaimer

> [!CAUTION]
> **RESEARCH ASSISTANCE PLATFORM — NOT AN AUTONOMOUS MEDICAL DEVICE**  
> GastricPath AI is a clinical research, documentation, and medical information assistance software platform. It is **not** an FDA-cleared, CE-marked, or CDSCO-approved medical diagnostic device. It does not replace the expert judgment of a qualified surgical oncologist, medical oncologist, gastroenterologist, or gastrointestinal pathologist.
>
> **The platform must never:**
> 1. Independently diagnose a patient with cancer.
> 2. Calculate or state that a patient is "cured" or will be cured.
> 3. Guarantee any percentage probability of survival or cure.
> 4. Prescribe medication or calculate chemotherapy dosages.
> 5. Direct whether a patient should or should not undergo surgery.

---

## 2. Mandatory Distinction: Curative Intent vs. Cure

In surgical and medical oncology, clear terminology is vital:
- **Curative-Intent Therapy:** Medical interventions (e.g., radical D2 gastrectomy and perioperative FLOT chemotherapy) administered with the goal of eliminating all detectable microscopic and macroscopic tumor burden.
- **Cure:** A retrospective biological state denoting long-term complete remission without disease recurrence. It cannot be predicted with certainty at baseline diagnosis.
- **System Invariant:** GastricPath AI explicitly documents the *"Treatment Intent Discussed in Source Records"* (Curative, Disease-Control, Palliative, or Not Documented). The software is hardcoded to **block** automated generation of statements claiming "This patient is cured" or "Treatment guarantees cure."

---

## 3. Patch-Level Model Boundaries vs. Clinical Pathology

1. **Input Scale:** The deep learning model (`GastricCancerRNN`) evaluates localized $224 \times 224$ pixel histological crops.
2. **Probability Interpretation:** A patch-level `TUM` (tumor) class probability of 92% means that within that isolated $224 \times 224$ patch, the model identifies features characteristic of malignant adenocarcinoma epithelium. It does **not** signify:
   - That 92% of the patient's stomach is involved by cancer.
   - That the patient has a 92% probability of malignancy overall.
   - That the cancer has metastasized or is advanced.
3. **No Automated Staging:** The deep learning model cannot infer T, N, or M categories from image tiles alone. TNM staging requires complete clinical, radiological, and surgical pathology documentation.

---

## 4. Regional Treatment & Regulatory Jurisdiction Disclaimers

Oncology drug approvals and reimbursement policies vary by jurisdiction:
- **India:** CDSCO regulatory status governs drug availability. While Trastuzumab biosimilars and standard immunotherapies (Nivolumab, Pembrolizumab) are available, novel agents such as Zolbetuximab may be accessible only through named-patient import or ongoing compassionate access programs.
- **United States:** FDA companion diagnostic criteria (e.g., VENTANA CLDN18 43-14A assay for Zolbetuximab, Dako 28-8 for Nivolumab) govern labeled indications.
- **European Union:** EMA approvals frequently require specific biomarker thresholds (e.g., Nivolumab reimbursed strictly in PD-L1 CPS $\ge 5$ advanced GC).
- **Japan & South Korea:** PMDA and MFDS standards reflect high endemic incidence, routine national endoscopic screening programs, routine D2 surgical standards, and regional chemotherapy regimens (such as S-1).

---

## 5. Local-Only Patient Privacy Policy

- **Zero Remote PHI Transmission:** All clinical dossiers, uploaded pathology images, and generated PDF reports are stored locally within the user's secure project environment.
- **No Identifiers in Prompts:** Even when research synthesis tools are invoked, zero direct patient identifiers (names, MRNs, phone numbers, complete dates of birth) are transmitted over external networks.
- **Audit Trails:** All access and compilation events are logged locally. Output files are tagged with "DRAFT — NOT FOR DIRECT CLINICAL DECISION-MAKING" until explicitly authorized by human medical reviewers.
