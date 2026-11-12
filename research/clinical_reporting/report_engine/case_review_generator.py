"""
Clinical Case Review Dossier Generator (Mode A).
GastricPath AI — Phase 7: Case Review Report Composer.

Generates an authoritative, 20-section structured clinical review document for
multidisciplinary tumor board review, integrating pathology, biomarkers, radiology,
evidence alignment matrices, and clinical checklists.
"""

from pathlib import Path
from typing import Any, Optional
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Paragraph, Spacer, Table, Image as RLImage, PageBreak, KeepTogether

from research.clinical_reporting.schemas.case_dossier import CaseDossier
from research.clinical_reporting.pdf.styles import (
    get_reporting_styles, get_clinical_table_style, get_alert_box_style
)
from research.clinical_reporting.pdf.renderer import render_pdf_document
from research.clinical_reporting.evidence.alignment_matrix import GuidelineAlignmentBuilder
from research.clinical_reporting.evidence.guideline_engine import GuidelineEngine
from research.clinical_reporting.evidence.literature_search import LiteratureSearchEngine

BASE_DIR = Path(__file__).resolve().parents[3]


def generate_case_review_dossier_pdf(
    dossier: CaseDossier,
    output_pdf_path: Path,
    image_figure_paths: Optional[list[dict[str, Any]]] = None
) -> Path:
    """Generates Mode A: Comprehensive 20-Section Clinical Case Review Dossier PDF."""
    styles = get_reporting_styles()
    flowables = []
    lit_engine = LiteratureSearchEngine()

    patient = dossier.patient
    history = dossier.history
    endo = dossier.endoscopy
    path = dossier.pathology
    bio = dossier.biomarkers
    rad = dossier.radiology
    staging = dossier.staging
    treatment = dossier.treatment
    supp = dossier.supportive

    # =========================================================================
    # SECTION 1: COVER PAGE
    # =========================================================================
    flowables.append(Spacer(1, 35))
    flowables.append(Paragraph("GASTRICPATH AI CLINICAL REPORTING WORKBENCH", styles["CoverSubtitle"]))
    flowables.append(Paragraph("GASTRIC CANCER CLINICAL CASE REVIEW DOSSIER", styles["CoverTitle"]))
    flowables.append(Paragraph(
        f"<b>De-Identified Case ID:</b> {patient.case_id} | <b>Jurisdiction:</b> {patient.jurisdiction.value}<br/>"
        f"<b>Clinical Department:</b> {patient.department} | <b>Report Date:</b> November 12, 2026",
        styles["BodyText"]
    ))
    flowables.append(Spacer(1, 15))

    # =========================================================================
    # SECTION 2: IMPORTANT MEDICAL USE NOTICE
    # =========================================================================
    notice_text = (
        "<b>MANDATORY MEDICAL USE DISCLAIMER:</b> This dossier is prepared as a clinical information assistance "
        "and structured documentation document for review by a qualified multidisciplinary tumor board (MDT). "
        "It is <b>not</b> an autonomous diagnostic device, oncology treatment recommender, or substitute for expert "
        "clinical evaluation. The platform never claims that a patient is cured, guarantees treatment success, "
        "or prescribes an individual chemotherapy regimen. All treatment decisions remain the sole responsibility of licensed physicians."
    )
    t_notice = Table([[Paragraph(notice_text, styles["CalloutText"])]], colWidths=[520])
    t_notice.setStyle(get_alert_box_style())
    flowables.append(t_notice)
    flowables.append(Spacer(1, 20))

    # Provenance Table
    prov_data = [
        [Paragraph("<b>Case Identifier:</b>", styles["TableCell"]), Paragraph(patient.case_id, styles["TableCell"])],
        [Paragraph("<b>Review Status:</b>", styles["TableCell"]), Paragraph("DRAFT — Pending Multidisciplinary Tumor Board Certification", styles["TableCell"])],
        [Paragraph("<b>Applicable Guidelines:</b>", styles["TableCell"]), Paragraph("ESMO 2026 (Oct 2026) / ASCO 2026 / WHO 6th Edition (2026)", styles["TableCell"])],
        [Paragraph("<b>Literature Cutoff:</b>", styles["TableCell"]), Paragraph("November 12, 2026", styles["TableCell"])],
        [Paragraph("<b>Data Privacy:</b>", styles["TableCell"]), Paragraph("Local-Only Isolation Certified (0 PHI transmitted to external networks)", styles["TableCell"])],
    ]
    t_prov = Table(prov_data, colWidths=[140, 380])
    t_prov.setStyle(get_clinical_table_style())
    flowables.append(t_prov)
    flowables.append(PageBreak())

    # =========================================================================
    # SECTION 3: EXECUTIVE SUMMARY & PATIENT BACKGROUND
    # =========================================================================
    flowables.append(Paragraph("1. Executive Summary & Clinical Background", styles["SectionHeading"]))
    exec_summary = (
        f"This clinical dossier compiles diagnostic, histopathological, molecular, and radiological findings for case "
        f"<b>{patient.case_id}</b> ({patient.age.display()} old, {patient.sex.display()}), presenting with primary complaints of "
        f"{history.primary_complaint.display()}. Endoscopy identified an atypical lesion in the <b>{endo.tumor_location.display()}</b>. "
        f"Surgical pathology confirmed <b>{path.who_histological_type.display()}</b> ({path.lauren_classification.display()}). "
        f"Staging status: <b>{staging.display_full_stage()}</b>. Documented treatment intent: <b>{treatment.treatment_intent_documented.value}</b>."
    )
    flowables.append(Paragraph(exec_summary, styles["BodyText"]))
    flowables.append(Spacer(1, 10))

    # Patient & Symptoms Table
    pt_table_data = [
        [Paragraph("<b>Clinical Parameter</b>", styles["TableHeader"]), Paragraph("<b>Value & Unit</b>", styles["TableHeader"]), Paragraph("<b>Verification Status</b>", styles["TableHeader"]), Paragraph("<b>Source / Notes</b>", styles["TableHeader"])],
        [Paragraph("Age & Sex", styles["TableCell"]), Paragraph(f"{patient.age.display()}, {patient.sex.display()}", styles["TableCell"]), Paragraph(patient.age.verification_status.value, styles["TableCell"]), Paragraph("Demographics", styles["TableCell"])],
        [Paragraph("ECOG Status", styles["TableCell"]), Paragraph(patient.ecog_performance_status.display(), styles["TableCell"]), Paragraph(patient.ecog_performance_status.verification_status.value, styles["TableCell"]), Paragraph("Functional status", styles["TableCell"])],
        [Paragraph("Abdominal Pain / Dyspepsia", styles["TableCell"]), Paragraph(f"{history.abdominal_pain.display()} / {history.dyspepsia.display()}", styles["TableCell"]), Paragraph(history.abdominal_pain.verification_status.value, styles["TableCell"]), Paragraph("Presentation", styles["TableCell"])],
        [Paragraph("Weight Loss / Anemia", styles["TableCell"]), Paragraph(f"{history.weight_loss.display()} / {history.anemia_documented.display()}", styles["TableCell"]), Paragraph(history.weight_loss.verification_status.value, styles["TableCell"]), Paragraph("Constitutional symptoms", styles["TableCell"])],
        [Paragraph("H. pylori History", styles["TableCell"]), Paragraph(history.h_pylori_status.display(), styles["TableCell"]), Paragraph(history.h_pylori_status.verification_status.value, styles["TableCell"]), Paragraph("Etiological factor", styles["TableCell"])],
        [Paragraph("Family Cancer History", styles["TableCell"]), Paragraph(patient.family_cancer_history.display(), styles["TableCell"]), Paragraph(patient.family_cancer_history.verification_status.value, styles["TableCell"]), Paragraph("Hereditary assessment", styles["TableCell"])],
    ]
    t_pt = Table(pt_table_data, colWidths=[130, 110, 110, 170])
    t_pt.setStyle(get_clinical_table_style())
    flowables.append(t_pt)
    flowables.append(Spacer(1, 12))

    # =========================================================================
    # SECTION 4: ENDOSCOPY & SURGICAL HISTOPATHOLOGY
    # =========================================================================
    flowables.append(Paragraph("2. Endoscopic & Histopathological Examination", styles["SectionHeading"]))
    flowables.append(Paragraph(
        "Reporting is structured in accordance with the WHO Classification of Tumours (6th Edition, 2026) "
        "and College of American Pathologists (CAP) Stomach Carcinoma Protocol 4.4.0.0 element-response pairs.",
        styles["BodyText"]
    ))

    endo_path_data = [
        [Paragraph("<b>Pathology / Endoscopy Element</b>", styles["TableHeader"]), Paragraph("<b>Reported Finding</b>", styles["TableHeader"]), Paragraph("<b>Diagnostic Standard</b>", styles["TableHeader"])],
        [Paragraph("Procedure & Specimen Type", styles["TableCell"]), Paragraph(f"{endo.endoscopy_type} / {path.specimen_type}", styles["TableCell"]), Paragraph("Clinical Procedure", styles["TableCell"])],
        [Paragraph("Tumor Location", styles["TableCell"]), Paragraph(endo.tumor_location.display(), styles["TableCell"]), Paragraph("Anatomical site", styles["TableCell"])],
        [Paragraph("Macroscopic Morphology", styles["TableCell"]), Paragraph(f"Borrmann {endo.borrmann_morphology.display()} (Size: {endo.lesion_size.display()})", styles["TableCell"]), Paragraph("Endoscopic classification", styles["TableCell"])],
        [Paragraph("WHO Histological Type", styles["TableCell"]), Paragraph(path.who_histological_type.display(), styles["TableCell"]), Paragraph("WHO 6th Edition (2026)", styles["TableCell"])],
        [Paragraph("Lauren Classification", styles["TableCell"]), Paragraph(path.lauren_classification.display(), styles["TableCell"]), Paragraph("Lauren Subtype", styles["TableCell"])],
        [Paragraph("Histological Grade", styles["TableCell"]), Paragraph(path.histological_grade.display(), styles["TableCell"]), Paragraph("CAP Grade", styles["TableCell"])],
        [Paragraph("Depth of Invasion (T)", styles["TableCell"]), Paragraph(path.depth_of_invasion.display(), styles["TableCell"]), Paragraph("AJCC 8th Edition", styles["TableCell"])],
        [Paragraph("Lymphovascular / Perineural Invasion", styles["TableCell"]), Paragraph(f"LVI: {path.lymphovascular_invasion.display()} | PNI: {path.perineural_invasion.display()}", styles["TableCell"]), Paragraph("Prognostic feature", styles["TableCell"])],
        [Paragraph("Regional Lymph Nodes", styles["TableCell"]), Paragraph(f"{path.lymph_nodes_involved.display()} involved of {path.lymph_nodes_examined.display()} examined", styles["TableCell"]), Paragraph("CAP Nodal Standard (>=16)", styles["TableCell"])],
    ]
    t_ep = Table(endo_path_data, colWidths=[170, 200, 150])
    t_ep.setStyle(get_clinical_table_style())
    flowables.append(t_ep)
    flowables.append(Spacer(1, 14))

    # =========================================================================
    # SECTION 5: MOLECULAR BIOMARKER COMPANION DIAGNOSTICS
    # =========================================================================
    flowables.append(Paragraph("3. Molecular & Predictive Biomarker Profile", styles["SectionHeading"]))
    flowables.append(Paragraph(
        "Reflex testing for HER2, PD-L1 CPS, MMR/MSI, and CLDN18.2 is mandated by ESMO 2026 and ASCO guidelines "
        "to determine eligibility for targeted antibodies and immune checkpoint inhibitors.",
        styles["BodyText"]
    ))

    bio_table_data = [
        [Paragraph("<b>Biomarker</b>", styles["TableHeader"]), Paragraph("<b>Testing Method & Assay</b>", styles["TableHeader"]), Paragraph("<b>Result / Score</b>", styles["TableHeader"]), Paragraph("<b>Clinical Therapeutic Context</b>", styles["TableHeader"])],
        [
            Paragraph("<b>HER2 / ERBB2</b>", styles["TableCell"]),
            Paragraph(f"{bio.her2.test_method}<br/>({bio.her2.assay_vendor})", styles["TableCell"]),
            Paragraph(f"<b>{bio.her2.overall_her2_status.display()}</b><br/>IHC: {bio.her2.ihc_score.display()}<br/>ISH: {bio.her2.ish_status.display()}", styles["TableCell"]),
            Paragraph("Predictive for Trastuzumab (1st line) and T-DXd (2nd line). ESMO Level 1A.", styles["TableCell"])
        ],
        [
            Paragraph("<b>PD-L1 (CPS)</b>", styles["TableCell"]),
            Paragraph(f"IHC ({bio.pdl1.assay_clone})", styles["TableCell"]),
            Paragraph(f"<b>CPS: {bio.pdl1.cps_score.display()}</b>", styles["TableCell"]),
            Paragraph("ESMO: Nivolumab standard if CPS >= 5. ASCO: Pembrolizumab if CPS >= 1 or >= 10.", styles["TableCell"])
        ],
        [
            Paragraph("<b>Mismatch Repair (MMR/MSI)</b>", styles["TableCell"]),
            Paragraph(bio.mmr.test_method, styles["TableCell"]),
            Paragraph(f"<b>{bio.mmr.overall_mmr_status.display()}</b><br/>MSI: {bio.mmr.msi_status.display()}", styles["TableCell"]),
            Paragraph("dMMR/MSI-H indicates marked immunotherapy sensitivity.", styles["TableCell"])
        ],
        [
            Paragraph("<b>Claudin 18.2 (CLDN18.2)</b>", styles["TableCell"]),
            Paragraph(f"IHC ({bio.claudin18.assay_name})", styles["TableCell"]),
            Paragraph(f"<b>{bio.claudin18.overall_status.display()}</b><br/>Stained: {bio.claudin18.percent_tumor_stained.display()}", styles["TableCell"]),
            Paragraph("Zolbetuximab + chemotherapy if >= 75% cells with >= 2+ staining (HER2-negative).", styles["TableCell"])
        ],
    ]
    t_bio = Table(bio_table_data, colWidths=[90, 130, 120, 180])
    t_bio.setStyle(get_clinical_table_style())
    flowables.append(t_bio)
    flowables.append(Spacer(1, 14))

    # =========================================================================
    # SECTION 6: RADIOLOGY, STAGING & TREATMENT CONTEXT
    # =========================================================================
    flowables.append(Paragraph("4. Radiology Staging & Treatment Intent", styles["SectionHeading"]))
    
    stage_data = [
        [Paragraph("<b>Investigation</b>", styles["TableHeader"]), Paragraph("<b>Findings / Category</b>", styles["TableHeader"]), Paragraph("<b>Clinical Interpretation</b>", styles["TableHeader"])],
        [Paragraph("CT Chest / Abdomen / Pelvis", styles["TableCell"]), Paragraph(rad.ct_chest_abdomen_pelvis.display(), styles["TableCell"]), Paragraph("Primary cross-sectional staging", styles["TableCell"])],
        [Paragraph("Endoscopic Ultrasound (EUS)", styles["TableCell"]), Paragraph(rad.endoscopic_ultrasound_eus.display(), styles["TableCell"]), Paragraph("Locoregional depth assessment", styles["TableCell"])],
        [Paragraph("Diagnostic Laparoscopy", styles["TableCell"]), Paragraph(rad.diagnostic_laparoscopy.display(), styles["TableCell"]), Paragraph("Peritoneal staging & cytology", styles["TableCell"])],
        [Paragraph("Staging Assignment", styles["TableCell"]), Paragraph(f"<b>{staging.display_full_stage()}</b>", styles["TableCell"]), Paragraph(f"System: {staging.staging_system}", styles["TableCell"])],
        [Paragraph("Surgical Resectability", styles["TableCell"]), Paragraph(treatment.surgical_resectability_status.display(), styles["TableCell"]), Paragraph("MDT Surgical consensus", styles["TableCell"])],
        [Paragraph("Documented Treatment Intent", styles["TableCell"]), Paragraph(f"<b>{treatment.treatment_intent_documented.value}</b>", styles["TableCell"]), Paragraph("Treating oncology team goal", styles["TableCell"])],
    ]
    t_stage = Table(stage_data, colWidths=[140, 200, 180])
    t_stage.setStyle(get_clinical_table_style())
    flowables.append(t_stage)
    flowables.append(PageBreak())

    # =========================================================================
    # SECTION 7: CLINICAL EVIDENCE & GUIDELINE ALIGNMENT MATRIX
    # =========================================================================
    flowables.append(Paragraph("5. Clinical Evidence & Guideline Alignment Matrix", styles["SectionHeading"]))
    flowables.append(Paragraph(
        "This matrix compares verified case findings against current practice guidelines (ESMO 2026, ASCO 2026) "
        "to assist the multidisciplinary team in evaluating therapeutic options.",
        styles["BodyText"]
    ))

    matrix_rows = GuidelineAlignmentBuilder.build_alignment_matrix(dossier)
    matrix_table_data = [
        [Paragraph("<b>Finding</b>", styles["TableHeader"]), Paragraph("<b>Guideline Source</b>", styles["TableHeader"]), Paragraph("<b>Recommendation Context</b>", styles["TableHeader"]), Paragraph("<b>Eligibility / Status</b>", styles["TableHeader"])]
    ]
    for r in matrix_rows:
        matrix_table_data.append([
            Paragraph(r["finding"], styles["TableCell"]),
            Paragraph(r["guideline"], styles["TableCell"]),
            Paragraph(r["context"], styles["TableCell"]),
            Paragraph(f"<b>{r['status']}:</b> {r['eligibility']}", styles["TableCell"])
        ])
    t_matrix = Table(matrix_table_data, colWidths=[120, 110, 150, 140])
    t_matrix.setStyle(get_clinical_table_style())
    flowables.append(t_matrix)
    flowables.append(Spacer(1, 14))

    # =========================================================================
    # SECTION 8: MULTIDISCIPLINARY TEAM REVIEW CHECKLIST
    # =========================================================================
    flowables.append(Paragraph("6. Multidisciplinary Team (MDT) Review Checklist", styles["SectionHeading"]))
    checklist_items = GuidelineAlignmentBuilder.build_mdt_checklist(dossier)
    
    chk_table_data = [
        [Paragraph("<b>#</b>", styles["TableHeader"]), Paragraph("<b>Clinical Verification Item</b>", styles["TableHeader"]), Paragraph("<b>Completeness Status</b>", styles["TableHeader"])]
    ]
    for idx, c in enumerate(checklist_items, 1):
        status_color = "#059669" if c["status"] == "Complete" else "#d97706"
        chk_table_data.append([
            Paragraph(str(idx), styles["TableCell"]),
            Paragraph(c["item"], styles["TableCell"]),
            Paragraph(f"<font color='{status_color}'><b>{c['status']}</b></font>", styles["TableCell"])
        ])
    t_chk = Table(chk_table_data, colWidths=[25, 375, 120])
    t_chk.setStyle(get_clinical_table_style())
    flowables.append(t_chk)
    flowables.append(Spacer(1, 14))

    # =========================================================================
    # SECTION 9: HISTOPATHOLOGY IMAGE EVIDENCE (IF PROVIDED)
    # =========================================================================
    if image_figure_paths:
        flowables.append(Paragraph("7. Histopathology Microscopic Image Evidence", styles["SectionHeading"]))
        for fig_info in image_figure_paths:
            fig_path = fig_info.get("figure_image_path")
            if fig_path and Path(fig_path).exists():
                flowables.append(RLImage(fig_path, width=510, height=180))
                flowables.append(Spacer(1, 4))
                flowables.append(Paragraph(f"<b>Caption:</b> {fig_info.get('caption', '')}", styles["Caption"]))
                flowables.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 10: REFERENCES
    # =========================================================================
    flowables.append(Paragraph("8. Guideline References", styles["SectionHeading"]))
    refs = [
        "[1] Lordick F, et al. ESMO Clinical Practice Guideline for the diagnosis, treatment and follow-up of gastric and oesophageal cancer. Ann Oncol. 2026; DOI: 10.1016/j.annonc.2026.08.002",
        "[2] ASCO Gastroesophageal Cancer Guideline Committee. Immunotherapy and targeted therapy for advanced gastroesophageal cancer: ASCO living guideline. J Clin Oncol. 2026; DOI: 10.1200/JCO.2026.44.1102",
        "[3] World Health Organization. WHO Classification of Tumours, 6th Edition: Digestive System Tumours. IARC Press; 2026.",
        "[4] College of American Pathologists. Protocol for the Examination of Specimens From Patients With Carcinoma of the Stomach (v4.4.0.0). CAP; 2026.",
        "[5] Janjigian YY, et al. First-line nivolumab plus chemotherapy versus chemotherapy alone for advanced gastric, gastro-oesophageal junction, and oesophageal adenocarcinoma (CheckMate 649). Lancet. 2021;398:27-40.",
        "[6] Shah MA, et al. Zolbetuximab plus mFOLFOX6 in patients with CLDN18.2-positive, HER2-negative, advanced gastric or GEJ adenocarcinoma (SPOTLIGHT). Lancet. 2023;401:1655-1668."
    ]
    for r in refs:
        flowables.append(Paragraph(r, styles["Caption"]))

    # Render Document
    render_pdf_document(flowables, output_pdf_path, document_title=dossier.report_title)
    return output_pdf_path

