"""
Clinical Evidence and Guideline Alignment Matrix Generator.
GastricPath AI — Phase 6: Evidence Retrieval and Literature Engine.

Evaluates structured case findings against authoritative guidelines (ESMO 2026, ASCO 2026)
to construct an actionable evidence alignment table and multidisciplinary checklist.
"""

from typing import Any
from research.clinical_reporting.schemas.case_dossier import CaseDossier
from research.clinical_reporting.evidence.source_registry import Jurisdiction, get_biomarker_context


class GuidelineAlignmentBuilder:
    """Builds clinical evidence alignment matrices for Case Review dossiers."""

    @staticmethod
    def build_alignment_matrix(dossier: CaseDossier) -> list[dict[str, Any]]:
        """Constructs row-by-row alignment matrix comparing case data against guidelines."""
        rows = []
        jurisdiction = dossier.patient.jurisdiction

        # 1. Histological Diagnosis
        path = dossier.pathology
        hist_val = path.who_histological_type.display()
        rows.append({
            "finding": f"Histological Type: {hist_val}",
            "source": "Surgical Pathology / Biopsy Report",
            "guideline": "WHO 6th Edition (2026) / CAP Stomach Protocol",
            "context": "Establishes epithelial adenocarcinoma vs non-epithelial neoplasms.",
            "eligibility": "Adenocarcinoma confirms eligibility for standard perioperative or systemic protocols.",
            "missing_info": "Lauren classification" if not path.lauren_classification.is_available else "Complete",
            "medical_review_required": "Yes (Pathology Review)",
            "status": "Verified" if path.who_histological_type.is_available else "Incomplete"
        })

        # 2. HER2 Status
        her2 = dossier.biomarkers.her2
        her2_val = her2.overall_her2_status.display()
        her2_ctx = get_biomarker_context("HER2", jurisdiction)
        rows.append({
            "finding": f"HER2 Status: {her2_val} (IHC: {her2.ihc_score.display()})",
            "source": f"IHC / ISH ({her2.testing_laboratory})",
            "guideline": "ESMO 2026 / ASCO 2026 Living Guideline",
            "context": "Predictive of benefit from Trastuzumab (1st line) and T-DXd (2nd line).",
            "eligibility": "Eligible for Trastuzumab if IHC 3+ or IHC 2+/ISH+." if "pos" in her2_val.lower() else "Trastuzumab not indicated.",
            "missing_info": "ISH reflex testing needed" if "2+" in her2.ihc_score.display() and not her2.ish_status.is_available else "Complete",
            "medical_review_required": "Yes",
            "status": "Verified" if her2.overall_her2_status.is_available else "Unresolved"
        })

        # 3. PD-L1 CPS Status
        pdl1 = dossier.biomarkers.pdl1
        pdl1_val = pdl1.cps_score.display()
        rows.append({
            "finding": f"PD-L1 Combined Positive Score: {pdl1_val}",
            "source": f"IHC Assay ({pdl1.assay_clone})",
            "guideline": "ESMO 2026 / CheckMate-649 / KEYNOTE-859",
            "context": "Nivolumab combination standard in ESMO for CPS >= 5; Pembrolizumab for CPS >= 1 / >= 10 in ASCO.",
            "eligibility": "Eligible for 1st-line Nivolumab + chemo if CPS >= 5." if (pdl1.cps_score.is_available and float(pdl1.cps_score.value or 0) >= 5) else "Evaluate clinical trial or alternative targeted options.",
            "missing_info": "Awaiting numerical CPS score" if not pdl1.cps_score.is_available else "Complete",
            "medical_review_required": "Yes (Medical Oncology)",
            "status": "Verified" if pdl1.cps_score.is_available else "Incomplete"
        })

        # 4. Claudin 18.2 Status
        cldn = dossier.biomarkers.claudin18
        cldn_val = cldn.overall_status.display()
        cldn_ctx = get_biomarker_context("CLDN18_2", jurisdiction)
        rows.append({
            "finding": f"CLDN18.2 Status: {cldn_val} ({cldn.percent_tumor_stained.display()} stained)",
            "source": f"IHC ({cldn.assay_name})",
            "guideline": "ESMO 2026 / SPOTLIGHT & GLOW Trials",
            "context": "Zolbetuximab + mFOLFOX6 recommended for CLDN18.2+ (>=75% cells >=2+), HER2- metastatic GC.",
            "eligibility": "Eligible for Zolbetuximab combination if HER2-negative." if "pos" in cldn_val.lower() else "Not eligible based on criteria.",
            "missing_info": "Percent cells stained and intensity criteria" if not cldn.overall_status.is_available else "Complete",
            "medical_review_required": "Yes (Check Local Drug Availability)",
            "status": "Verified" if cldn.overall_status.is_available else "Not Assessed"
        })

        # 5. Mismatch Repair (MMR / MSI)
        mmr = dossier.biomarkers.mmr
        mmr_val = mmr.overall_mmr_status.display()
        rows.append({
            "finding": f"Mismatch Repair / MSI: {mmr_val} ({mmr.msi_status.display()})",
            "source": "IHC (4-panel MLH1/PMS2/MSH2/MSH6) / PCR",
            "guideline": "ESMO 2026 / ASCO 2026 Immunotherapy Guidelines",
            "context": "dMMR/MSI-H tumors demonstrate profound sensitivity to immune checkpoint inhibitors.",
            "eligibility": "Eligible for pembrolizumab/nivolumab; reconsider adjuvant fluoropyrimidine monotherapy if resected.",
            "missing_info": "Complete 4-protein IHC assessment" if not mmr.overall_mmr_status.is_available else "Complete",
            "medical_review_required": "Yes",
            "status": "Verified" if mmr.overall_mmr_status.is_available else "Incomplete"
        })

        # 6. Clinical & Pathological Staging
        staging = dossier.staging
        stage_str = staging.display_full_stage()
        rows.append({
            "finding": f"Staging Assignment: {stage_str}",
            "source": f"{staging.staging_system} (MDT)",
            "guideline": "AJCC 8th Edition / ESMO Locoregional Guidelines",
            "context": "Directs curative-intent perioperative therapy (FLOT) vs definitive palliative systemic therapy.",
            "eligibility": "Perioperative FLOT indicated if stage >= cT2 and/or cN+ M0." if ("II" in stage_str or "III" in stage_str or "T2" in stage_str) else "Stage-specific evaluation.",
            "missing_info": "Diagnostic laparoscopy washings" if not dossier.radiology.diagnostic_laparoscopy.is_available else "Complete",
            "medical_review_required": "Yes (MDT Triage)",
            "status": "Verified" if staging.can_establish_stage() else "Incomplete"
        })

        # 7. Nutritional Status & Supportive Care
        supp = dossier.supportive
        nutr_val = supp.nutritional_risk_screening.display()
        rows.append({
            "finding": f"Nutritional Screening: {nutr_val} (BMI: {supp.bmi.display()}, Alb: {supp.serum_albumin.display()})",
            "source": "Clinical Nutrition / Nursing Assessment",
            "guideline": "ESPEN Oncology Nutrition Guidelines / ESMO Supportive Care",
            "context": "Early dietitian consultation and oral nutritional supplementation to prevent sarcopenia.",
            "eligibility": "Indication for nutritional prehabilitation prior to surgery or systemic chemotherapy.",
            "missing_info": "Dietitian consult status" if not supp.dietitian_consultation_status.is_available else "Complete",
            "medical_review_required": "Yes (Clinical Nutrition)",
            "status": "Verified" if supp.nutritional_risk_screening.is_available else "Not Assessed"
        })

        return rows

    @staticmethod
    def build_mdt_checklist(dossier: CaseDossier) -> list[dict[str, Any]]:
        """Constructs a 10-point completeness checklist for Multidisciplinary Tumor Board review."""
        checklist = [
            {"item": "Histological diagnosis of adenocarcinoma confirmed?", "checked": dossier.pathology.who_histological_type.is_available, "status": "Complete" if dossier.pathology.who_histological_type.is_available else "Pending"},
            {"item": "Primary anatomical location documented (Cardia vs Non-cardia)?", "checked": dossier.endoscopy.tumor_location.is_available, "status": "Complete" if dossier.endoscopy.tumor_location.is_available else "Pending"},
            {"item": "Biopsy specimen count adequate (>= 6-8 fragments)?", "checked": dossier.endoscopy.biopsy_count.is_available, "status": "Documented" if dossier.endoscopy.biopsy_count.is_available else "Unrecorded"},
            {"item": "Contrast CT chest/abdomen/pelvis completed within 4 weeks?", "checked": dossier.radiology.ct_chest_abdomen_pelvis.is_available, "status": "Complete" if dossier.radiology.ct_chest_abdomen_pelvis.is_available else "Pending"},
            {"item": "Diagnostic laparoscopy performed for locoregional disease?", "checked": dossier.radiology.diagnostic_laparoscopy.is_available, "status": "Documented" if dossier.radiology.diagnostic_laparoscopy.is_available else "Pending / Omitted"},
            {"item": "HER2 IHC/ISH testing complete?", "checked": dossier.biomarkers.her2.overall_her2_status.is_available, "status": "Complete" if dossier.biomarkers.her2.overall_her2_status.is_available else "Pending"},
            {"item": "PD-L1 CPS numerical score reported with assay provenance?", "checked": dossier.biomarkers.pdl1.cps_score.is_available, "status": "Complete" if dossier.biomarkers.pdl1.cps_score.is_available else "Pending"},
            {"item": "MMR IHC (4-protein) or MSI status established?", "checked": dossier.biomarkers.mmr.overall_mmr_status.is_available, "status": "Complete" if dossier.biomarkers.mmr.overall_mmr_status.is_available else "Pending"},
            {"item": "CLDN18.2 IHC testing evaluated (SPOTLIGHT criteria)?", "checked": dossier.biomarkers.claudin18.overall_status.is_available, "status": "Complete" if dossier.biomarkers.claudin18.overall_status.is_available else "Pending"},
            {"item": "Nutritional risk screening (NRS-2002 / MUST) completed?", "checked": dossier.supportive.nutritional_risk_screening.is_available, "status": "Complete" if dossier.supportive.nutritional_risk_screening.is_available else "Pending"}
        ]
        return checklist

