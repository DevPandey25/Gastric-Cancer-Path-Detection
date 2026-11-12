"""
Clinical Knowledge and Guideline-Directed Treatment Engine.
GastricPath AI — Phase 5: Clinical Knowledge and Guideline Engine.

Provides evidence-grounded treatment context, biomarker eligibility criteria,
and regional regulatory indications across ESMO, ASCO, NCI, and Asian guidelines.
"""

from typing import Any, Optional
from research.clinical_reporting.evidence.source_registry import (
    Jurisdiction, get_biomarker_context, get_guidelines_for_jurisdiction, GUIDELINE_REGISTRY
)
from research.clinical_reporting.schemas.common import TreatmentIntent


class GuidelineEngine:
    """Evaluates case findings against current clinical practice guidelines."""

    @staticmethod
    def get_disease_context(histological_type: str) -> dict[str, Any]:
        """Provides histological differential context."""
        h_type = histological_type.lower()
        if "adenocarcinoma" in h_type or "carcinoma" in h_type:
            return {
                "entity": "Gastric Adenocarcinoma",
                "standard": "WHO 6th Edition / ESMO 2026 Guidelines apply.",
                "pathway": "Adenocarcinoma multimodal staging and systemic therapy pathway.",
                "warning": None
            }
        elif "gist" in h_type or "stromal" in h_type:
            return {
                "entity": "Gastrointestinal Stromal Tumor (GIST)",
                "standard": "ESMO / NCCN GIST Guidelines apply (KIT/PDGFRA mutational testing).",
                "pathway": "Tyrosine kinase inhibitors (Imatinib) and surgical resection.",
                "warning": "Gastric adenocarcinoma chemotherapy/immunotherapy pathways do NOT apply to GIST."
            }
        elif "neuroendocrine" in h_type or "net" in h_type or "nec" in h_type:
            return {
                "entity": "Gastric Neuroendocrine Neoplasm (NEN)",
                "standard": "ENETS / WHO Neuroendocrine Guidelines apply (Ki-67 grading).",
                "pathway": "Somatostatin analogs, PRRT, or platinum/etoposide depending on grade (G1-G3 / NEC).",
                "warning": "Gastric adenocarcinoma biomarker algorithms do not apply to neuroendocrine neoplasms."
            }
        elif "lymphoma" in h_type or "malt" in h_type:
            return {
                "entity": "Gastric Lymphoma (MALT / DLBCL)",
                "standard": "ESMO Lymphoma Guidelines apply (H. pylori eradication / R-CHOP).",
                "pathway": "H. pylori eradication first-line for early MALT; systemic immunochemotherapy for DLBCL.",
                "warning": "Gastrectomy is generally avoided; systemic lymphoma protocols apply."
            }
        return {
            "entity": histological_type,
            "standard": "Requires expert gastrointestinal pathology review.",
            "pathway": "General oncology diagnostic work-up.",
            "warning": "Ensure accurate histological confirmation prior to initiating systemic protocols."
        }

    @staticmethod
    def get_treatment_pathway_by_stage(stage_group: str, resectability: str = "Resectable") -> dict[str, Any]:
        """Maps AJCC stage and resectability to ESMO 2026 / ASCO guideline pathways."""
        stage = stage_group.upper().strip()
        
        if "IV" in stage or "METASTATIC" in stage:
            return {
                "category": "Metastatic or Unresectable Gastric Adenocarcinoma",
                "goal": "Disease control, prolongation of overall survival, and symptom palliation.",
                "primary_approach": "Systemic therapy guided by HER2, PD-L1 (CPS), MMR/MSI, and CLDN18.2 biomarkers.",
                "first_line_options": [
                    "HER2-Positive (IHC 3+ or 2+/ISH+): Trastuzumab + Fluoropyrimidine/Platinum doublet (ToGA regimen).",
                    "CLDN18.2-Positive (>=75% cells >=2+), HER2-Negative: Zolbetuximab + mFOLFOX6 or CAPOX (SPOTLIGHT/GLOW).",
                    "PD-L1 CPS >= 5, HER2-Negative: Nivolumab + Fluoropyrimidine/Platinum (CheckMate-649).",
                    "PD-L1 CPS >= 1 or >= 10: Pembrolizumab + Fluoropyrimidine/Platinum (KEYNOTE-859).",
                    "dMMR / MSI-H: Immune checkpoint inhibitor monotherapy or combination (marked sensitivity)."
                ],
                "subsequent_lines": [
                    "HER2-Positive: Trastuzumab deruxtecan (T-DXd) second/third line (DESTINY-Gastric).",
                    "HER2-Negative: Paclitaxel + Ramucirumab second line (RAINBOW regimen), FOLFIRI, or Trifluridine/tipiracil (TAGS)."
                ],
                "curative_intent_note": "Curative-intent surgery is generally not indicated in systemic metastatic disease (M1), except in highly selected clinical trials with oligometastatic response."
            }
        elif any(s in stage for s in ["II", "III", "IB"]):
            return {
                "category": "Locally Advanced Resectable Gastric Adenocarcinoma (Stage >= cT2 and/or cN+)",
                "goal": "Curative-intent multimodal therapy.",
                "primary_approach": "Perioperative systemic chemotherapy plus radical D2 gastrectomy.",
                "regimen": "FLOT: 4 cycles pre-operative and 4 cycles post-operative (fluorouracil, leucovorin, oxaliplatin, docetaxel).",
                "surgical_standard": "Partial or total gastrectomy with formal D2 lymphadenectomy (minimum 16 lymph nodes evaluated).",
                "curative_intent_note": "Treatment is delivered with curative intent; however, individual cure cannot be guaranteed and long-term recurrence risk depends on final pathological ypTNM response."
            }
        elif "IA" in stage or "0" in stage:
            return {
                "category": "Very Early / Localized Gastric Adenocarcinoma (cT1a N0)",
                "goal": "Curative-intent organ-preserving or minimally invasive resection.",
                "primary_approach": "Endoscopic Submucosal Dissection (ESD) for non-ulcerated, well-differentiated lesions <= 2 cm (expanded criteria in East Asian guidelines).",
                "surgical_standard": "Laparoscopic subtotal gastrectomy with lymph node dissection if criteria for endoscopic resection are not met.",
                "curative_intent_note": "Favorable 5-year overall survival (>90% in clinical literature), but requires endoscopic follow-up surveillance."
            }
        return {
            "category": f"Gastric Cancer Stage {stage_group}",
            "goal": "Multidisciplinary team evaluation required.",
            "primary_approach": "Complete diagnostic laparoscopy and contrast imaging to determine resectability.",
            "curative_intent_note": "Treatment intent must be confirmed by the multidisciplinary tumor board."
        }

