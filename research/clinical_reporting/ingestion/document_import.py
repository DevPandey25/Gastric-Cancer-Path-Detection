"""
Clinical Document Ingestion and Case Dossier Parser.
GastricPath AI — Phase 3: Document and Image Ingestion.
"""

import json
from pathlib import Path
from typing import Any, Optional

from research.clinical_reporting.schemas.case_dossier import CaseDossier
from research.clinical_reporting.schemas.patient_case import PatientCase
from research.clinical_reporting.security.file_validation import validate_file_path, SecurityValidationError
from research.clinical_reporting.security.privacy_guard import redact_direct_identifiers


def load_case_dossier_from_json(json_path: Path) -> CaseDossier:
    """Safely loads and validates a CaseDossier from a local JSON file."""
    if not json_path.exists():
        raise FileNotFoundError(f"Case file not found: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
        sanitized_text = redact_direct_identifiers(raw_text)
        data = json.loads(sanitized_text)
        
    return CaseDossier.model_validate(data)


def save_case_dossier_to_json(dossier: CaseDossier, output_path: Path) -> Path:
    """Saves a CaseDossier to a local JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(dossier.model_dump_json(indent=2))
    return output_path


def create_blank_case_template(case_id: str = "GP-CASE-TEMPLATE") -> CaseDossier:
    """Generates an unpopulated, validated case template with all required fields initialized to Not Available."""
    patient = PatientCase(case_id=case_id)
    return CaseDossier(
        patient=patient,
        report_title="Gastric Adenocarcinoma Clinical Case Review Dossier",
        mdt_review_questions=[
            "Is the histological adenocarcinoma subtype confirmed on permanent sections?",
            "Are reflex biomarker companion diagnostics (HER2, PD-L1 CPS, MMR, CLDN18.2) complete?",
            "Has clinical staging (CT chest/abdomen/pelvis +/- laparoscopy) excluded distant metastasis?",
            "Has nutritional risk screening been documented by clinical nutrition?"
        ],
        missing_information_items=[
            "Awaiting final molecular biomarker report",
            "Awaiting diagnostic laparoscopy peritoneal cytology results"
        ]
    )

