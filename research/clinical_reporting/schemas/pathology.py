"""
Pathology and Synoptic Histopathology Reporting Schema.
GastricPath AI — Phase 2: Clinical Information Schema.

Aligned with WHO 6th Edition (2026) and CAP Stomach Protocol 4.4.0.0.
"""

from typing import Optional
from pydantic import BaseModel, Field
from research.clinical_reporting.schemas.common import ClinicalField, VerificationStatus


class HistopathologyReport(BaseModel):
    """Structured surgical pathology and biopsy findings."""
    specimen_type: str = Field("Biopsy", description="Biopsy / Endoscopic Resection / Gastrectomy / Post-neoadjuvant Resection")
    anatomical_site: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    who_histological_type: ClinicalField = Field(
        default_factory=lambda: ClinicalField(
            value="Adenocarcinoma, NOS",
            verification_status=VerificationStatus.NOT_AVAILABLE,
            notes="WHO 6th Edition classification"
        )
    )
    lauren_classification: ClinicalField = Field(
        default_factory=lambda: ClinicalField(
            verification_status=VerificationStatus.NOT_AVAILABLE,
            notes="Intestinal / Diffuse / Mixed"
        )
    )
    histological_grade: ClinicalField = Field(
        default_factory=lambda: ClinicalField(
            verification_status=VerificationStatus.NOT_AVAILABLE,
            notes="G1 Well / G2 Moderate / G3 Poor / Undifferentiated"
        )
    )
    depth_of_invasion: ClinicalField = Field(
        default_factory=lambda: ClinicalField(
            verification_status=VerificationStatus.NOT_AVAILABLE,
            notes="e.g., Invades muscularis propria (pT2)"
        )
    )
    serosal_involvement: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    lymphovascular_invasion: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    perineural_invasion: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    
    # Surgical margins (for resections)
    proximal_margin: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_APPLICABLE))
    distal_margin: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_APPLICABLE))
    radial_omental_margin: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_APPLICABLE))
    
    # Lymph nodes
    lymph_nodes_examined: ClinicalField = Field(default_factory=lambda: ClinicalField(unit="nodes", verification_status=VerificationStatus.NOT_AVAILABLE))
    lymph_nodes_involved: ClinicalField = Field(default_factory=lambda: ClinicalField(unit="nodes", verification_status=VerificationStatus.NOT_AVAILABLE))
    extranodal_extension: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    
    # Neoadjuvant treatment effect (if post-chemo)
    treatment_effect_trg: ClinicalField = Field(
        default_factory=lambda: ClinicalField(
            verification_status=VerificationStatus.NOT_APPLICABLE,
            notes="Ryan / CAP Tumor Regression Grade (0: Complete, 1: Moderate, 2: Minimal, 3: Poor/No response)"
        )
    )
    pathologist_name_or_id: str = "De-identified Pathologist ID"
    pathology_report_date: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))

