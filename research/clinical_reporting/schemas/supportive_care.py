"""
Supportive Care, Nutrition, and Survivorship Schema.
GastricPath AI — Phase 2: Clinical Information Schema.
"""

from typing import Optional
from pydantic import BaseModel, Field
from research.clinical_reporting.schemas.common import ClinicalField, VerificationStatus


class SupportiveCare(BaseModel):
    """Nutritional evaluation, symptom burden, and palliative supportive care."""
    nutritional_risk_screening: ClinicalField = Field(
        default_factory=lambda: ClinicalField(
            verification_status=VerificationStatus.NOT_AVAILABLE,
            notes="NRS-2002 / MUST score / Subjective Global Assessment (SGA)"
        )
    )
    dietitian_consultation_status: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    current_weight: ClinicalField = Field(default_factory=lambda: ClinicalField(unit="kg", verification_status=VerificationStatus.NOT_AVAILABLE))
    bmi: ClinicalField = Field(default_factory=lambda: ClinicalField(unit="kg/m2", verification_status=VerificationStatus.NOT_AVAILABLE))
    serum_albumin: ClinicalField = Field(default_factory=lambda: ClinicalField(unit="g/dL", verification_status=VerificationStatus.NOT_AVAILABLE))
    hemoglobin: ClinicalField = Field(default_factory=lambda: ClinicalField(unit="g/dL", verification_status=VerificationStatus.NOT_AVAILABLE))
    pain_score_numeric: ClinicalField = Field(default_factory=lambda: ClinicalField(unit="/10", verification_status=VerificationStatus.NOT_AVAILABLE))
    antiemetic_support: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    palliative_care_involvement: ClinicalField = Field(
        default_factory=lambda: ClinicalField(
            verification_status=VerificationStatus.NOT_AVAILABLE,
            notes="Early palliative integration recommended alongside oncologic care for advanced disease"
        )
    )
    psychological_support: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))

