"""
Treatment History and Documented Intent Schema.
GastricPath AI — Phase 2: Clinical Information Schema.

Enforces strict separation of curative intent from cure claims.
"""

from typing import Optional
from pydantic import BaseModel, Field
from research.clinical_reporting.schemas.common import ClinicalField, TreatmentIntent, VerificationStatus


class TreatmentRecord(BaseModel):
    modality: str = Field(..., description="Surgery / Chemotherapy / Radiotherapy / Targeted / Immunotherapy")
    regimen_or_procedure: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    cycles_completed: Optional[int] = None
    response_evaluation: Optional[str] = None  # Complete Response, Partial Response, Stable Disease, Progressive Disease
    tolerability_and_toxicities: Optional[str] = None


class TreatmentHistory(BaseModel):
    """Prior and current oncology treatments documented in medical records."""
    treatment_intent_documented: TreatmentIntent = TreatmentIntent.NOT_DOCUMENTED
    surgical_resectability_status: ClinicalField = Field(
        default_factory=lambda: ClinicalField(
            verification_status=VerificationStatus.NOT_AVAILABLE,
            notes="Resectable / Potentially Resectable (Borderline) / Unresectable"
        )
    )
    prior_treatments: list[TreatmentRecord] = Field(default_factory=list)
    current_treatment_plan: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    clinical_trial_participation: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))

