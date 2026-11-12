"""
Clinical Presentation and History Schema.
GastricPath AI — Phase 2: Clinical Information Schema.
"""

from typing import Optional
from pydantic import BaseModel, Field
from research.clinical_reporting.schemas.common import ClinicalField, VerificationStatus


class ClinicalHistory(BaseModel):
    """Patient symptoms, risk factors, and prior medical history."""
    primary_complaint: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    abdominal_pain: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    dyspepsia: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    early_satiety: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    nausea_vomiting: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    dysphagia: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    gi_bleeding: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    weight_loss: ClinicalField = Field(default_factory=lambda: ClinicalField(unit="kg", verification_status=VerificationStatus.NOT_AVAILABLE))
    anemia_documented: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    symptom_duration: ClinicalField = Field(default_factory=lambda: ClinicalField(unit="months", verification_status=VerificationStatus.NOT_AVAILABLE))
    
    # Risk factors
    h_pylori_status: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    smoking_history: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    alcohol_exposure: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    prior_gastric_surgery: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    relevant_medications: list[str] = Field(default_factory=list)
    nutritional_concerns: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))

