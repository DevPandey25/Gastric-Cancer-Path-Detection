"""
De-Identified Patient and Case Metadata Schema.
GastricPath AI — Phase 2: Clinical Information Schema.
"""

from typing import Optional
from pydantic import BaseModel, Field
from research.clinical_reporting.schemas.common import ClinicalField, VerificationStatus
from research.clinical_reporting.evidence.source_registry import Jurisdiction


class PatientCase(BaseModel):
    """De-identified patient case record with privacy safeguards."""
    case_id: str = Field(..., description="De-identified alphanumeric case ID (e.g., GP-GC-2026-001)")
    age: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    sex: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    jurisdiction: Jurisdiction = Jurisdiction.GLOBAL
    department: str = "Gastrointestinal Oncology"
    initial_presentation_date: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    diagnosis_date: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    ecog_performance_status: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    comorbidities: list[str] = Field(default_factory=list)
    prior_malignancies: list[str] = Field(default_factory=list)
    family_cancer_history: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    hereditary_syndrome_assessment: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    consent_and_data_use_status: str = "Authorized for De-Identified Clinical Case Review"

    def validate_no_direct_identifiers(self) -> bool:
        """Verifies that no direct personal identifiers exist in case_id."""
        prohibited = ["mrn", "ssn", "aadhar", "name", "patient", "dob"]
        for p in prohibited:
            if p in self.case_id.lower() and not self.case_id.startswith("GP-"):
                return False
        return True

