"""
Endoscopy and Biopsy Findings Schema.
GastricPath AI — Phase 2: Clinical Information Schema.
"""

from typing import Optional
from pydantic import BaseModel, Field
from research.clinical_reporting.schemas.common import ClinicalField, VerificationStatus


class EndoscopyFindings(BaseModel):
    """Endoscopic examination details and macroscopic tumor descriptors."""
    procedure_date: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    endoscopy_type: str = "Esophagogastroduodenoscopy (EGD)"
    anatomical_site: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    # Anatomical locations: Cardia, Fundus, Body, Antrum, Pylorus, GE Junction
    tumor_location: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    lesion_size: ClinicalField = Field(default_factory=lambda: ClinicalField(unit="cm", verification_status=VerificationStatus.NOT_AVAILABLE))
    borrmann_morphology: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    ulceration_present: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    biopsy_count: ClinicalField = Field(default_factory=lambda: ClinicalField(unit="fragments", verification_status=VerificationStatus.NOT_AVAILABLE))
    sampling_sites: list[str] = Field(default_factory=list)
    endoscopist_impression: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    specimen_identifier: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))

