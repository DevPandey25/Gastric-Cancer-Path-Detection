"""
Master Unified Clinical Case Review Dossier Schema.
GastricPath AI — Phase 2: Clinical Information Schema.

Assembles all domain-specific components for Mode A report generation.
"""

from typing import Optional
from pydantic import BaseModel, Field

from research.clinical_reporting.schemas.common import ClinicalField, VerificationStatus
from research.clinical_reporting.schemas.patient_case import PatientCase
from research.clinical_reporting.schemas.clinical_history import ClinicalHistory
from research.clinical_reporting.schemas.endoscopy import EndoscopyFindings
from research.clinical_reporting.schemas.pathology import HistopathologyReport
from research.clinical_reporting.schemas.biomarkers import MolecularBiomarkers
from research.clinical_reporting.schemas.radiology import RadiologyFindings
from research.clinical_reporting.schemas.staging import StagingClassification
from research.clinical_reporting.schemas.treatment_history import TreatmentHistory
from research.clinical_reporting.schemas.supportive_care import SupportiveCare


class CaseDossier(BaseModel):
    """Complete structured clinical case for multidisciplinary team review."""
    patient: PatientCase
    history: ClinicalHistory = Field(default_factory=ClinicalHistory)
    endoscopy: EndoscopyFindings = Field(default_factory=EndoscopyFindings)
    pathology: HistopathologyReport = Field(default_factory=HistopathologyReport)
    biomarkers: MolecularBiomarkers = Field(default_factory=MolecularBiomarkers)
    radiology: RadiologyFindings = Field(default_factory=RadiologyFindings)
    staging: StagingClassification = Field(default_factory=StagingClassification)
    treatment: TreatmentHistory = Field(default_factory=TreatmentHistory)
    supportive: SupportiveCare = Field(default_factory=SupportiveCare)
    
    # Image references
    image_sample_ids: list[str] = Field(default_factory=list)
    
    # Report compilation metadata
    report_title: str = "Gastric Adenocarcinoma Clinical Case Review Dossier"
    dossier_version: str = "1.0"
    is_draft: bool = True
    mdt_review_questions: list[str] = Field(default_factory=list)
    missing_information_items: list[str] = Field(default_factory=list)

