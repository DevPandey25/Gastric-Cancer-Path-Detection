"""
Radiology and Clinical Staging Investigations Schema.
GastricPath AI — Phase 2: Clinical Information Schema.
"""

from typing import Optional
from pydantic import BaseModel, Field
from research.clinical_reporting.schemas.common import ClinicalField, VerificationStatus


class RadiologyFindings(BaseModel):
    """Cross-sectional imaging, endoscopic ultrasound, and laparoscopy findings."""
    ct_chest_abdomen_pelvis: ClinicalField = Field(
        default_factory=lambda: ClinicalField(
            verification_status=VerificationStatus.NOT_AVAILABLE,
            notes="Primary staging modality: assess primary gastric wall thickening, regional nodes, distant spread"
        )
    )
    endoscopic_ultrasound_eus: ClinicalField = Field(
        default_factory=lambda: ClinicalField(
            verification_status=VerificationStatus.NOT_AVAILABLE,
            notes="Locoregional T and N staging assessment"
        )
    )
    pet_ct_findings: ClinicalField = Field(
        default_factory=lambda: ClinicalField(
            verification_status=VerificationStatus.NOT_AVAILABLE,
            notes="Metabolic evaluation for occult distant metastatic disease"
        )
    )
    diagnostic_laparoscopy: ClinicalField = Field(
        default_factory=lambda: ClinicalField(
            verification_status=VerificationStatus.NOT_AVAILABLE,
            notes="Peritoneal inspection and cytology washings (mandatory for stage >= T1b resectable disease in many guidelines)"
        )
    )
    peritoneal_metastases_or_ascites: ClinicalField = Field(
        default_factory=lambda: ClinicalField(
            verification_status=VerificationStatus.NOT_AVAILABLE,
            notes="Presence of peritoneal nodules, carcinomatosis, or malignant ascites"
        )
    )
    liver_metastases: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    lung_metastases: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    distant_lymphadenopathy: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    radiology_examination_date: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    radiologist_impression: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))

