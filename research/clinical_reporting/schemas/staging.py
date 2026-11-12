"""
TNM Staging Classification Schema.
GastricPath AI — Phase 2: Clinical Information Schema.

Strictly enforces separation of clinical (cTNM), pathological (pTNM),
and post-neoadjuvant (ypTNM) staging, preventing automated guessing.
"""

from typing import Optional
from pydantic import BaseModel, Field
from research.clinical_reporting.schemas.common import ClinicalField, VerificationStatus


class StagingClassification(BaseModel):
    """Clinical and pathological stage assignment based on AJCC/UICC 8th Edition."""
    staging_system: str = "AJCC/UICC TNM 8th Edition (Stomach)"
    staging_type: str = Field("cTNM", description="cTNM (Clinical) / pTNM (Pathological) / ypTNM (Post-Neoadjuvant)")
    
    t_category: ClinicalField = Field(
        default_factory=lambda: ClinicalField(
            verification_status=VerificationStatus.NOT_AVAILABLE,
            notes="TX, T0, Tis, T1a, T1b, T2, T3, T4a, T4b"
        )
    )
    n_category: ClinicalField = Field(
        default_factory=lambda: ClinicalField(
            verification_status=VerificationStatus.NOT_AVAILABLE,
            notes="NX, N0 (0 nodes), N1 (1-2), N2 (3-6), N3a (7-15), N3b (>=16)"
        )
    )
    m_category: ClinicalField = Field(
        default_factory=lambda: ClinicalField(
            verification_status=VerificationStatus.NOT_AVAILABLE,
            notes="M0 (No distant metastasis), M1 (Distant metastasis present)"
        )
    )
    overall_stage_group: ClinicalField = Field(
        default_factory=lambda: ClinicalField(
            verification_status=VerificationStatus.NOT_AVAILABLE,
            notes="Stage 0, IA, IB, IIA, IIB, IIIA, IIIB, IIIC, IV"
        )
    )
    staging_date: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE))
    staging_clinician_or_mdt: str = "Multidisciplinary Tumor Board (MDT)"
    
    def can_establish_stage(self) -> bool:
        """Determines if sufficient verified data exists to establish staging."""
        return (
            self.t_category.is_available and
            self.n_category.is_available and
            self.m_category.is_available
        )

    def display_full_stage(self) -> str:
        """Returns full TNM string or explicit disclaimer."""
        if not self.can_establish_stage():
            return "TNM stage cannot be established from the currently supplied information."
        prefix = "c" if "clinical" in self.staging_type.lower() else ("yp" if "yp" in self.staging_type.lower() else "p")
        stage_str = f"{prefix}T{self.t_category.value}N{self.n_category.value}M{self.m_category.value}"
        if self.overall_stage_group.is_available:
            stage_str += f" (Stage {self.overall_stage_group.value})"
        return stage_str

