"""
Molecular and Predictive Biomarker Reporting Schema.
GastricPath AI — Phase 2: Clinical Information Schema.

Captures HER2 (IHC/ISH), PD-L1 (CPS), MMR/MSI, and CLDN18.2 with companion diagnostic assay metadata.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field
from research.clinical_reporting.schemas.common import ClinicalField, VerificationStatus


class HER2Biomarker(BaseModel):
    test_method: str = "Immunohistochemistry (IHC) with reflex ISH"
    ihc_score: ClinicalField = Field(
        default_factory=lambda: ClinicalField(
            verification_status=VerificationStatus.NOT_AVAILABLE,
            notes="Hofmann criteria: 0, 1+, 2+, 3+"
        )
    )
    ish_status: ClinicalField = Field(
        default_factory=lambda: ClinicalField(
            verification_status=VerificationStatus.NOT_APPLICABLE,
            notes="Positive if HER2:CEP17 ratio >= 2.0 or HER2 copy >= 6.0"
        )
    )
    overall_her2_status: ClinicalField = Field(
        default_factory=lambda: ClinicalField(
            verification_status=VerificationStatus.NOT_AVAILABLE,
            notes="HER2 Positive: IHC 3+ or IHC 2+/ISH+; HER2 Negative: IHC 0, 1+, or IHC 2+/ISH-"
        )
    )
    assay_vendor: str = "PATHWAY anti-HER2/neu (4B5) / HercepTest"
    testing_laboratory: str = "Clinical Pathology / Molecular Diagnostics"


class PDL1Biomarker(BaseModel):
    test_method: str = "Immunohistochemistry (IHC)"
    assay_clone: str = "Dako 28-8 pharmDx / 22C3 pharmDx"
    cps_score: ClinicalField = Field(
        default_factory=lambda: ClinicalField(
            verification_status=VerificationStatus.NOT_AVAILABLE,
            notes="Combined Positive Score (numerical value 0-100)"
        )
    )
    interpretation_status: ClinicalField = Field(
        default_factory=lambda: ClinicalField(
            verification_status=VerificationStatus.NOT_AVAILABLE,
            notes="ESMO CheckMate-649 threshold: CPS >= 5; ASCO/FDA threshold: CPS >= 1 or >= 10"
        )
    )
    testing_laboratory: str = "Molecular Pathology Laboratory"


class MMRBiomarker(BaseModel):
    test_method: str = "Immunohistochemistry (4-antibody panel) / MSI-PCR"
    mlh1_status: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE, notes="Intact / Lost"))
    pms2_status: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE, notes="Intact / Lost"))
    msh2_status: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE, notes="Intact / Lost"))
    msh6_status: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_AVAILABLE, notes="Intact / Lost"))
    overall_mmr_status: ClinicalField = Field(
        default_factory=lambda: ClinicalField(
            verification_status=VerificationStatus.NOT_AVAILABLE,
            notes="pMMR (Proficient) / dMMR (Deficient)"
        )
    )
    msi_status: ClinicalField = Field(
        default_factory=lambda: ClinicalField(
            verification_status=VerificationStatus.NOT_AVAILABLE,
            notes="MSS (Microsatellite Stable) / MSI-H (High)"
        )
    )


class Claudin18Biomarker(BaseModel):
    test_method: str = "Immunohistochemistry (IHC)"
    assay_name: str = "VENTANA CLDN18 (43-14A) RxDx Assay"
    percent_tumor_stained: ClinicalField = Field(
        default_factory=lambda: ClinicalField(
            unit="%",
            verification_status=VerificationStatus.NOT_AVAILABLE,
            notes="Percentage of viable tumor cells with moderate-to-strong membranous staining"
        )
    )
    staining_intensity: ClinicalField = Field(
        default_factory=lambda: ClinicalField(
            verification_status=VerificationStatus.NOT_AVAILABLE,
            notes="0 (Negative), 1+ (Weak), 2+ (Moderate), 3+ (Strong)"
        )
    )
    overall_status: ClinicalField = Field(
        default_factory=lambda: ClinicalField(
            verification_status=VerificationStatus.NOT_AVAILABLE,
            notes="SPOTLIGHT / GLOW criteria: Positive if >= 75% cells with >= 2+ intensity"
        )
    )


class MolecularBiomarkers(BaseModel):
    """Unified biomarker panel supporting companion diagnostics."""
    her2: HER2Biomarker = Field(default_factory=HER2Biomarker)
    pdl1: PDL1Biomarker = Field(default_factory=PDL1Biomarker)
    mmr: MMRBiomarker = Field(default_factory=MMRBiomarker)
    claudin18: Claudin18Biomarker = Field(default_factory=Claudin18Biomarker)
    
    # Optional / exploratory genomic data
    ebv_status: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_ASSESSED, notes="EBER In-situ hybridization"))
    tmb_score: ClinicalField = Field(default_factory=lambda: ClinicalField(unit="mut/Mb", verification_status=VerificationStatus.NOT_ASSESSED))
    ntrk_fusion: ClinicalField = Field(default_factory=lambda: ClinicalField(verification_status=VerificationStatus.NOT_ASSESSED))
    ngs_actionable_variants: list[str] = Field(default_factory=list)

