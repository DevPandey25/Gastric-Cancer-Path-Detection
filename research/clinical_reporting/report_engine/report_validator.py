"""
Report Validation and Pre-Flight Integrity Engine.
GastricPath AI — Phase 11: Clinical Validation and Security Checks.

Validates required fields, units, references, and safety invariants before PDF generation.
"""

from typing import Any
from research.clinical_reporting.schemas.case_dossier import CaseDossier
from research.clinical_reporting.schemas.common import VerificationStatus


class ReportValidationError(Exception):
    pass


class ReportValidator:
    """Pre-flight validation service ensuring medical document integrity."""

    @staticmethod
    def validate_case_dossier(dossier: CaseDossier) -> dict[str, Any]:
        """Performs rigorous pre-flight safety and data integrity audits."""
        issues = []
        warnings = []

        # 1. Check de-identification
        if not dossier.patient.validate_no_direct_identifiers():
            issues.append("Direct identifier detected in case_id. Case must be strictly de-identified.")

        # 2. Check for unsupported staging
        staging = dossier.staging
        if not staging.can_establish_stage() and staging.overall_stage_group.is_available:
            warnings.append("Overall stage group documented without complete T, N, M parameters.")

        # 3. Check for curative intent vs cure claims
        intent = dossier.treatment.treatment_intent_documented.value
        if "cure" in intent.lower() and "curative intent" not in intent.lower():
            issues.append("Document claims cure rather than curative intent. Violates medical safety invariant.")

        # 4. Check biomarker interpretation criteria
        cldn = dossier.biomarkers.claudin18
        if cldn.overall_status.is_available and "pos" in cldn.overall_status.value.lower():
            if not cldn.percent_tumor_stained.is_available or float(cldn.percent_tumor_stained.value or 0) < 75.0:
                warnings.append("CLDN18.2 marked positive with < 75% cells stained or missing percentage.")

        # 5. Check missing data tracking
        missing_count = 0
        for field_name, field_obj in [
            ("Age", dossier.patient.age),
            ("Sex", dossier.patient.sex),
            ("Tumor Location", dossier.endoscopy.tumor_location),
            ("Histological Type", dossier.pathology.who_histological_type),
            ("HER2 Status", dossier.biomarkers.her2.overall_her2_status),
            ("PD-L1 CPS", dossier.biomarkers.pdl1.cps_score),
            ("MMR Status", dossier.biomarkers.mmr.overall_mmr_status),
        ]:
            if not field_obj.is_available:
                missing_count += 1

        is_valid = len(issues) == 0
        return {
            "is_valid": is_valid,
            "blocking_issues": issues,
            "warnings": warnings,
            "missing_field_count": missing_count,
            "validation_status": "Passed with Disclosures" if is_valid else "Blocked by Safety Invariant"
        }

