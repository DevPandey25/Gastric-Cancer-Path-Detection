"""
Common Clinical Information Schemas and Value Types.
GastricPath AI — Phase 2: Clinical Information Schema.

Ensures every medical data field preserves its verification status, provenance,
units, and explicit missingness indicators.
"""

from datetime import date
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class VerificationStatus(str, Enum):
    DOCUMENTED = "Documented"
    CLINICIAN_VERIFIED = "Clinician verified"
    PATIENT_REPORTED = "Patient reported"
    IMPORTED_FROM_SOURCE = "Imported from source document"
    MODEL_GENERATED = "Model generated"
    UNKNOWN = "Unknown"
    NOT_AVAILABLE = "Not available"
    NOT_ASSESSED = "Not assessed"
    NOT_APPLICABLE = "Not applicable"
    CONFLICTING = "Conflicting information"


class ClinicalField(BaseModel):
    """Encapsulates a clinical value with provenance, unit, and verification status."""
    value: Any = None
    unit: Optional[str] = None
    source: str = "Clinical Record"
    collection_date: Optional[str] = None
    verification_status: VerificationStatus = VerificationStatus.NOT_AVAILABLE
    reviewer: Optional[str] = None
    notes: Optional[str] = None

    @property
    def is_available(self) -> bool:
        return self.verification_status not in (
            VerificationStatus.UNKNOWN,
            VerificationStatus.NOT_AVAILABLE,
            VerificationStatus.NOT_ASSESSED,
            VerificationStatus.NOT_APPLICABLE
        ) and self.value is not None

    def display(self) -> str:
        """Renders value and unit or explicit missingness status."""
        if not self.is_available:
            return f"[{self.verification_status.value}]"
        unit_str = f" {self.unit}" if self.unit else ""
        return f"{self.value}{unit_str}"


class TreatmentIntent(str, Enum):
    CURATIVE = "Curative intent documented by treating team"
    DISEASE_CONTROL = "Disease-control intent documented"
    PALLIATIVE = "Palliative intent documented"
    NOT_DOCUMENTED = "Not documented in supplied records"
    NEEDS_CLARIFICATION = "Requires oncology-team clarification"

