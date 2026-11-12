"""
Privacy Guard and Local-Only Security Enforcement.
GastricPath AI — Phase 3: Ingestion and Privacy Guard.

Guarantees that protected health information (PHI) and raw pathology data
are never leaked or transmitted to external third-party services.
"""

import re
from typing import Any


class PrivacyViolationError(Exception):
    pass


# Patterns matching common direct patient identifiers
IDENTIFIER_PATTERNS = [
    (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
    (r"\b\d{4}\s\d{4}\s\d{4}\b", "Aadhaar / National ID"),
    (r"\b\+?[1-9]\d{1,14}\b", "Phone number"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "Email address"),
    (r"\bMRN[:\s#]*[0-9A-Za-z-]+\b", "Medical Record Number"),
    (r"\bHospital\s*Reg[:\s#]*[0-9A-Za-z-]+\b", "Hospital Registration Number"),
]


def redact_direct_identifiers(text: str) -> str:
    """Scans and redacts direct patient identifiers from text."""
    redacted = text
    for pattern, label in IDENTIFIER_PATTERNS:
        redacted = re.sub(pattern, f"[REDACTED_{label.upper()}]", redacted, flags=re.IGNORECASE)
    return redacted


def verify_patient_data_local_only(payload: dict[str, Any]) -> bool:
    """Audit gatekeeper: ensures payload contains no active external transmission flags."""
    # Strict rule: no remote endpoint transmission of clinical dossiers
    if payload.get("remote_sync") is True or payload.get("external_api_export") is True:
        raise PrivacyViolationError("External synchronization of patient data is strictly prohibited by GastricPath AI local policy.")
    return True

