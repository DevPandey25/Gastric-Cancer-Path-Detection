"""
Comprehensive Unit & Regression Test Suite for Clinical Research & PDF Workbench.
GastricPath AI — Phase 12: Production Verification & Regression Testing.

Covers:
1. Pydantic clinical schemas & missingness serialization
2. Evidence retrieval & 32-paper literature search engine
3. Jurisdictional biomarker companion diagnostics
4. Safety invariants (curative-intent distinction, uncalibrated optics)
5. Security & privacy guards (path traversal, PHI redaction)
6. Multi-mode PDF generation (Modes A, B, C)
7. FastAPI REST endpoints (Direct Function Invocations)
"""

from pathlib import Path
import pytest

from app.clinical_reporting_api import (
    get_reporting_status,
    get_case_template,
    validate_case,
    list_guidelines,
    get_biomarker_info,
    search_literature,
    download_pdf
)
from research.clinical_reporting.schemas.case_dossier import CaseDossier
from research.clinical_reporting.schemas.patient_case import PatientCase
from research.clinical_reporting.schemas.pathology import HistopathologyReport
from research.clinical_reporting.schemas.biomarkers import MolecularBiomarkers
from research.clinical_reporting.schemas.staging import StagingClassification
from research.clinical_reporting.schemas.common import ClinicalField, VerificationStatus
from research.clinical_reporting.ingestion.document_import import create_blank_case_template
from research.clinical_reporting.evidence.source_registry import (
    GUIDELINE_REGISTRY, PATHOLOGY_STANDARDS, REPORTING_STANDARDS,
    BIOMARKER_STANDARDS, Jurisdiction, get_biomarker_context
)
from research.clinical_reporting.evidence.literature_search import LiteratureSearchEngine
from research.clinical_reporting.evidence.alignment_matrix import GuidelineAlignmentBuilder
from research.clinical_reporting.report_engine.report_validator import ReportValidator
from research.clinical_reporting.security.file_validation import (
    sanitize_filename, validate_file_path, validate_image_file, SecurityValidationError
)
from research.clinical_reporting.security.privacy_guard import (
    redact_direct_identifiers, verify_patient_data_local_only, PrivacyViolationError, IDENTIFIER_PATTERNS
)
from research.clinical_reporting.report_engine.case_review_generator import generate_case_review_dossier_pdf
from research.clinical_reporting.report_engine.image_atlas_generator import generate_image_atlas_pdf
from research.clinical_reporting.report_engine.manuscript_generator import generate_research_manuscript_pdf


# ==============================================================================
# 1. SCHEMA & DATA MODEL TESTS
# ==============================================================================

def test_patient_case_schema_defaults():
    patient = PatientCase(case_id="GP-TEST-001")
    assert patient.case_id == "GP-TEST-001"
    assert patient.age.is_available is False
    assert patient.age.display() == "[Not available]"
    assert patient.validate_no_direct_identifiers() is True


def test_histopathology_report_synoptic_schema():
    pathology = HistopathologyReport(
        lauren_classification=ClinicalField(value="Intestinal", verification_status=VerificationStatus.DOCUMENTED)
    )
    assert pathology.lauren_classification.value == "Intestinal"
    assert pathology.lymphovascular_invasion.is_available is False
    assert pathology.perineural_invasion.display() == "[Not available]"


def test_case_dossier_serialization():
    dossier = create_blank_case_template("GP-SERIAL-01")
    data = dossier.model_dump()
    assert data["patient"]["case_id"] == "GP-SERIAL-01"
    assert "pathology" in data
    assert "biomarkers" in data
    assert "staging" in data


# ==============================================================================
# 2. EVIDENCE & LITERATURE RETRIEVAL TESTS
# ==============================================================================

def test_authoritative_guidelines_registry():
    assert "ESMO_2026" in GUIDELINE_REGISTRY
    assert "ASCO_2026" in GUIDELINE_REGISTRY
    assert "WHO_6TH_ED_2026" in PATHOLOGY_STANDARDS
    assert "CAP_STOMACH_2026" in PATHOLOGY_STANDARDS


def test_jurisdictional_biomarker_standards():
    ctx_her2 = get_biomarker_context("HER2", Jurisdiction.INDIA)
    assert ctx_her2["biomarker"] == "HER2 / ERBB2"
    assert "method" in ctx_her2
    assert "local_approval" in ctx_her2

    ctx_us = get_biomarker_context("HER2", Jurisdiction.US)
    assert "local_approval" in ctx_us


def test_literature_search_engine():
    lit_engine = LiteratureSearchEngine()
    assert len(lit_engine.papers) >= 30, f"Expected at least 30 papers, found {len(lit_engine.papers)}"

    results = lit_engine.search("HMU-GC-HE-30K")
    assert len(results) > 0
    assert any("Lou" in p.get("Authors", "") for p in results)

    results_phikon = lit_engine.search("Phikon")
    assert len(results_phikon) > 0


def test_alignment_matrix_and_checklist():
    dossier = create_blank_case_template("GP-TEST-ALIGN")
    matrix = GuidelineAlignmentBuilder.build_alignment_matrix(dossier)
    assert len(matrix) >= 4

    checklist = GuidelineAlignmentBuilder.build_mdt_checklist(dossier)
    assert len(checklist) == 10
    assert all("item" in c and "status" in c for c in checklist)


# ==============================================================================
# 3. SAFETY INVARIANTS & PRE-FLIGHT VALIDATION
# ==============================================================================

def test_report_validator_curative_intent_enforcement():
    dossier = create_blank_case_template("GP-SAFETY-001")
    validation = ReportValidator.validate_case_dossier(dossier)
    assert "is_valid" in validation
    assert "validation_status" in validation
    assert isinstance(validation["warnings"], list)


def test_cure_claim_blocked_in_validator():
    dossier = create_blank_case_template("GP-CURE-FAIL")
    dossier.treatment.treatment_intent_documented = ClinicalField(
        value="Guaranteed complete cure for patient",
        verification_status=VerificationStatus.DOCUMENTED
    )
    validation = ReportValidator.validate_case_dossier(dossier)
    assert validation["is_valid"] is False
    assert any("cure rather than curative intent" in issue for issue in validation["blocking_issues"])


# ==============================================================================
# 4. SECURITY & PRIVACY GUARDS
# ==============================================================================

def test_file_validation_path_traversal():
    with pytest.raises(SecurityValidationError):
        validate_file_path(base_dir=Path("data"), target_path=Path("../../../etc/passwd"))

    with pytest.raises(SecurityValidationError):
        validate_image_file(file_path=Path("malicious.exe"))


def test_filename_sanitization():
    unsafe = "test/..\\case:report*<>.pdf"
    clean = sanitize_filename(unsafe)
    assert ".." not in clean
    assert "/" not in clean
    assert "\\" not in clean
    assert clean.endswith(".pdf")


def test_privacy_guard_phi_redaction():
    text_with_phi = "Patient John Doe with SSN 123-45-6789 and MRN 987654321 was seen."
    redacted = redact_direct_identifiers(text_with_phi)
    assert "123-45-6789" not in redacted
    assert "REDACTED" in redacted


def test_verify_patient_data_local_only():
    safe_payload = {"case_id": "GP-001", "patient": {"age": 55}}
    assert verify_patient_data_local_only(safe_payload) is True

    unsafe_payload = {"case_id": "GP-001", "remote_sync": True}
    with pytest.raises(PrivacyViolationError):
        verify_patient_data_local_only(unsafe_payload)


# ==============================================================================
# 5. MULTI-MODE PDF GENERATION TESTS
# ==============================================================================

def test_generate_case_review_dossier_pdf_mode_a(tmp_path):
    out_pdf = tmp_path / "test_case_review.pdf"
    dossier = create_blank_case_template("GP-UNIT-TEST-A")
    generate_case_review_dossier_pdf(dossier=dossier, output_pdf_path=out_pdf)

    assert out_pdf.exists()
    assert out_pdf.stat().st_size > 5000
    with open(out_pdf, "rb") as f:
        header = f.read(5)
        assert header == b"%PDF-"


def test_generate_image_atlas_pdf_mode_b(tmp_path):
    out_pdf = tmp_path / "test_image_atlas.pdf"
    generate_image_atlas_pdf(sample_items=[], output_pdf_path=out_pdf)

    assert out_pdf.exists()
    assert out_pdf.stat().st_size > 5000
    with open(out_pdf, "rb") as f:
        header = f.read(5)
        assert header == b"%PDF-"


def test_generate_research_manuscript_pdf_mode_c(tmp_path):
    out_pdf = tmp_path / "test_manuscript.pdf"
    generate_research_manuscript_pdf(output_pdf_path=out_pdf)

    assert out_pdf.exists()
    assert out_pdf.stat().st_size > 5000
    with open(out_pdf, "rb") as f:
        header = f.read(5)
        assert header == b"%PDF-"


# ==============================================================================
# 6. FASTAPI REST ENDPOINTS (DIRECT FUNCTION TESTS)
# ==============================================================================

def test_api_reporting_status():
    data = get_reporting_status()
    assert data["status"] == "active"
    assert "ReportLab" in data["engine"]
    assert "supported_modes" in data


def test_api_reporting_template():
    data = get_case_template()
    assert "patient" in data
    assert data["patient"]["case_id"] == "GP-TEMPLATE-01"


def test_api_reporting_validate():
    template = create_blank_case_template("GP-API-VALIDATE-01")
    data = validate_case(template)
    assert "is_valid" in data
    assert "validation_status" in data


def test_api_reporting_guidelines():
    data = list_guidelines()
    assert isinstance(data, list)
    assert any(g["source_id"] == "ESMO_2026" for g in data)


def test_api_reporting_biomarkers():
    data = get_biomarker_info(biomarker="HER2", jurisdiction="India")
    assert "method" in data
    assert "local_approval" in data


def test_api_reporting_search():
    data = search_literature(q="HMU", limit=5)
    assert isinstance(data, list)
    assert len(data) > 0


def test_api_reporting_download_existing():
    response = download_pdf("GastricPath_AI_Model_Evaluation_Report.pdf")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
