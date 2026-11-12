"""
Clinical Research & PDF Workbench API Router.
GastricPath AI — Phase 10: Dashboard Integration.

Exposes REST endpoints for clinical case management, evidence retrieval,
pre-flight validation, and multi-mode PDF generation.
"""

from pathlib import Path
from typing import Any, Optional
from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from research.clinical_reporting.schemas.case_dossier import CaseDossier
from research.clinical_reporting.ingestion.document_import import create_blank_case_template
from research.clinical_reporting.report_engine.report_validator import ReportValidator
from research.clinical_reporting.report_engine.case_review_generator import generate_case_review_dossier_pdf
from research.clinical_reporting.report_engine.image_atlas_generator import generate_image_atlas_pdf
from research.clinical_reporting.report_engine.manuscript_generator import generate_research_manuscript_pdf
from research.clinical_reporting.evidence.source_registry import (
    GUIDELINE_REGISTRY, PATHOLOGY_STANDARDS, REPORTING_STANDARDS,
    BIOMARKER_STANDARDS, Jurisdiction, get_biomarker_context
)
from research.clinical_reporting.evidence.literature_search import LiteratureSearchEngine
from research.clinical_reporting.security.file_validation import sanitize_filename, validate_file_path, SecurityValidationError

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "research" / "clinical_reporting" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter(prefix="/api/reporting", tags=["clinical-reporting"])
lit_engine = LiteratureSearchEngine()


@router.get("/status")
def get_reporting_status() -> dict[str, Any]:
    """Returns reporting workbench status, supported modes, and guideline cutoff."""
    return {
        "status": "active",
        "engine": "ReportLab Platypus Two-Pass A4 Engine",
        "literature_cutoff": "November 12, 2026",
        "supported_modes": [
            {
                "mode_id": "mode_a",
                "title": "Clinical Case Review Dossier (Mode A)",
                "description": "20-section structured case review with CAP synoptics, guideline alignment matrix, and MDT checklist."
            },
            {
                "mode_id": "mode_b",
                "title": "Histopathology Image Atlas (Mode B)",
                "description": "Image-centered multi-panel figures with genuine patches, Grad-CAM heatmaps, and probability distributions."
            },
            {
                "mode_id": "mode_c",
                "title": "Gastric Cancer Research Manuscript (Mode C)",
                "description": "Publication-grade manuscript reporting duplicate audit, foundation model pooling, and cell graph benchmarks."
            }
        ],
        "guidelines_monitored": len(GUIDELINE_REGISTRY) + len(PATHOLOGY_STANDARDS),
        "literature_records": len(lit_engine.papers),
        "output_directory": str(OUTPUT_DIR)
    }


@router.get("/case/template")
def get_case_template() -> dict[str, Any]:
    """Returns a blank de-identified case template with explicit missingness indicators."""
    template = create_blank_case_template(case_id="GP-TEMPLATE-01")
    return template.model_dump()


@router.post("/case/validate")
def validate_case(dossier: CaseDossier) -> dict[str, Any]:
    """Validates a clinical case dossier against medical safety and privacy rules."""
    return ReportValidator.validate_case_dossier(dossier)


@router.get("/evidence/guidelines")
def list_guidelines(jurisdiction: Optional[str] = None) -> list[dict[str, Any]]:
    """Returns authoritative clinical guidelines, optionally filtered by jurisdiction."""
    results = []
    for g in GUIDELINE_REGISTRY.values():
        if not jurisdiction or g.jurisdiction.value.lower() == jurisdiction.lower() or g.jurisdiction == Jurisdiction.GLOBAL:
            results.append({
                "source_id": g.source_id,
                "title": g.title,
                "organization": g.organization,
                "publication_date": g.publication_date,
                "version": g.version_or_edition,
                "url": g.url,
                "jurisdiction": g.jurisdiction.value,
                "summary": g.summary,
                "key_recommendations": g.key_recommendations
            })
    return results


@router.get("/evidence/biomarkers")
def get_biomarker_info(biomarker: str = "HER2", jurisdiction: str = "Global / General") -> dict[str, Any]:
    """Returns companion diagnostic criteria and jurisdictional approval status for a biomarker."""
    jur_enum = Jurisdiction.GLOBAL
    for j in Jurisdiction:
        if j.value.lower() == jurisdiction.lower():
            jur_enum = j
            break
    return get_biomarker_context(biomarker, jur_enum)


@router.get("/evidence/search")
def search_literature(q: str = Query(..., min_length=2), limit: int = 5) -> list[dict[str, Any]]:
    """Searches the verified 32-publication literature database."""
    return lit_engine.search(q, limit=limit)


@router.post("/generate/case-dossier")
def generate_dossier(dossier: Optional[CaseDossier] = None) -> dict[str, Any]:
    """Generates Mode A Clinical Case Review Dossier PDF."""
    if dossier is None:
        dossier = create_blank_case_template("GP-GC-2026-DEMO")

    validation = ReportValidator.validate_case_dossier(dossier)
    if not validation["is_valid"]:
        raise HTTPException(status_code=400, detail={"message": "Case validation failed", "issues": validation["blocking_issues"]})
    
    clean_id = sanitize_filename(dossier.patient.case_id)
    pdf_filename = f"GastricPath_Case_Dossier_{clean_id}.pdf"
    pdf_path = OUTPUT_DIR / pdf_filename
    generate_case_review_dossier_pdf(dossier, pdf_path)
    
    return {
        "status": "success",
        "mode": "Mode A: Clinical Case Review Dossier",
        "filename": pdf_filename,
        "download_url": f"/api/reporting/download/{pdf_filename}",
        "file_size_bytes": pdf_path.stat().st_size,
        "validation_summary": validation
    }


@router.post("/generate/image-atlas")
def generate_atlas(limit_plates: int = 8) -> dict[str, Any]:
    """Generates Mode B Histopathology Image Atlas PDF from genuine repository patches."""
    all_img_dir = BASE_DIR / "data" / "raw" / "HMU-GC-HE-30K" / "all_image"
    attr_dir = BASE_DIR / "research" / "explainability" / "attribution_maps"
    classes = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]
    sample_items = []

    for c in classes[:limit_plates]:
        class_dir = all_img_dir / c
        if class_dir.exists():
            imgs = sorted(list(class_dir.glob("*.png")))
            if imgs:
                attr_matches = list(attr_dir.glob(f"*{c}*.png"))
                attr_p = attr_matches[0] if attr_matches else None
                sample_items.append({
                    "image_path": imgs[0],
                    "sample_id": f"{c}-PLATE-01",
                    "known_class": c,
                    "attribution_path": attr_p
                })

    pdf_filename = "GastricPath_AI_Research_Image_Atlas.pdf"
    pdf_path = OUTPUT_DIR / pdf_filename
    generate_image_atlas_pdf(sample_items, pdf_path)

    return {
        "status": "success",
        "mode": "Mode B: Histopathology Image Atlas",
        "filename": pdf_filename,
        "download_url": f"/api/reporting/download/{pdf_filename}",
        "file_size_bytes": pdf_path.stat().st_size,
        "plates_rendered": len(sample_items)
    }


@router.post("/generate/manuscript")
def generate_manuscript() -> dict[str, Any]:
    """Generates Mode C Scientific Research Manuscript PDF."""
    pdf_filename = "GastricPath_AI_Model_Evaluation_Report.pdf"
    pdf_path = OUTPUT_DIR / pdf_filename
    generate_research_manuscript_pdf(pdf_path)

    return {
        "status": "success",
        "mode": "Mode C: Scientific Research Manuscript",
        "filename": pdf_filename,
        "download_url": f"/api/reporting/download/{pdf_filename}",
        "file_size_bytes": pdf_path.stat().st_size
    }


@router.get("/download/{filename}")
def download_pdf(filename: str):
    """Safely downloads a generated report PDF from allowlisted output directory."""
    clean_name = sanitize_filename(filename)
    if not clean_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files may be downloaded.")
        
    target_path = OUTPUT_DIR / clean_name
    try:
        validate_file_path(OUTPUT_DIR, target_path)
    except SecurityValidationError:
        raise HTTPException(status_code=403, detail="Access denied: invalid file path.")
        
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"File {clean_name} not found.")

    return FileResponse(
        path=str(target_path),
        filename=clean_name,
        media_type="application/pdf"
    )
