"""
Automated Test Suite for Original Phase 12: Testing, Deployment, and Research Release.
Verifies security hardening, path traversal rejection, HTTP security headers,
TUM wording, baseline checkpoint preservation, CPU benchmark telemetry,
deployment configuration files, dataset cards, model cards, and documentation integrity.
"""

import asyncio
import hashlib
import io
import json
from pathlib import Path

import pytest
from fastapi import HTTPException, UploadFile
from starlette.requests import Request
from starlette.responses import Response

from app.core.config import settings
from app.db.models import PredictionRecord
from app.main import apply_security_headers, predict
from app.ml.inference import InferenceService
from app.research_workbench import (
    get_report_content,
    get_reports_list,
    get_visual_artifact,
)
from app.services import serialize_record

BASE_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = BASE_DIR / "docs"
RESEARCH_DIR = BASE_DIR / "research"


# ---------------------------------------------------------------------------
# 1. Baseline Checkpoint Invariance
# ---------------------------------------------------------------------------
def test_baseline_checkpoint_sha256_unmodified():
    """Verify baseline model checkpoint is bitwise identical to the authoritative hash."""
    checkpoint_path = BASE_DIR / "models" / "checkpoints" / "gastric_rnn_classifier.pt"
    assert checkpoint_path.exists(), f"Missing baseline checkpoint at {checkpoint_path}"

    hasher = hashlib.sha256()
    with checkpoint_path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)

    expected_sha256 = "65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8"
    assert hasher.hexdigest().lower() == expected_sha256, (
        f"Baseline model checkpoint SHA-256 altered! Expected {expected_sha256}, got {hasher.hexdigest()}"
    )


# ---------------------------------------------------------------------------
# 2. HTTP Security Headers
# ---------------------------------------------------------------------------
def test_security_headers_middleware():
    """Verify required HTTP security headers are set on responses."""
    dummy_response = Response("OK")
    response = apply_security_headers(dummy_response)

    headers = response.headers
    assert headers.get("X-Content-Type-Options") == "nosniff"
    assert headers.get("X-Frame-Options") == "DENY"
    assert headers.get("X-XSS-Protection") == "1; mode=block"
    assert "strict-origin-when-cross-origin" in headers.get("Referrer-Policy", "")
    assert "default-src 'self'" in headers.get("Content-Security-Policy", "")


# ---------------------------------------------------------------------------
# 3. Path Traversal Rejection on Report Retrieval
# ---------------------------------------------------------------------------
def test_report_path_traversal_rejection():
    """Verify that path traversal attempts on report endpoints are strictly rejected."""
    # Attempt directory traversal with ..
    with pytest.raises(HTTPException) as exc_info:
        get_report_content("../app/main.py")
    assert exc_info.value.status_code in (400, 404)

    with pytest.raises(HTTPException) as exc_info:
        get_report_content("../../etc/passwd")
    assert exc_info.value.status_code in (400, 404)

    # Attempt requesting a non-existent file
    with pytest.raises(HTTPException) as exc_info:
        get_report_content("nonexistent_secret.md")
    assert exc_info.value.status_code == 404

    # Valid report should succeed
    data = get_report_content("PHASE12_TESTING_DEPLOYMENT_RESEARCH_RELEASE_REPORT.md")
    assert "filename" in data
    assert "Phase 12" in data["content"]


# ---------------------------------------------------------------------------
# 4. Visual Artifact Allowlist & Path Traversal Guards
# ---------------------------------------------------------------------------
def test_visual_artifact_security_controls():
    """Verify category allowlisting, extension restrictions, and traversal protections on visual artifact endpoints."""
    # Invalid category
    with pytest.raises(HTTPException) as exc_info:
        get_visual_artifact("unauthorized_category", "plot.png")
    assert exc_info.value.status_code == 404

    # Traversal in category or filename
    with pytest.raises(HTTPException) as exc_info:
        get_visual_artifact("explainability", "../../main.py")
    assert exc_info.value.status_code in (400, 403, 404)

    # Disallowed extension (e.g. .py, .exe, .json)
    with pytest.raises(HTTPException) as exc_info:
        get_visual_artifact("explainability", "config.json")
    assert exc_info.value.status_code in (400, 404)

    # SVG extension disallowed to prevent script execution
    with pytest.raises(HTTPException) as exc_info:
        get_visual_artifact("explainability", "diagram.svg")
    assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# 5. Tumor Metric Wording & Scientific Disclaimers
# ---------------------------------------------------------------------------
def test_inference_tum_probability_and_disclaimer():
    """Verify inference results contain predicted_tum_probability and scientific disclaimer."""
    service = InferenceService()
    mock_probabilities = {
        "ADI": 0.05,
        "DEB": 0.05,
        "LYM": 0.10,
        "MUC": 0.05,
        "MUS": 0.05,
        "NOR": 0.05,
        "STR": 0.15,
        "TUM": 0.50,
    }
    result = service._format_result(mock_probabilities)

    assert "predicted_tum_probability" in result
    assert result["predicted_tum_probability"] == 50.0
    assert "disclaimer" in result
    assert "Predicted TUM class probability represents patch-level histological classification" in result["disclaimer"]
    assert "cancer_percentage" in result  # Backwards compatibility preserved


def test_database_serializer_includes_tum_and_disclaimer():
    """Verify serialized database records include predicted_tum_probability and disclaimer."""
    mock_record = PredictionRecord(
        id=999,
        filename="test_tile.png",
        predicted_class="TUM",
        predicted_label="Gastric Adenocarcinoma",
        cancer_percentage=85.0,
        risk_band="High",
        confidence_score=92.5,
        top_probability=92.5,
        model_source="test-model",
        class_probabilities_json=json.dumps([
            {"class_code": "TUM", "label": "Gastric Adenocarcinoma", "color": "#c4572e", "probability": 85.0},
            {"class_code": "STR", "label": "Cancer-Associated Stroma", "color": "#4f6d7a", "probability": 15.0},
        ]),
    )
    serialized = serialize_record(mock_record)
    assert serialized["predicted_tum_probability"] == 85.0
    assert "disclaimer" in serialized
    assert "patch-level histological classification" in serialized["disclaimer"]


# ---------------------------------------------------------------------------
# 6. Upload File Validation on predict()
# ---------------------------------------------------------------------------
def test_predict_disallows_invalid_file_extension(temp_db):
    """Verify predict rejects non-image file uploads."""
    fake_file = io.BytesIO(b"fake script content")
    upload = UploadFile(file=fake_file, filename="malicious_script.sh")
    with pytest.raises(HTTPException) as exc_info:
        predict(file=upload, db=temp_db)
    assert exc_info.value.status_code == 400
    assert "Unsupported file extension" in exc_info.value.detail


def test_predict_rejects_empty_file(temp_db):
    """Verify predict rejects 0-byte image files."""
    empty_file = io.BytesIO(b"")
    upload = UploadFile(file=empty_file, filename="empty.png")
    with pytest.raises(HTTPException) as exc_info:
        predict(file=upload, db=temp_db)
    assert exc_info.value.status_code == 400
    assert "empty" in exc_info.value.detail.lower()


def test_predict_rejects_corrupted_image(temp_db):
    """Verify predict decodes and rejects pseudo-image files with corrupted contents."""
    corrupt_bytes = io.BytesIO(b"\x89PNG\r\n\x1a\ncorrupted_trailing_payload_not_valid_image")
    upload = UploadFile(file=corrupt_bytes, filename="corrupt.png")
    with pytest.raises(HTTPException) as exc_info:
        predict(file=upload, db=temp_db)
    assert exc_info.value.status_code == 400
    assert "could not be decoded" in exc_info.value.detail.lower() or "corrupted" in exc_info.value.detail.lower()


def test_predict_rejects_oversized_file(temp_db):
    """Verify predict rejects files exceeding 10 MB."""
    large_data = b"0" * (10 * 1024 * 1024 + 1024)
    large_file = io.BytesIO(large_data)
    upload = UploadFile(file=large_file, filename="giant_image.png")
    with pytest.raises(HTTPException) as exc_info:
        predict(file=upload, db=temp_db)
    assert exc_info.value.status_code == 413
    assert "exceeds maximum permitted size" in exc_info.value.detail


# ---------------------------------------------------------------------------
# 7. Deployment Configuration Files Verification
# ---------------------------------------------------------------------------
def test_deployment_configuration_files_exist():
    """Verify that all production deployment configurations exist and are populated."""
    env_example = BASE_DIR / ".env.example"
    requirements = BASE_DIR / "requirements.txt"
    dockerfile = BASE_DIR / "Dockerfile"
    dockerignore = BASE_DIR / ".dockerignore"
    compose = BASE_DIR / "docker-compose.yml"

    for path in (env_example, requirements, dockerfile, dockerignore, compose):
        assert path.exists(), f"Missing deployment file: {path}"
        assert path.stat().st_size > 0, f"Empty deployment file: {path}"

    # Verify requirements.txt contains essential dependencies
    req_text = requirements.read_text(encoding="utf-8")
    for pkg in ("fastapi", "uvicorn", "torch", "scikit-learn", "numpy", "pandas"):
        assert pkg in req_text, f"Missing essential dependency {pkg} in requirements.txt"

    # Verify Dockerfile contains non-root user and healthcheck
    docker_text = dockerfile.read_text(encoding="utf-8")
    assert "useradd" in docker_text
    assert "USER gastricpath" in docker_text
    assert "HEALTHCHECK" in docker_text

    # Verify .dockerignore contains raw data exclusion
    ignore_text = dockerignore.read_text(encoding="utf-8")
    assert "data/" in ignore_text
    assert ".git" in ignore_text


# ---------------------------------------------------------------------------
# 8. Dataset Cards and Model Cards Presence
# ---------------------------------------------------------------------------
def test_dataset_cards_present_and_grounded():
    """Verify dataset cards are present and contain required scientific terms."""
    gchtid_card = DOCS_DIR / "datasets" / "GCHTID_DATASET_CARD.md"
    clinical_card = DOCS_DIR / "datasets" / "HMU_GC_CLINICAL_DATA_CARD.md"
    mirna_card = DOCS_DIR / "datasets" / "GASTRIC_MIRNA_TARGET_DATA_CARD.md"

    for card in (gchtid_card, clinical_card, mirna_card):
        assert card.exists(), f"Missing dataset card: {card}"

    clinical_text = clinical_card.read_text(encoding="utf-8")
    assert "IHC 2+/3+ exploratory category — not confirmed HER2-positive status" in clinical_text
    assert "Individual identifiers withheld from application outputs" in clinical_text
    assert "N = 300" in clinical_text

    mirna_text = mirna_card.read_text(encoding="utf-8")
    assert "In-silico miRNA target ranking" in mirna_text or "in-silico" in mirna_text.lower()
    assert "TargetScan" in mirna_text
    assert "SEMA6D" in mirna_text
    assert "11 unique target genes" in mirna_text


def test_model_cards_present():
    """Verify all 6 model cards are present in docs/models/ and verified against experimental artifacts."""
    expected_cards = [
        "GASTRIC_RNN_MODEL_CARD.md",
        "PHIKON_FOUNDATION_MODEL_CARD.md",
        "DINOV2_FOUNDATION_MODEL_CARD.md",
        "UNET_SEGMENTATION_MODEL_CARD.md",
        "CELLULAR_HED_WATERSHED_CARD.md",
        "SPATIAL_GNN_MODEL_CARD.md",
    ]
    for filename in expected_cards:
        card_path = DOCS_DIR / "models" / filename
        assert card_path.exists(), f"Missing model card: {card_path}"
        assert card_path.stat().st_size > 100, f"Model card too small: {card_path}"

    # Verify baseline model card reflects verified checkpoint val_accuracy and retraction
    rnn_text = (DOCS_DIR / "models" / "GASTRIC_RNN_MODEL_CARD.md").read_text(encoding="utf-8")
    assert "0.6486" in rnn_text or "64.86%" in rnn_text
    assert "88.22%" in rnn_text  # Documents retraction of ungrounded draft figure

    # Verify Phikon card reflects evaluation_summary.json metrics
    phikon_text = (DOCS_DIR / "models" / "PHIKON_FOUNDATION_MODEL_CARD.md").read_text(encoding="utf-8")
    assert "66.38%" in phikon_text
    assert "0.9450" in phikon_text

    # Verify DINOv2 card distinguishes ViT-S/14 from ViT-B/14
    dino_text = (DOCS_DIR / "models" / "DINOV2_FOUNDATION_MODEL_CARD.md").read_text(encoding="utf-8")
    assert "dinov2_vits14" in dino_text
    assert "dinov2_vitb14" in dino_text
    assert "Surveyed Only" in dino_text or "survey" in dino_text.lower()


def test_release_documentation_present():
    """Verify Reproducibility Guide, Validation & Limitations, Checklist, and Final Audit exist."""
    docs = [
        DOCS_DIR / "REPRODUCIBILITY.md",
        DOCS_DIR / "VALIDATION_AND_LIMITATIONS.md",
        DOCS_DIR / "PHASE12_TESTING_DEPLOYMENT_RESEARCH_RELEASE_REPORT.md",
        DOCS_DIR / "RESEARCH_RELEASE_CHECKLIST.md",
        DOCS_DIR / "FINAL_RELEASE_INTEGRITY_AUDIT.md",
    ]
    for doc in docs:
        assert doc.exists(), f"Missing release document: {doc}"
        assert doc.stat().st_size > 500, f"Release document unexpectedly small: {doc}"


# ---------------------------------------------------------------------------
# 9. CPU Benchmark Telemetry Integrity
# ---------------------------------------------------------------------------
def test_cpu_benchmark_results_artifact():
    """Verify CPU benchmark telemetry artifact exists and accurately documents performance."""
    bench_path = RESEARCH_DIR / "benchmarks" / "cpu_benchmark_results.json"
    assert bench_path.exists(), f"Missing CPU benchmark results at {bench_path}"

    with bench_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    perf = data.get("performance_metrics", {})
    assert "warm_mean_latency_ms" in perf
    assert perf["warm_mean_latency_ms"] > 0
    assert "throughput_samples_per_second" in perf
    assert perf["throughput_samples_per_second"] > 10.0

    mem = data.get("memory_metrics", {})
    assert "peak_rss_mb" in mem
    assert mem["peak_rss_mb"] > 50.0

    gpu = data.get("gpu_environment", {})
    disclosure = gpu.get("scientific_disclosure", "")
    assert "cuda" in disclosure.lower() or "not benchmarked" in disclosure.lower() or not gpu.get("cuda_available", True)


# ---------------------------------------------------------------------------
# 10. Workbench Catalog Completeness
# ---------------------------------------------------------------------------
def test_workbench_reports_catalog_includes_phase12():
    """Verify workbench reports endpoint includes Phase 12, release docs, and final integrity audit."""
    catalog = get_reports_list()
    reports = catalog["reports"]
    filenames = {r["filename"] for r in reports}

    assert "FINAL_RELEASE_INTEGRITY_AUDIT.md" in filenames
    assert "PHASE12_TESTING_DEPLOYMENT_RESEARCH_RELEASE_REPORT.md" in filenames
    assert "REPRODUCIBILITY.md" in filenames
    assert "VALIDATION_AND_LIMITATIONS.md" in filenames
    assert "RESEARCH_RELEASE_CHECKLIST.md" in filenames
    assert len(reports) >= 16
