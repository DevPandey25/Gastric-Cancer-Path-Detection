"""Automated tests for Phase 4.2.1: Scientific Reporting & Linear Probe Evaluation Audit."""

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = WORKSPACE_ROOT / "research" / "foundation_models" / "evaluation"
AUDIT_DIR = WORKSPACE_ROOT / "research" / "audits" / "phase4_2_1"
DOCS_DIR = WORKSPACE_ROOT / "docs"
BASELINE_CHECKPOINT_PATH = WORKSPACE_ROOT / "models" / "checkpoints" / "gastric_rnn_classifier.pt"
BASELINE_CHECKPOINT_SHA256 = "65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8"
CANONICAL_CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]


def test_audit_metrics_json_exists_and_verified():
    """Verify that audit_metrics.json exists and contains recomputed metrics matching reported numbers."""
    audit_json_path = AUDIT_DIR / "audit_metrics.json"
    assert audit_json_path.exists(), f"Missing audit file: {audit_json_path}"

    with open(audit_json_path, "r") as f:
        audit_data = json.load(f)

    for repr_type in ["cls", "mean_patch"]:
        assert repr_type in audit_data, f"Missing {repr_type} in audit data"
        # Zero significant discrepancies allowed
        discrepancies = audit_data[repr_type]["discrepancies"]
        assert len(discrepancies) == 0, f"Found unexpected discrepancies in {repr_type}: {discrepancies}"

        recomputed_ov = audit_data[repr_type]["recomputed_overall"]
        assert recomputed_ov["total_samples"] == 4665
        assert 0.60 <= recomputed_ov["accuracy"] <= 0.70
        assert 0.60 <= recomputed_ov["macro_f1"] <= 0.70
        assert recomputed_ov["macro_auroc"] >= 0.90


def test_canonical_class_ordering_and_support():
    """Verify test prediction CSVs preserve canonical class columns and probability normalization."""
    for repr_type in ["cls", "mean_patch"]:
        csv_path = EVAL_DIR / f"linear_probe_{repr_type}_test_predictions.csv"
        assert csv_path.exists(), f"Missing CSV: {csv_path}"

        df = pd.read_csv(csv_path)
        assert len(df) == 4665, f"Expected 4665 rows, got {len(df)}"

        # Check canonical probability columns
        prob_cols = [f"prob_{c}" for c in CANONICAL_CLASSES]
        for col in prob_cols:
            assert col in df.columns, f"Column {col} missing in {csv_path.name}"

        # Check probability normalization (sum to 1.0 within CSV rounding tolerance)
        prob_matrix = df[prob_cols].to_numpy()
        sums = np.sum(prob_matrix, axis=1)
        assert np.all(np.abs(sums - 1.0) < 1e-3), "Row probabilities do not sum to 1.0"


def test_per_class_table_reconciliation():
    """Verify that all 8 canonical classes exist in recomputed per-class metrics with valid supports."""
    audit_json_path = AUDIT_DIR / "audit_metrics.json"
    with open(audit_json_path, "r") as f:
        audit_data = json.load(f)

    expected_support = {
        "ADI": 583,
        "DEB": 583,
        "LYM": 583,
        "MUC": 584,
        "MUS": 583,
        "NOR": 583,
        "STR": 583,
        "TUM": 583,
    }

    for repr_type in ["cls", "mean_patch"]:
        per_class = audit_data[repr_type]["recomputed_per_class"]
        for c, sup in expected_support.items():
            assert c in per_class, f"Class {c} missing in {repr_type} per-class metrics"
            assert per_class[c]["support"] == sup, f"Support mismatch for {c}"
            assert 0.40 <= per_class[c]["precision"] <= 1.0
            assert 0.40 <= per_class[c]["recall_sensitivity"] <= 1.0
            assert 0.90 <= per_class[c]["specificity"] <= 1.0
            assert 0.40 <= per_class[c]["f1_score"] <= 1.0


def test_deb_metric_algebraic_identity():
    """Verify that DEB in Mean Patch has exact precision, recall, and F1 equality due to FP==FN."""
    audit_json_path = AUDIT_DIR / "audit_metrics.json"
    with open(audit_json_path, "r") as f:
        audit_data = json.load(f)

    cm = np.array(audit_data["mean_patch"]["confusion_matrix"])
    deb_idx = CANONICAL_CLASSES.index("DEB")  # Index 1

    tp = cm[deb_idx, deb_idx]
    fn = np.sum(cm[deb_idx, :]) - tp
    fp = np.sum(cm[:, deb_idx]) - tp

    assert tp == 339, f"Expected 339 TP for DEB, got {tp}"
    assert fn == 244, f"Expected 244 FN for DEB, got {fn}"
    assert fp == 244, f"Expected 244 FP for DEB, got {fp}"
    assert fn == fp, "FP and FN are not equal for DEB!"

    deb_metrics = audit_data["mean_patch"]["recomputed_per_class"]["DEB"]
    assert deb_metrics["precision"] == deb_metrics["recall_sensitivity"] == deb_metrics["f1_score"] == 0.5815


def test_audit_report_and_documentation_exists():
    """Verify that Phase 4.2.1 audit report exists and contains essential sections."""
    audit_report = DOCS_DIR / "PHASE4_2_1_METRIC_AND_EVALUATION_AUDIT.md"
    assert audit_report.exists(), "PHASE4_2_1_METRIC_AND_EVALUATION_AUDIT.md missing!"

    content = audit_report.read_text(encoding="utf-8")
    assert "GP-AUDIT-PHASE4-2-1-2026-V1" in content
    assert "Final Probe Implementation Verification" in content
    assert "Test Evaluation History & Execution Lineage Audit" in content
    assert "Independent Metric Recomputation & Verification" in content
    assert "Corrected Per-Class Benchmark Table" in content
    assert "Scientific Terminology & Conceptual Corrections" in content
    assert "numerical proximity" in content.lower()


def test_baseline_checkpoint_and_db_unmodified():
    """Bitwise verification that historical baseline checkpoint and app.db records are preserved."""
    assert BASELINE_CHECKPOINT_PATH.exists(), "Baseline checkpoint missing!"

    hasher = hashlib.sha256()
    with open(BASELINE_CHECKPOINT_PATH, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    actual_sha = hasher.hexdigest().lower()
    assert actual_sha == BASELINE_CHECKPOINT_SHA256, f"Checkpoint SHA mismatch: {actual_sha}"

    db_path = WORKSPACE_ROOT / "app.db"
    assert db_path.exists(), "app.db missing!"

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM prediction_records")
    count = cur.fetchone()[0]
    conn.close()
    assert count == 4, f"app.db record count altered! Expected 4, got {count}"

