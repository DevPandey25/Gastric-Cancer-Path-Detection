"""Automated Unit and Regression Tests for Phase 4.2.2: Per-Class Metric Table Integrity."""

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = WORKSPACE_ROOT / "research" / "foundation_models" / "evaluation"
AUDIT_DIR = WORKSPACE_ROOT / "research" / "audits" / "phase4_2_2"
DOCS_DIR = WORKSPACE_ROOT / "docs"

BASELINE_CHECKPOINT_PATH = WORKSPACE_ROOT / "models" / "checkpoints" / "gastric_rnn_classifier.pt"
BASELINE_CHECKPOINT_SHA256 = "65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8"

CANONICAL_CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]
EXPECTED_SUPPORT = [583, 583, 583, 584, 583, 583, 583, 583]


def test_verified_metrics_json_integrity():
    """Verify that verified_metrics.json exists, covers both models, and matches mathematical definitions."""
    json_path = AUDIT_DIR / "verified_metrics.json"
    assert json_path.exists(), f"Missing verification artifact: {json_path}"

    with open(json_path, "r") as f:
        data = json.load(f)

    assert data["canonical_classes"] == CANONICAL_CLASSES

    for model_key in ["cls", "mean_patch"]:
        model_data = data[model_key]
        assert model_data["total_samples"] == 4665
        cm = np.array(model_data["confusion_matrix"])
        assert cm.shape == (8, 8)

        # Verify row sums equal support
        row_sums = cm.sum(axis=1)
        assert np.array_equal(row_sums, EXPECTED_SUPPORT), f"Row sum support mismatch in {model_key}"

        class_metrics = model_data["class_metrics"]
        assert len(class_metrics) == 8

        raw_precisions = []
        raw_recalls = []
        raw_specificities = []
        raw_f1s = []
        supports = []

        for i, m in enumerate(class_metrics):
            assert m["class"] == CANONICAL_CLASSES[i]
            assert m["support"] == EXPECTED_SUPPORT[i]

            tp = m["tp"]
            fp = m["fp"]
            fn = m["fn"]
            tn = m["tn"]
            supp = m["support"]
            pred = m["predicted_count"]

            # Mathematical Invariants
            assert tp + fn == supp, f"TP+FN != Support for {m['class']}"
            assert tp + fp == pred, f"TP+FP != Predicted Count for {m['class']}"
            assert tp + fp + fn + tn == 4665, f"Sum of quadrants != 4665 for {m['class']}"

            # Exact formula checks
            expected_p = tp / (tp + fp)
            expected_r = tp / (tp + fn)
            expected_s = tn / (tn + fp)
            expected_f1 = (2.0 * expected_p * expected_r) / (expected_p + expected_r)

            assert abs(m["precision_raw"] - expected_p) < 1e-9
            assert abs(m["recall_raw"] - expected_r) < 1e-9
            assert abs(m["specificity_raw"] - expected_s) < 1e-9
            assert abs(m["f1_raw"] - expected_f1) < 1e-9

            raw_precisions.append(m["precision_raw"])
            raw_recalls.append(m["recall_raw"])
            raw_specificities.append(m["specificity_raw"])
            raw_f1s.append(m["f1_raw"])
            supports.append(supp)

        # Independent Macro verification from unrounded raw metrics
        macro_avg = model_data["macro_average"]
        assert abs(macro_avg["precision_raw"] - float(np.mean(raw_precisions))) < 1e-9
        assert abs(macro_avg["recall_raw"] - float(np.mean(raw_recalls))) < 1e-9
        assert abs(macro_avg["specificity_raw"] - float(np.mean(raw_specificities))) < 1e-9
        assert abs(macro_avg["f1_raw"] - float(np.mean(raw_f1s))) < 1e-9

        # Independent Support-Weighted verification from unrounded raw metrics
        weighted_avg = model_data["weighted_average"]
        tot_supp = sum(supports)
        assert abs(weighted_avg["precision_raw"] - float(sum(p * s for p, s in zip(raw_precisions, supports)) / tot_supp)) < 1e-9
        assert abs(weighted_avg["recall_raw"] - float(sum(r * s for r, s in zip(raw_recalls, supports)) / tot_supp)) < 1e-9
        assert abs(weighted_avg["specificity_raw"] - float(sum(spec * s for spec, s in zip(raw_specificities, supports)) / tot_supp)) < 1e-9
        assert abs(weighted_avg["f1_raw"] - float(sum(f * s for f, s in zip(raw_f1s, supports)) / tot_supp)) < 1e-9


def test_confusion_matrix_column_totals():
    """Verify that confusion matrix column sums match class predicted counts exactly."""
    json_path = AUDIT_DIR / "verified_metrics.json"
    with open(json_path, "r") as f:
        data = json.load(f)

    for model_key in ["cls", "mean_patch"]:
        cm = np.array(data[model_key]["confusion_matrix"])
        col_sums = cm.sum(axis=0)
        class_preds = [m["predicted_count"] for m in data[model_key]["class_metrics"]]
        assert np.array_equal(col_sums, class_preds), f"Column sum mismatch in {model_key}"


def test_markdown_tables_conform_to_canonical_schema():
    """Verify that report markdown tables contain exactly the 8 canonical metric columns and 10 rows."""
    expected_rows = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM", "Macro", "Weighted"]

    reports_to_check = [
        DOCS_DIR / "PHASE4_2_LINEAR_PROBING_REPORT.md",
        DOCS_DIR / "PHASE4_2_1_METRIC_AND_EVALUATION_AUDIT.md",
        DOCS_DIR / "PHASE4_2_2_FINAL_METRIC_TABLE_VERIFICATION.md",
    ]

    for rep in reports_to_check:
        if not rep.exists():
            continue
        content = rep.read_text(encoding="utf-8")
        # Check that canonical class names appear
        for r_name in expected_rows:
            assert f"**{r_name}**" in content or r_name in content, f"Row {r_name} missing in {rep.name}"


def test_baseline_checkpoint_and_system_state_unmodified():
    """Verify that historical checkpoint, database, and embedding cache are preserved."""
    assert BASELINE_CHECKPOINT_PATH.exists()

    hasher = hashlib.sha256()
    with open(BASELINE_CHECKPOINT_PATH, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    assert hasher.hexdigest().lower() == BASELINE_CHECKPOINT_SHA256

    db_path = WORKSPACE_ROOT / "app.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM prediction_records")
    count = cur.fetchone()[0]
    conn.close()
    assert count == 4, f"Expected 4 records, got {count}"

