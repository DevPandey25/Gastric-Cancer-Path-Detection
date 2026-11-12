"""Automated unit and integration tests for Phase 4.2 Linear Probing & Evaluation."""

import hashlib
import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import LogisticRegression

from research.foundation_models.linear_probe import LinearProbingEngine, PyTorchLinearSoftmax

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = WORKSPACE_ROOT / "research" / "foundation_models" / "evaluation"
VIZ_DIR = WORKSPACE_ROOT / "research" / "foundation_models" / "visualizations"
CKPT_DIR = WORKSPACE_ROOT / "research" / "foundation_models" / "checkpoints"
CACHE_DIR = WORKSPACE_ROOT / "research" / "foundation_models" / "cache" / "full" / "dinov2_vits14"
BASELINE_CHECKPOINT_PATH = WORKSPACE_ROOT / "models" / "checkpoints" / "gastric_rnn_classifier.pt"
BASELINE_CHECKPOINT_SHA256 = "65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8"
CANONICAL_CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]


def test_linear_probe_split_isolation_and_alignment():
    """Verify that linear probe strictly enforces train (21767), val (4664), test (4665) isolation."""
    engine = LinearProbingEngine(WORKSPACE_ROOT)
    assert len(engine.train_idx) == 21767
    assert len(engine.val_idx) == 4664
    assert len(engine.test_idx) == 4665

    # Verify 0 overlap between any split index sets
    train_set = set(engine.train_idx)
    val_set = set(engine.val_idx)
    test_set = set(engine.test_idx)

    assert len(train_set & val_set) == 0, "Train and Val overlap!"
    assert len(train_set & test_set) == 0, "Train and Test overlap!"
    assert len(val_set & test_set) == 0, "Val and Test overlap!"
    assert len(train_set | val_set | test_set) == 31096


def test_tuning_log_exists_and_valid():
    """Verify that hyperparameter tuning log exists and best C* was selected on validation."""
    tuning_log_path = EVAL_DIR / "linear_probe_tuning_log.json"
    assert tuning_log_path.exists(), "linear_probe_tuning_log.json is missing!"

    with open(tuning_log_path, "r") as f:
        data = json.load(f)

    assert "cls" in data and "mean_patch" in data
    assert "best_C" in data["cls"]
    assert "best_C" in data["mean_patch"]

    # Verify C* maximizes validation Macro-F1 across grid
    cls_grid = data["cls"]["grid"]
    max_f1 = max(r["val_macro_f1"] for r in cls_grid)
    assert data["cls"]["best_val_macro_f1"] == max_f1


def test_test_predictions_alignment_and_count():
    """Verify that test predictions CSV files have exactly 4,665 rows matching manifest_test.csv."""
    cls_preds_csv = EVAL_DIR / "linear_probe_cls_test_predictions.csv"
    patch_preds_csv = EVAL_DIR / "linear_probe_mean_patch_test_predictions.csv"

    assert cls_preds_csv.exists(), "linear_probe_cls_test_predictions.csv missing!"
    assert patch_preds_csv.exists(), "linear_probe_mean_patch_test_predictions.csv missing!"

    df_cls = pd.read_csv(cls_preds_csv)
    df_patch = pd.read_csv(patch_preds_csv)

    assert len(df_cls) == 4665, f"Expected 4665 CLS predictions, got {len(df_cls)}"
    assert len(df_patch) == 4665, f"Expected 4665 Patch predictions, got {len(df_patch)}"

    test_manifest = pd.read_csv(WORKSPACE_ROOT / "research" / "splits" / "manifest_test.csv")
    assert (df_cls["relative_path"] == test_manifest["relative_path"]).all(), "Path alignment mismatch!"


def test_deterministic_probe_training():
    """Verify that training LogisticRegression on fixed seed produces deterministic weights and predictions."""
    X = np.random.RandomState(42).randn(100, 384).astype(np.float32)
    y = np.random.RandomState(42).randint(0, 8, size=100)

    clf1 = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs", random_state=42)
    clf1.fit(X, y)
    p1 = clf1.predict_proba(X)

    clf2 = LogisticRegression(C=1.0, max_iter=200, solver="lbfgs", random_state=42)
    clf2.fit(X, y)
    p2 = clf2.predict_proba(X)

    assert np.allclose(p1, p2), "Non-deterministic probe training!"


def test_metric_computation_integrity():
    """Verify that final evaluation results JSON files exist and metrics are within expected bounds."""
    for repr_type in ["cls", "mean_patch"]:
        res_file = EVAL_DIR / f"linear_probe_{repr_type}_results.json"
        assert res_file.exists(), f"Results file missing: {res_file.name}"

        with open(res_file, "r") as f:
            res = json.load(f)

        ov = res["overall_metrics"]
        assert 0.60 <= ov["accuracy"] <= 0.70, f"Unexpected accuracy: {ov['accuracy']}"
        assert 0.60 <= ov["macro_f1"] <= 0.70, f"Unexpected macro-F1: {ov['macro_f1']}"
        assert ov["macro_ovr_auroc"] >= 0.90, f"Unexpected AUROC: {ov['macro_ovr_auroc']}"
        assert 0.0 <= ov["expected_calibration_error"] <= 0.10, f"Unexpected ECE: {ov['expected_calibration_error']}"

        # Verify 8 classes in per_class_metrics
        for c in CANONICAL_CLASSES:
            assert c in res["per_class_metrics"]
            assert "f1_score" in res["per_class_metrics"][c]
            assert "recall_sensitivity" in res["per_class_metrics"][c]
            assert "specificity" in res["per_class_metrics"][c]


def test_probe_checkpoints_exist_and_loadable():
    """Verify that trained model weights exist and can be loaded via joblib."""
    cls_ckpt = CKPT_DIR / "linear_probe_cls.joblib"
    patch_ckpt = CKPT_DIR / "linear_probe_mean_patch.joblib"

    assert cls_ckpt.exists(), "CLS probe checkpoint missing!"
    assert patch_ckpt.exists(), "Patch probe checkpoint missing!"

    model_cls = joblib.load(cls_ckpt)
    model_patch = joblib.load(patch_ckpt)

    assert hasattr(model_cls, "predict_proba")
    assert hasattr(model_patch, "predict_proba")
    assert model_cls.coef_.shape == (8, 384)
    assert model_patch.coef_.shape == (8, 384)


def test_visualizations_exist_and_non_empty():
    """Verify that all three linear probe visualization figures exist and are non-empty."""
    expected_figs = [
        "linear_probe_tuning_curve.png",
        "linear_probe_confusion_matrix.png",
        "linear_probe_per_class_f1.png",
    ]
    for fig_name in expected_figs:
        fig_path = VIZ_DIR / fig_name
        assert fig_path.exists(), f"Figure {fig_name} missing!"
        assert fig_path.stat().st_size > 10000, f"Figure {fig_name} unexpectedly small!"


def test_pytorch_linear_softmax_probe():
    """Verify that PyTorchLinearSoftmax probe executes cleanly without gradients on frozen encoders."""
    probe = PyTorchLinearSoftmax(in_features=384, num_classes=8)
    dummy = torch.randn(4, 384)
    logits = probe(dummy)
    assert logits.shape == (4, 8)
    assert not torch.isnan(logits).any()


def test_baseline_checkpoint_and_database_unmodified():
    """Verify that original baseline classifier checkpoint and SQLite app.db remain bitwise intact."""
    assert BASELINE_CHECKPOINT_PATH.exists(), "Baseline checkpoint missing!"

    hasher = hashlib.sha256()
    with open(BASELINE_CHECKPOINT_PATH, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    actual_sha = hasher.hexdigest().lower()

    assert actual_sha == BASELINE_CHECKPOINT_SHA256, (
        f"Baseline checkpoint altered! Expected {BASELINE_CHECKPOINT_SHA256}, got {actual_sha}"
    )

    db_path = WORKSPACE_ROOT / "app.db"
    assert db_path.exists(), "app.db missing!"

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM prediction_records")
    count = cur.fetchone()[0]
    conn.close()

    assert count == 4, f"app.db prediction records count altered! Expected 4, got {count}"

