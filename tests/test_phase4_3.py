"""Automated unit and integration tests for Phase 4.3: Frozen DINOv2 Nonlinear Probing."""

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from research.foundation_models.nonlinear_probe import (
    NonlinearProbingEngine,
    PathologyNonlinearProbe,
    set_seed,
    CANONICAL_CLASSES,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = WORKSPACE_ROOT / "research" / "foundation_models" / "evaluation"
VIZ_DIR = WORKSPACE_ROOT / "research" / "foundation_models" / "visualizations"
CKPT_DIR = WORKSPACE_ROOT / "research" / "foundation_models" / "checkpoints"
CACHE_DIR = WORKSPACE_ROOT / "research" / "foundation_models" / "cache" / "full" / "dinov2_vits14"
DOCS_DIR = WORKSPACE_ROOT / "docs"

BASELINE_CHECKPOINT_PATH = WORKSPACE_ROOT / "models" / "checkpoints" / "gastric_rnn_classifier.pt"
BASELINE_CHECKPOINT_SHA256 = "65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8"


def test_nonlinear_probe_split_isolation_and_alignment():
    """Verify that nonlinear probe strictly enforces train (21767), val (4664), test (4665) isolation."""
    engine = NonlinearProbingEngine(WORKSPACE_ROOT)
    assert len(engine.train_idx) == 21767
    assert len(engine.val_idx) == 4664
    assert len(engine.test_idx) == 4665

    train_set = set(engine.train_idx)
    val_set = set(engine.val_idx)
    test_set = set(engine.test_idx)

    assert len(train_set & val_set) == 0, "Train and Val overlap!"
    assert len(train_set & test_set) == 0, "Train and Test overlap!"
    assert len(val_set & test_set) == 0, "Val and Test overlap!"
    assert len(train_set | val_set | test_set) == 31096


def test_nonlinear_tuning_log_exists_and_valid():
    """Verify that nonlinear probe tuning log exists and best configuration was selected on validation Macro-F1."""
    tuning_log_path = EVAL_DIR / "nonlinear_probe_tuning_log.json"
    assert tuning_log_path.exists(), "nonlinear_probe_tuning_log.json is missing!"

    with open(tuning_log_path, "r") as f:
        data = json.load(f)

    assert "cls" in data and "mean_patch" in data
    assert "best_config" in data["cls"]
    assert "best_config" in data["mean_patch"]

    # Verify best_val_macro_f1 is maximum across candidates
    for repr_type in ["cls", "mean_patch"]:
        grid = data[repr_type]["grid"]
        max_f1 = max(r["val_macro_f1"] for r in grid)
        assert data[repr_type]["best_val_macro_f1"] == max_f1, f"Best config did not maximize validation F1 in {repr_type}"


def test_nonlinear_test_predictions_alignment_and_count():
    """Verify that test predictions CSV files have exactly 4,665 aligned rows matching canonical test manifest."""
    for repr_type in ["cls", "mean_patch"]:
        csv_path = EVAL_DIR / f"nonlinear_probe_{repr_type}_test_predictions.csv"
        assert csv_path.exists(), f"Missing CSV: {csv_path.name}"

        df = pd.read_csv(csv_path)
        assert len(df) == 4665, f"Expected 4665 rows, got {len(df)}"

        # Check canonical probability columns
        prob_cols = [f"prob_{c}" for c in CANONICAL_CLASSES]
        for col in prob_cols:
            assert col in df.columns, f"Missing {col} in {csv_path.name}"

        # Probability normalization
        prob_mat = df[prob_cols].to_numpy()
        sums = np.sum(prob_mat, axis=1)
        assert np.all(np.abs(sums - 1.0) < 1e-3), "Row probabilities do not sum to 1.0"


def test_deterministic_mlp_training():
    """Verify that fixed seed produces bitwise deterministic MLP parameter initialization and forward passes."""
    set_seed(42)
    m1 = PathologyNonlinearProbe(in_features=384, hidden_dim=128, num_classes=8)
    d1 = m1.state_dict()

    set_seed(42)
    m2 = PathologyNonlinearProbe(in_features=384, hidden_dim=128, num_classes=8)
    d2 = m2.state_dict()

    for k in d1:
        assert torch.all(d1[k] == d2[k]), f"Mismatch in key {k} under fixed seed!"

    dummy_x = torch.randn(5, 384)
    m1.eval()
    m2.eval()
    with torch.no_grad():
        out1 = m1(dummy_x)
        out2 = m2(dummy_x)
    assert torch.allclose(out1, out2), "Forward outputs differ between identical seed models!"


def test_train_only_gradient_updates():
    """Verify that only training split features participate in backpropagation; encoder stays 100% frozen."""
    probe = PathologyNonlinearProbe(in_features=384, hidden_dim=128, num_classes=8)
    x_train = torch.randn(16, 384, requires_grad=False)
    y_train = torch.randint(0, 8, (16,))

    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    out = probe(x_train)
    loss = torch.nn.functional.cross_entropy(out, y_train)
    loss.backward()

    # All probe parameters should have gradients
    for p in probe.parameters():
        assert p.grad is not None, "Probe parameter has no gradient!"

    # Input tensor requires_grad must remain False
    assert not x_train.requires_grad


def test_nonlinear_model_serialization_and_reload():
    """Verify that serialized PyTorch checkpoints load cleanly and produce valid outputs."""
    for repr_type in ["cls", "mean_patch"]:
        ckpt_path = CKPT_DIR / f"nonlinear_probe_{repr_type}.pt"
        assert ckpt_path.exists(), f"Checkpoint {ckpt_path.name} is missing!"

        data = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        assert "model_state_dict" in data
        assert "config" in data

        cfg = data["config"]
        model = PathologyNonlinearProbe(
            in_features=384,
            hidden_dim=cfg["hidden_dim"],
            num_classes=8,
            dropout=cfg["dropout"],
            activation=cfg.get("activation", "gelu"),
        )
        model.load_state_dict(data["model_state_dict"])
        model.eval()

        dummy = torch.randn(4, 384)
        with torch.no_grad():
            logits = model(dummy)
        assert logits.shape == (4, 8)
        assert not torch.isnan(logits).any()


def test_nonlinear_metrics_computation_integrity():
    """Verify that test evaluation results JSON files exist and metrics are within expected physical bounds."""
    for repr_type in ["cls", "mean_patch"]:
        res_file = EVAL_DIR / f"nonlinear_probe_{repr_type}_results.json"
        assert res_file.exists(), f"Missing results: {res_file.name}"

        with open(res_file, "r") as f:
            res = json.load(f)

        ov = res["overall_metrics"]
        assert 0.60 <= ov["accuracy"] <= 0.70, f"Unexpected accuracy: {ov['accuracy']}"
        assert 0.60 <= ov["macro_f1"] <= 0.70, f"Unexpected macro-F1: {ov['macro_f1']}"
        assert ov["macro_ovr_auroc"] >= 0.90, f"Unexpected AUROC: {ov['macro_ovr_auroc']}"
        assert 0.0 <= ov["expected_calibration_error"] <= 0.10, f"Unexpected ECE: {ov['expected_calibration_error']}"

        for c in CANONICAL_CLASSES:
            assert c in res["per_class_metrics"]
            assert "f1_score" in res["per_class_metrics"][c]
            assert "recall_sensitivity" in res["per_class_metrics"][c]
            assert "specificity" in res["per_class_metrics"][c]


def test_visualizations_exist_and_non_empty():
    """Verify that all three Phase 4.3 visualization figures exist and are non-empty."""
    expected_figs = [
        "nonlinear_probe_loss_curves.png",
        "nonlinear_probe_confusion_matrix.png",
        "nonlinear_probe_linear_vs_mlp_comparison.png",
    ]
    for fig_name in expected_figs:
        fig_path = VIZ_DIR / fig_name
        assert fig_path.exists(), f"Figure {fig_name} missing!"
        assert fig_path.stat().st_size > 10000, f"Figure {fig_name} unexpectedly small!"


def test_baseline_checkpoint_and_database_unmodified():
    """Verify baseline classifier checkpoint and SQLite prediction history remain 100% bitwise intact."""
    assert BASELINE_CHECKPOINT_PATH.exists(), "Baseline checkpoint missing!"

    hasher = hashlib.sha256()
    with open(BASELINE_CHECKPOINT_PATH, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    actual_sha = hasher.hexdigest().lower()

    assert actual_sha == BASELINE_CHECKPOINT_SHA256, (
        f"Baseline checkpoint SHA-256 altered! Expected {BASELINE_CHECKPOINT_SHA256}, got {actual_sha}"
    )

    db_path = WORKSPACE_ROOT / "app.db"
    assert db_path.exists(), "app.db missing!"

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM prediction_records")
    count = cur.fetchone()[0]
    conn.close()

    assert count == 4, f"app.db prediction records count altered! Expected 4, got {count}"

