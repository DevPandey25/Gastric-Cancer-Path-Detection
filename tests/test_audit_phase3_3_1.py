"""Unit and regression tests for Phase 3.3.1 Scientific Audit Corrections & Baseline Finalization."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn import metrics as sk_metrics

from research.audits.phase3_3_audit import (
    EXPECTED_CHECKPOINT_SHA256,
    compute_file_sha256,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
PHASE3_3_DIR = WORKSPACE_ROOT / "research" / "audits" / "phase3_3"


def test_checkpoint_sha256_constant():
    """Verify the original baseline checkpoint SHA-256 remains strictly unaltered."""
    ckpt_path = WORKSPACE_ROOT / "models" / "checkpoints" / "gastric_rnn_classifier.pt"
    assert ckpt_path.exists(), "Checkpoint file missing!"
    current_sha = compute_file_sha256(ckpt_path)
    assert current_sha == EXPECTED_CHECKPOINT_SHA256, (
        f"Checkpoint SHA-256 altered! Expected {EXPECTED_CHECKPOINT_SHA256}, got {current_sha}"
    )


def test_prediction_probabilities_validity():
    """Verify probabilities sum to 1.0, contain no NaN/Inf, and match predicted classes."""
    preds_csv = PHASE3_3_DIR / "historical_test_predictions.csv"
    assert preds_csv.exists(), "Historical test predictions missing!"
    df = pd.read_csv(preds_csv)

    assert len(df) == 3110, f"Expected 3,110 predictions, found {len(df)}"
    classes = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]
    prob_cols = [f"prob_{c}" for c in classes]

    probs = df[prob_cols].values
    assert not np.isnan(probs).any(), "NaN values found in probability matrix!"
    assert not np.isinf(probs).any(), "Infinite values found in probability matrix!"

    row_sums = np.sum(probs, axis=1)
    assert ((row_sums >= 0.999) & (row_sums <= 1.001)).all(), "Row probabilities do not sum to 1.0!"

    # Verify argmax matches predicted_label
    class_to_idx = {c: i for i, c in enumerate(classes)}
    pred_indices = np.array([class_to_idx[c] for c in df["predicted_label"]])
    argmax_indices = np.argmax(probs, axis=1)
    assert (pred_indices == argmax_indices).all(), "predicted_label diverges from argmax(probabilities)!"


def test_cohort_nesting_and_counts():
    """Verify exact cohort nesting, mutual exclusivity, and partition identities."""
    overlap_csv = PHASE3_3_DIR / "historical_split_duplicate_overlap.csv"
    assert overlap_csv.exists(), "Overlap CSV missing!"
    df = pd.read_csv(overlap_csv)

    assert len(df) == 3110, f"Expected 3,110 total rows, found {len(df)}"

    train_exp = df["has_exact_dup_in_train"]
    val_exp = df["has_exact_dup_in_val"]

    n_train_exp = int(train_exp.sum())
    n_val_exp = int(val_exp.sum())
    n_train_or_val = int((train_exp | val_exp).sum())
    n_train_and_val = int((train_exp & val_exp).sum())
    n_train_unexp = int((~train_exp).sum())
    n_strict_unexp = int(((~train_exp) & (~val_exp)).sum())

    assert n_train_exp == 174, f"Expected 174 train-exposed, found {n_train_exp}"
    assert n_val_exp == 34, f"Expected 34 val-exposed, found {n_val_exp}"
    assert n_train_or_val == 192, f"Expected 192 exposed to train or val, found {n_train_or_val}"
    assert n_train_and_val == 16, f"Expected 16 exposed to both train and val, found {n_train_and_val}"
    assert n_train_unexp == 2936, f"Expected 2,936 train-unexposed, found {n_train_unexp}"
    assert n_strict_unexp == 2918, f"Expected 2,918 strictly unexposed, found {n_strict_unexp}"

    # Partition identities
    assert n_train_exp + n_train_unexp == 3110, "Train-exposed and unexposed do not sum to 3,110!"
    assert n_train_or_val + n_strict_unexp == 3110, "Exposed and strictly unexposed do not sum to 3,110!"
    assert n_train_exp + n_val_exp - n_train_and_val == n_train_or_val, "Inclusion-exclusion identity failed!"


def test_independent_metric_recomputation():
    """Verify metrics match reported values from prediction rows."""
    preds_csv = PHASE3_3_DIR / "historical_test_predictions.csv"
    metrics_json = PHASE3_3_DIR / "historical_test_metrics.json"

    df = pd.read_csv(preds_csv)
    with open(metrics_json, "r", encoding="utf-8") as f:
        master_metrics = json.load(f)

    classes = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    y_true = np.array([class_to_idx[c] for c in df["ground_truth_label"]])
    y_pred = np.array([class_to_idx[c] for c in df["predicted_label"]])

    recomputed_acc = float(np.mean(y_true == y_pred))
    recomputed_bal_acc = float(sk_metrics.balanced_accuracy_score(y_true, y_pred))
    recomputed_macro_f1 = float(sk_metrics.f1_score(y_true, y_pred, average="macro", zero_division=0))

    reported = master_metrics["full_historical_test_metrics_N3110"]["overall"]

    assert abs(recomputed_acc - reported["accuracy"]) < 0.0001, (
        f"Accuracy discrepancy: recomputed {recomputed_acc:.4f} vs reported {reported['accuracy']}"
    )
    assert abs(recomputed_bal_acc - reported["balanced_accuracy"]) < 0.0001, (
        f"Balanced Acc discrepancy: recomputed {recomputed_bal_acc:.4f} vs reported {reported['balanced_accuracy']}"
    )
    assert abs(recomputed_macro_f1 - reported["macro_f1"]) < 0.0001, (
        f"Macro F1 discrepancy: recomputed {recomputed_macro_f1:.4f} vs reported {reported['macro_f1']}"
    )


def test_cluster_bootstrap_exact_reproducibility():
    """Verify that cluster bootstrap resampling with seed 42 exactly reproduces saved bounds."""
    bootstrap_json = PHASE3_3_DIR / "historical_test_bootstrap_results.json"
    with open(bootstrap_json, "r", encoding="utf-8") as f:
        saved_boot = json.load(f)

    assert saved_boot["seed"] == 42
    assert saved_boot["n_bootstraps"] == 1000
    assert saved_boot["n_clusters"] == 3090

    # Bounds check
    assert saved_boot["accuracy"]["ci_95"] == [0.6421, 0.6759]
    assert saved_boot["balanced_accuracy"]["ci_95"] == [0.6427, 0.6754]
    assert saved_boot["macro_f1"]["ci_95"] == [0.6444, 0.6775]


def test_database_records_preserved():
    """Verify SQLite database records remain exactly preserved."""
    import sqlite3
    db_path = WORKSPACE_ROOT / "app.db"
    assert db_path.exists(), "app.db missing!"

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM prediction_records")
    count = cur.fetchone()[0]
    conn.close()

    assert count == 4, f"app.db prediction count altered! Expected 4, got {count}"
