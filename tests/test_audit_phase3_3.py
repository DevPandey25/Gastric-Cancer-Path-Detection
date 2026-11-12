"""Unit and regression tests for Phase 3.3 Historical Test Evaluation & Final Baseline Verification."""

import json
from pathlib import Path
import pandas as pd
import pytest

from research.audits.phase3_3_audit import (
    EXPECTED_CHECKPOINT_SHA256,
    compute_file_sha256,
    extract_relative_path,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = WORKSPACE_ROOT / "research" / "audits" / "phase3_3"


def test_checkpoint_sha256_unaltered():
    """Verify that the baseline model checkpoint has not been altered or overwritten."""
    ckpt_path = WORKSPACE_ROOT / "models" / "checkpoints" / "gastric_rnn_classifier.pt"
    assert ckpt_path.exists(), "Baseline checkpoint missing!"
    current_sha = compute_file_sha256(ckpt_path)
    assert current_sha == EXPECTED_CHECKPOINT_SHA256, (
        f"Checkpoint SHA-256 altered! Expected {EXPECTED_CHECKPOINT_SHA256}, got {current_sha}"
    )


def test_historical_test_cohort_reconstruction():
    """Verify that the historical test cohort has exactly 3,110 rows and 0 path overlap with train/val."""
    hist_manifest_path = WORKSPACE_ROOT / "model_train" / "runs" / "dataset_manifest.csv"
    assert hist_manifest_path.exists(), "Historical training manifest missing!"

    df_hist = pd.read_csv(hist_manifest_path)
    df_hist["rel_path"] = df_hist["image_path"].apply(extract_relative_path)

    test_paths = set(df_hist[df_hist["split"] == "test"]["rel_path"])
    train_paths = set(df_hist[df_hist["split"] == "train"]["rel_path"])
    val_paths = set(df_hist[df_hist["split"] == "val"]["rel_path"])

    assert len(test_paths) == 3110, f"Expected 3,110 unique test paths, found {len(test_paths)}"
    assert len(train_paths) == 24876, f"Expected 24,876 train paths, found {len(train_paths)}"
    assert len(val_paths) == 3110, f"Expected 3,110 val paths, found {len(val_paths)}"

    # Mutual exclusivity of paths
    assert len(train_paths.intersection(test_paths)) == 0, "Historical Train and Test paths overlap!"
    assert len(val_paths.intersection(test_paths)) == 0, "Historical Val and Test paths overlap!"


def test_phase3_3_artifacts_exist():
    """Verify all 6 required Phase 3.3 audit artifacts exist on disk and are non-empty."""
    required_files = [
        "historical_test_predictions.csv",
        "historical_test_metrics.json",
        "historical_split_duplicate_overlap.csv",
        "historical_test_duplicate_summary.csv",
        "historical_test_unique_predictions.csv",
        "historical_test_bootstrap_results.json",
    ]
    for fname in required_files:
        fpath = AUDIT_DIR / fname
        assert fpath.exists(), f"Required Phase 3.3 artifact missing: {fname}"
        assert fpath.stat().st_size > 0, f"Artifact is empty: {fname}"


def test_prediction_count_and_path_uniqueness():
    """Verify prediction file has exactly 3,110 unique predictions with valid probabilities."""
    preds_csv = AUDIT_DIR / "historical_test_predictions.csv"
    df = pd.read_csv(preds_csv)

    assert len(df) == 3110, f"Expected 3,110 predictions, got {len(df)}"
    assert len(df["relative_path"].unique()) == 3110, "Duplicate relative paths in predictions!"

    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    assert len(prob_cols) == 8, f"Expected 8 probability columns, got {len(prob_cols)}"

    # Verify probability normalization
    sums = df[prob_cols].sum(axis=1)
    assert ((sums >= 0.99) & (sums <= 1.01)).all(), "Probabilities do not sum to 1.0!"


def test_exact_hash_cross_split_leakage():
    """Verify cross-split exact SHA-256 duplicate counts matching audit report."""
    overlap_csv = AUDIT_DIR / "historical_split_duplicate_overlap.csv"
    df_overlap = pd.read_csv(overlap_csv)

    assert len(df_overlap) == 3110, f"Expected 3,110 rows in overlap table, got {len(df_overlap)}"

    # Verify exact matches in train and val
    train_exposed = df_overlap["has_exact_dup_in_train"].sum()
    val_exposed = df_overlap["has_exact_dup_in_val"].sum()

    assert train_exposed == 174, f"Expected 174 test images with exact match in Train, got {train_exposed}"
    assert val_exposed == 34, f"Expected 34 test images with exact match in Val, got {val_exposed}"

    # Verify that memorization delta is positive (model performed better on training duplicates)
    acc_exposed = df_overlap[df_overlap["has_exact_dup_in_train"]]["is_correct"].mean()
    acc_unexposed = df_overlap[~df_overlap["has_exact_dup_in_train"]]["is_correct"].mean()
    assert acc_exposed > acc_unexposed, (
        f"Expected training duplicate accuracy ({acc_exposed:.4f}) > unexposed ({acc_unexposed:.4f})"
    )


def test_dhash_cross_split_leakage():
    """Verify dHash cluster leakage counts across historical splits."""
    overlap_csv = AUDIT_DIR / "historical_split_duplicate_overlap.csv"
    df_overlap = pd.read_csv(overlap_csv)

    # At Hamming threshold <= 2, dHash clusters match exact duplicates
    dhash_train_exposed = df_overlap["has_dhash_dup_in_train"].sum()
    assert dhash_train_exposed == 174, f"Expected 174 test images in train dHash clusters, got {dhash_train_exposed}"


def test_unique_predictions_integrity():
    """Verify deduplicated predictions contain exactly 3,090 unique images."""
    unique_csv = AUDIT_DIR / "historical_test_unique_predictions.csv"
    df_unique = pd.read_csv(unique_csv)

    assert len(df_unique) == 3090, f"Expected 3,090 unique images, got {len(df_unique)}"
    assert len(df_unique["relative_path"].unique()) == 3090, "Duplicate paths in unique predictions!"


def test_metrics_json_integrity():
    """Verify historical_test_metrics.json has correct structure and values."""
    metrics_json = AUDIT_DIR / "historical_test_metrics.json"
    with open(metrics_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "full_historical_test_metrics_N3110" in data
    assert "deduplicated_unique_metrics_N3090" in data
    assert "train_val_excluded_metrics_N2918" in data
    assert "cluster_bootstrap_uncertainty" in data

    acc = data["full_historical_test_metrics_N3110"]["overall"]["accuracy"]
    assert 0.65 <= acc <= 0.67, f"Historical test accuracy out of expected range: {acc}"

    macro_f1 = data["full_historical_test_metrics_N3110"]["overall"]["macro_f1"]
    assert 0.65 <= macro_f1 <= 0.68, f"Historical test macro-F1 out of expected range: {macro_f1}"

