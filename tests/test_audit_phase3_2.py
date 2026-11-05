"""Tests for Phase 3.2 Checkpoint Lineage & Duplicate Audit Engine."""

import json
from pathlib import Path
import pandas as pd
import pytest

from research.audits.phase3_2_audit import extract_relative_path, run_phase3_2_audit


def test_extract_relative_path():
    assert extract_relative_path("C:/path/to/data/ADI/ADI_100.png") == "ADI/ADI_100.png"
    assert extract_relative_path("TUM\\TUM_500.png") == "TUM/TUM_500.png"


def test_audit_artifacts_generated():
    audit_dir = Path("research/audits/phase3_2")
    assert (audit_dir / "checkpoint_train_test_overlap.csv").exists()
    assert (audit_dir / "test_duplicate_group_summary.csv").exists()
    assert (audit_dir / "deduplicated_predictions.csv").exists()
    assert (audit_dir / "duplicate_aware_metrics.json").exists()
    assert (audit_dir / "cluster_bootstrap_results.json").exists()


def test_lineage_overlap_integrity():
    overlap_df = pd.read_csv("research/audits/phase3_2/checkpoint_train_test_overlap.csv")
    assert len(overlap_df) == 4665
    assert overlap_df["sha256"].nunique() == 4496

    # 100% mapped lineage
    assert (overlap_df["historical_split"] == "unresolved").sum() == 0

    # Overlap breakdown matches
    train_count = overlap_df["in_original_train"].sum()
    val_count = overlap_df["in_original_val"].sum()
    test_count = overlap_df["in_original_test"].sum()

    assert train_count == 3758
    assert val_count == 465
    assert test_count == 442
    assert train_count + val_count + test_count == 4665


def test_deduplicated_predictions_integrity():
    dedup_df = pd.read_csv("research/audits/phase3_2/deduplicated_predictions.csv")
    assert len(dedup_df) == 4496
    assert dedup_df["sha256"].nunique() == 4496
    assert "is_correct" in dedup_df.columns
    assert "confidence" in dedup_df.columns


def test_duplicate_group_summary_consistency():
    summary_df = pd.read_csv("research/audits/phase3_2/test_duplicate_group_summary.csv")
    assert len(summary_df) == 136

    # Zero label conflicts
    assert summary_df["label_consistent"].all()

    # Zero prediction inconsistencies within exact duplicate groups
    assert summary_df["predictions_consistent"].all()


def test_duplicate_aware_metrics_data():
    with open("research/audits/phase3_2/duplicate_aware_metrics.json") as f:
        data = json.load(f)

    meta = data["metadata"]
    assert meta["total_test_files"] == 4665
    assert meta["exact_unique_images"] == 4496
    assert meta["integrity_classification"] == "CONTAMINATED"

    # Compare file-weighted vs exact-unique accuracy
    file_acc = data["file_weighted_metrics"]["accuracy"]
    dedup_acc = data["exact_unique_metrics"]["accuracy"]
    assert file_acc == 0.7104
    assert dedup_acc == 0.7093

    # Check lineage breakdown
    lineage = data["lineage_subset_comparison"]
    assert lineage["historical_train_overlap_subset"]["sample_count"] == 3758
    assert lineage["historical_train_overlap_subset"]["accuracy"] == 0.7299
    assert lineage["historical_unseen_test_subset"]["sample_count"] == 442
    assert lineage["historical_unseen_test_subset"]["accuracy"] == 0.6584

