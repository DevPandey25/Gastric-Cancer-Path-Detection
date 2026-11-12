"""Unit and Regression Tests for Phase 5 (Explainable AI Implementation).

Verifies:
1. Historical baseline checkpoint and DINOv2 full embedding cache invariants.
2. Phase 4.4 subset artifacts preservation.
3. 32-sample explanation manifest stratification (4 per class, split=test).
4. 8-sample faithfulness subset stratification (1 per class).
5. Attribution grid shape (32, 14, 14), float type, and finiteness.
6. Reconstructed Phikon Mean Patch linear probe weights (8 classes, d=768).
7. Faithfulness deletion experiment validity (MIF score drop > Random drop > LIF drop at 25% and 50%).
8. Visualization artifacts (contact sheet, curves plot, 32 overlay images).
"""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
BASELINE_CKPT = WORKSPACE_ROOT / "models" / "checkpoints" / "gastric_rnn_classifier.pt"
BASELINE_SHA256 = "65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8"

DINOV2_FULL_DIR = WORKSPACE_ROOT / "research" / "foundation_models" / "cache" / "full" / "dinov2_vits14"
SUBSET_DIR = WORKSPACE_ROOT / "research" / "foundation_models" / "cache" / "subset10pct"
EXP_DIR = WORKSPACE_ROOT / "research" / "explainability"

CANONICAL_CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]


def test_baseline_checkpoint_sha256_invariant():
    """Historical baseline CNN-RNN classifier checkpoint must remain bitwise identical."""
    assert BASELINE_CKPT.exists(), "Baseline checkpoint missing!"
    h = hashlib.sha256(BASELINE_CKPT.read_bytes()).hexdigest()
    assert h == BASELINE_SHA256, "Baseline checkpoint SHA-256 altered!"


def test_dinov2_full_cache_intact():
    """Full DINOv2 cache files must exist with exact byte sizes."""
    cls_file = DINOV2_FULL_DIR / "cls_embeddings.npy"
    patch_file = DINOV2_FULL_DIR / "mean_patch_embeddings.npy"
    assert cls_file.exists() and patch_file.exists()
    assert cls_file.stat().st_size == 31096 * 384 * 4
    assert patch_file.stat().st_size == 31096 * 384 * 4


def test_phase4_4_subset_artifacts_intact():
    """Phase 4.4 10% subset caches and evaluation summary must remain intact."""
    assert (SUBSET_DIR / "manifest_subset.csv").exists()
    assert (SUBSET_DIR / "phikon" / "cls_embeddings.npy").stat().st_size == 3110 * 768 * 4
    assert (SUBSET_DIR / "phikon" / "mean_patch_embeddings.npy").stat().st_size == 3110 * 768 * 4
    assert (SUBSET_DIR / "evaluation" / "evaluation_summary.json").exists()
    assert (SUBSET_DIR / "evaluation" / "phikon_mean_patch_test_predictions.csv").exists()


def test_explainability_sample_manifest_integrity():
    """Manifest must contain exactly 32 samples from the test split, 4 per canonical class."""
    manifest_p = EXP_DIR / "manifest_explainability_32.csv"
    assert manifest_p.exists(), "32-sample explanation manifest missing!"
    df = pd.read_csv(manifest_p)
    assert len(df) == 32

    # Verify split == test
    assert (df["split"] == "test").all()

    # Verify 4 per class
    counts = df["class_name"].value_counts().to_dict()
    for c in CANONICAL_CLASSES:
        assert counts.get(c, 0) == 4, f"Class {c} has {counts.get(c, 0)} != 4 samples"

    # Verify exactly 8 faithfulness samples (1 per class)
    faith_df = df[df["is_faithfulness_subset"]]
    assert len(faith_df) == 8
    assert set(faith_df["class_name"]) == set(CANONICAL_CLASSES)


def test_attribution_grid_shape_and_finiteness():
    """Attribution grid array must have shape (32, 14, 14) and finite float values."""
    grid_p = EXP_DIR / "attributions_grid_32x14x14.npy"
    assert grid_p.exists(), "Attribution grid array missing!"
    grid = np.load(grid_p)
    assert grid.shape == (32, 14, 14)
    assert np.all(np.isfinite(grid))


def test_fitted_phikon_linear_probe_weights():
    """Fitted joblib model must exist, have 8 classes, and coefficients shape (8, 768)."""
    probe_p = EXP_DIR / "fitted_phikon_linear_probe.joblib"
    assert probe_p.exists(), "Fitted linear probe missing!"
    clf = joblib.load(probe_p)
    assert hasattr(clf, "coef_")
    assert clf.coef_.shape == (8, 768)
    assert len(clf.classes_) == 8


def test_faithfulness_deletion_curves_validity():
    """Verifies MIF score drop exceeds Random and LIF score drops at 25% and 50% masking."""
    res_p = EXP_DIR / "deletion_faithfulness_results.json"
    assert res_p.exists(), "Deletion results JSON missing!"
    with open(res_p, "r") as f:
        data = json.load(f)

    curves = data["aggregate_curves"]
    assert curves["levels_pct"] == [0, 25, 50, 75]

    # At 25% masking: MIF drop > Random drop > LIF drop
    mif_drop_25 = curves["mif_mean_drops"][1]
    rnd_drop_25 = curves["rnd_mean_drops"][1]
    lif_drop_25 = curves["lif_mean_drops"][1]
    assert mif_drop_25 > rnd_drop_25, f"MIF drop ({mif_drop_25}) <= Random drop ({rnd_drop_25}) at 25%"
    assert rnd_drop_25 > lif_drop_25, f"Random drop ({rnd_drop_25}) <= LIF drop ({lif_drop_25}) at 25%"

    # At 50% masking: MIF drop > Random drop > LIF drop
    mif_drop_50 = curves["mif_mean_drops"][2]
    rnd_drop_50 = curves["rnd_mean_drops"][2]
    lif_drop_50 = curves["lif_mean_drops"][2]
    assert mif_drop_50 > rnd_drop_50, f"MIF drop ({mif_drop_50}) <= Random drop ({rnd_drop_50}) at 50%"
    assert rnd_drop_50 > lif_drop_50, f"Random drop ({rnd_drop_50}) <= LIF drop ({lif_drop_50}) at 50%"


def test_explanations_visualizations_exist():
    """Contact sheet, faithfulness plot, and individual attribution overlays must exist."""
    contact_sheet = EXP_DIR / "explanation_contact_sheet.png"
    curves_plot = EXP_DIR / "faithfulness_deletion_curves.png"
    maps_dir = EXP_DIR / "attribution_maps"

    assert contact_sheet.exists() and contact_sheet.stat().st_size > 100_000
    assert curves_plot.exists() and curves_plot.stat().st_size > 50_000
    assert len(list(maps_dir.glob("*.png"))) == 32

