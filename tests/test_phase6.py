"""Unit and Regression Tests for Phase 6 (Controlled Robustness Analysis).

Verifies:
1. Invariant integrity: Baseline checkpoint SHA-256, DINOv2 full cache, Phase 4.4 subset artifacts.
2. Robustness test manifest: Exactly 467 test images, all 8 canonical classes represented (min 58 per class).
3. Perturbation functions: Preserve (224, 224) spatial resolution and RGB mode, non-empty, finite.
4. Evaluation coverage: Exactly 5 conditions per sample (2,335 total rows in CSV and JSON).
5. Prediction validity: Probabilities sum to 1.0, confidences equal max probability, predictions in [0, 7].
6. Mathematical metric integrity:
   - Ground-truth error rate == 1.0 - Accuracy.
   - Prediction consistency + Label-flip rate == 1.0.
   - Ground-truth error rate is mathematically distinct from label-flip rate.
7. Artifact generation: All CSV, JSON, and 3 publication-ready PNG visualization plots exist and are non-empty.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from research.robustness.perturbations import (
    apply_brightness_contrast,
    apply_gaussian_blur,
    apply_hed_stain_variation,
    apply_jpeg_compression,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
BASELINE_CKPT = WORKSPACE_ROOT / "models" / "checkpoints" / "gastric_rnn_classifier.pt"
BASELINE_SHA256 = "65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8"

DINOV2_FULL_DIR = WORKSPACE_ROOT / "research" / "foundation_models" / "cache" / "full" / "dinov2_vits14"
SUBSET_DIR = WORKSPACE_ROOT / "research" / "foundation_models" / "cache" / "subset10pct"
ROBUSTNESS_DIR = WORKSPACE_ROOT / "research" / "robustness"

CANONICAL_CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]
CONDITIONS = [
    "clean",
    "brightness_contrast",
    "gaussian_blur",
    "jpeg_compression",
    "stain_hed_variation",
]


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


def test_robustness_manifest_integrity():
    """Robustness manifest must contain exactly 467 test samples across 8 classes."""
    manifest_p = ROBUSTNESS_DIR / "robustness_subset_manifest.csv"
    assert manifest_p.exists(), "Robustness subset manifest missing!"
    df = pd.read_csv(manifest_p)
    assert len(df) == 467
    assert (df["split"] == "test").all()

    class_counts = df["class_name"].value_counts().to_dict()
    for c in CANONICAL_CLASSES:
        assert class_counts.get(c, 0) >= 58, f"Class {c} has {class_counts.get(c, 0)} < 58 samples"


def test_perturbation_functions_preserve_dimensions_and_mode():
    """All perturbation functions must preserve 224x224 RGB image format and produce finite values."""
    img = Image.new("RGB", (224, 224), (180, 120, 160))
    for fn in [
        apply_brightness_contrast,
        apply_gaussian_blur,
        apply_jpeg_compression,
        apply_hed_stain_variation,
    ]:
        res = fn(img)
        assert isinstance(res, Image.Image)
        assert res.size == (224, 224)
        assert res.mode == "RGB"
        arr = np.asarray(res)
        assert arr.shape == (224, 224, 3)
        assert np.all(np.isfinite(arr))


def test_robustness_predictions_dataset_coverage():
    """Predictions CSV and JSON must contain exactly 2,335 records (467 samples x 5 conditions)."""
    csv_p = ROBUSTNESS_DIR / "predictions_robustness.csv"
    json_p = ROBUSTNESS_DIR / "predictions_robustness.json"
    assert csv_p.exists(), "predictions_robustness.csv missing!"
    assert json_p.exists(), "predictions_robustness.json missing!"

    df = pd.read_csv(csv_p)
    assert len(df) == 467 * 5

    with open(json_p, "r") as f:
        records = json.load(f)
    assert len(records) == 467 * 5

    # Check each condition has exactly 467 samples
    cond_counts = df["condition"].value_counts().to_dict()
    for c in CONDITIONS:
        assert cond_counts.get(c, 0) == 467


def test_predictions_numerical_validity():
    """Predicted probabilities must sum to 1.0, confidences match max prob, and classes are valid."""
    df = pd.read_csv(ROBUSTNESS_DIR / "predictions_robustness.csv")
    prob_cols = [f"prob_{c}" for c in CANONICAL_CLASSES]

    prob_sums = df[prob_cols].sum(axis=1)
    np.testing.assert_allclose(prob_sums, 1.0, atol=1e-3)

    max_probs = df[prob_cols].max(axis=1)
    np.testing.assert_allclose(df["confidence"], max_probs, atol=1e-3)

    assert df["predicted_class"].isin(CANONICAL_CLASSES).all()
    assert df["predicted_index"].between(0, 7).all()


def test_metric_identities_and_error_vs_flip_rate():
    """Ground truth error rate and label flip rate must be mathematically sound and distinct."""
    json_p = ROBUSTNESS_DIR / "metrics_robustness.json"
    assert json_p.exists(), "metrics_robustness.json missing!"
    with open(json_p, "r") as f:
        metrics = json.load(f)["summary_metrics"]

    for cond in CONDITIONS:
        cm = metrics[cond]
        acc = cm["accuracy"]
        err = cm["ground_truth_error_rate"]
        consistency = cm["prediction_consistency"]
        flip = cm["label_flip_rate"]

        # Math identity 1: Accuracy + Error rate == 1.0
        assert abs((acc + err) - 1.0) < 1e-4

        # Math identity 2: Consistency + Label flip rate == 1.0
        assert abs((consistency + flip) - 1.0) < 1e-4

        # Clean condition sanity
        if cond == "clean":
            assert consistency == 1.0
            assert flip == 0.0

    # Ensure error rate is distinct from flip rate across perturbed conditions
    for cond in ["brightness_contrast", "gaussian_blur", "jpeg_compression", "stain_hed_variation"]:
        cm = metrics[cond]
        err = cm["ground_truth_error_rate"]
        flip = cm["label_flip_rate"]
        assert err != flip, f"Condition {cond} has error rate == flip rate ({err}), which conflates them!"


def test_visualization_artifacts_exist_and_non_empty():
    """All 3 required visualization plots must exist and have non-zero file sizes."""
    contact_sheet = ROBUSTNESS_DIR / "contact_sheet_robustness.png"
    overall_plot = ROBUSTNESS_DIR / "overall_robustness_plot.png"
    per_class_plot = ROBUSTNESS_DIR / "per_class_robustness.png"

    for plot_p in [contact_sheet, overall_plot, per_class_plot]:
        assert plot_p.exists(), f"Missing visualization plot: {plot_p.name}"
        assert plot_p.stat().st_size > 5000, f"Plot {plot_p.name} is suspiciously small ({plot_p.stat().st_size} bytes)"

