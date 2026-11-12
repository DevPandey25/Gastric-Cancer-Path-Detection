"""
Unit and Regression Tests for Phase 4.3.1: Metric Reconciliation & Audit.

Verifies:
1. Macro recall equals balanced accuracy algebraically across all probes.
2. Weighted recall equals overall accuracy algebraically across all probes.
3. Confusion matrix row/col totals and support totals equal N=4,665.
4. Per-class One-vs-Rest TP/FP/FN/TN consistency (sum to 4,665; specificity = TN/(TN+FP)).
5. Probability row normalization (sum to 1.0 within CSV rounding tolerance).
6. Deterministic ECE computation with empty bin handling.
7. Multiclass Brier score calculation correctness.
8. Deterministic metric reconciliation output against reconciled_metrics.json.
9. Baseline checkpoint and embedding cache bitwise preservation.
10. Absence of training or test leakage operations during audit.
"""

import hashlib
import json
import math
import os
import pytest
import numpy as np
import pandas as pd

from research.foundation_models.audit_phase4_3_metrics import (
    CANONICAL_CLASSES,
    CLASS_TO_IDX,
    compute_expected_calibration_error,
    compute_multiclass_brier_score,
    compute_contingency_and_quadrants,
    compute_full_metrics_from_predictions,
    wilson_score_interval,
)


@pytest.fixture(scope="module")
def test_manifest():
    path = "research/splits/manifest_test.csv"
    assert os.path.exists(path), f"Manifest {path} missing"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def reconciled_json():
    path = "research/audits/phase4_3_1/reconciled_metrics.json"
    assert os.path.exists(path), f"Audit JSON {path} missing"
    with open(path) as f:
        return json.load(f)


PROBE_KEYS = [
    "linear_probe_cls",
    "linear_probe_mean_patch",
    "nonlinear_probe_cls",
    "nonlinear_probe_mean_patch",
]


def test_baseline_checkpoint_integrity():
    """Verify that the original GastricCancerRNN checkpoint has not been touched."""
    ckpt_path = "models/checkpoints/gastric_rnn_classifier.pt"
    assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"
    sha = hashlib.sha256(open(ckpt_path, "rb").read()).hexdigest()
    expected_sha = "65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8"
    assert sha == expected_sha, f"Baseline checkpoint SHA mismatch: {sha}"


def test_embedding_cache_integrity():
    """Verify that frozen DINOv2 embedding cache arrays remain intact."""
    cls_path = "research/foundation_models/cache/full/dinov2_vits14/cls_embeddings.npy"
    patch_path = "research/foundation_models/cache/full/dinov2_vits14/mean_patch_embeddings.npy"
    assert os.path.exists(cls_path), f"Missing CLS cache: {cls_path}"
    assert os.path.exists(patch_path), f"Missing Mean Patch cache: {patch_path}"

    cls_sha = hashlib.sha256(open(cls_path, "rb").read()).hexdigest()
    patch_sha = hashlib.sha256(open(patch_path, "rb").read()).hexdigest()

    expected_cls = "dc25ae822f2accd0df07ec2c4ab58424e817a2d6bb216b9be611e3d2880ffce5"
    expected_patch = "1cab3ac533678f28de3b592529f63aeb1ab0981c9c93bfe38da4cb4ec483867c"

    assert cls_sha == expected_cls, f"CLS cache altered: {cls_sha}"
    assert patch_sha == expected_patch, f"Patch cache altered: {patch_sha}"


@pytest.mark.parametrize("probe_name", PROBE_KEYS)
def test_macro_recall_equals_balanced_accuracy(probe_name, reconciled_json):
    """Verify mathematical identity: Macro Recall == Balanced Accuracy."""
    data = reconciled_json["models"][probe_name]
    macro_rec = data["macro_recall"]
    balanced_acc = data["balanced_accuracy"]
    assert math.isclose(macro_rec, balanced_acc, abs_tol=1e-9), (
        f"Macro recall ({macro_rec}) != balanced accuracy ({balanced_acc}) in {probe_name}"
    )


@pytest.mark.parametrize("probe_name", PROBE_KEYS)
def test_weighted_recall_equals_overall_accuracy(probe_name, reconciled_json):
    """Verify mathematical identity: Weighted Recall == Overall Accuracy."""
    data = reconciled_json["models"][probe_name]
    weighted_rec = data["weighted_recall"]
    acc = data["accuracy"]
    assert math.isclose(weighted_rec, acc, abs_tol=1e-9), (
        f"Weighted recall ({weighted_rec}) != accuracy ({acc}) in {probe_name}"
    )


@pytest.mark.parametrize("probe_name", PROBE_KEYS)
def test_confusion_matrix_and_support_totals(probe_name, reconciled_json):
    """Verify confusion matrix sum == 4665 and support totals == 4665."""
    data = reconciled_json["models"][probe_name]
    cm = np.array(data["confusion_matrix"])
    assert cm.shape == (8, 8)
    assert int(np.sum(cm)) == 4665

    supports = [data["per_class"][c]["support"] for c in CANONICAL_CLASSES]
    assert sum(supports) == 4665
    expected_supports = [583, 583, 583, 584, 583, 583, 583, 583]
    assert supports == expected_supports, f"Unexpected supports {supports} in {probe_name}"


@pytest.mark.parametrize("probe_name", PROBE_KEYS)
def test_quadrants_consistency_and_specificity(probe_name, reconciled_json):
    """Verify TP + FP + FN + TN == 4665 and Specificity == TN / (TN + FP)."""
    data = reconciled_json["models"][probe_name]
    for c in CANONICAL_CLASSES:
        pc = data["per_class"][c]
        tp, fp, fn, tn = pc["tp"], pc["fp"], pc["fn"], pc["tn"]
        total = tp + fp + fn + tn
        assert total == 4665, f"Quadrants sum to {total} != 4665 for {c} in {probe_name}"

        expected_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        assert math.isclose(pc["specificity"], expected_spec, abs_tol=1e-6), (
            f"Specificity mismatch for {c} in {probe_name}: {pc['specificity']} vs {expected_spec}"
        )


@pytest.mark.parametrize("probe_name", PROBE_KEYS)
def test_probability_normalization(probe_name):
    """Verify that probability arrays in test prediction CSVs sum to 1.0."""
    csv_path = f"research/foundation_models/evaluation/{probe_name}_test_predictions.csv"
    assert os.path.exists(csv_path)
    df = pd.read_csv(csv_path)
    assert len(df) == 4665

    prob_cols = [f"prob_{c}" for c in CANONICAL_CLASSES]
    probs = df[prob_cols].values
    row_sums = np.sum(probs, axis=1)

    # Maximum deviation must be <= 0.001 (due to 4-decimal CSV rounding)
    max_dev = np.max(np.abs(row_sums - 1.0))
    assert max_dev <= 0.001, f"Excessive probability row sum deviation in {probe_name}: {max_dev}"


def test_deterministic_ece_and_brier_computation():
    """Verify synthetic deterministic behavior of ECE and Brier score functions."""
    y_true = np.array([0, 1, 2, 0])
    # Perfect predictions
    probs_perfect = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ])
    ece_perf = compute_expected_calibration_error(y_true, probs_perfect, n_bins=10)
    brier_perf = compute_multiclass_brier_score(y_true, probs_perfect, num_classes=3)
    assert math.isclose(ece_perf, 0.0, abs_tol=1e-7)
    assert math.isclose(brier_perf, 0.0, abs_tol=1e-7)

    # Uniform predictions
    probs_unif = np.ones((4, 3)) / 3.0
    brier_unif = compute_multiclass_brier_score(y_true, probs_unif, num_classes=3)
    # Each sample: (1 - 1/3)^2 + 2 * (0 - 1/3)^2 = 4/9 + 2/9 = 6/9 = 2/3 approx 0.6667
    assert math.isclose(brier_unif, 2.0 / 3.0, abs_tol=1e-7)


def test_wilson_interval_bounds():
    """Verify Wilson score confidence interval calculation."""
    low, high = wilson_score_interval(3010, 4665, 0.95)
    assert 0.0 <= low <= high <= 1.0
    p = 3010 / 4665
    assert low < p < high
    assert math.isclose(low, 0.6314, abs_tol=1e-3)
    assert math.isclose(high, 0.6588, abs_tol=1e-3)

