"""Tests for Comprehensive Multiclass Benchmark Metrics."""

import numpy as np

from research.benchmark.metrics import (
    compute_comprehensive_metrics,
    compute_expected_calibration_error,
    compute_multiclass_brier_score,
)


def test_perfect_prediction_metrics():
    # 4 classes, 4 samples, 100% correct
    y_true = [0, 1, 2, 3]
    y_prob = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    class_names = ["C0", "C1", "C2", "C3"]

    results = compute_comprehensive_metrics(y_true, y_prob, class_names=class_names)
    overall = results["overall"]

    assert overall["accuracy"] == 1.0
    assert overall["macro_f1"] == 1.0
    assert overall["balanced_accuracy"] == 1.0
    assert overall["cohen_kappa"] == 1.0
    assert overall["brier_score"] == 0.0
    assert overall["expected_calibration_error"] == 0.0

    # Confusion matrix
    cm = np.array(results["confusion_matrix"])
    assert np.all(cm == np.eye(4))

    # Per-class metrics
    for c in class_names:
        assert results["per_class"][c]["precision"] == 1.0
        assert results["per_class"][c]["recall_sensitivity"] == 1.0
        assert results["per_class"][c]["specificity"] == 1.0


def test_calibration_and_brier_score():
    y_true = np.array([0, 1])
    # Overconfident wrong prediction
    y_prob = np.array([
        [0.2, 0.8],  # True is 0, predicted 1 with 0.8 confidence (wrong!)
        [0.1, 0.9],  # True is 1, predicted 1 with 0.9 confidence (correct)
    ])
    ece, bins = compute_expected_calibration_error(y_true, y_prob, n_bins=5)
    assert ece > 0.0
    assert len(bins) > 0

    brier = compute_multiclass_brier_score(y_true, y_prob, num_classes=2)
    assert brier > 0.0


def test_metrics_handles_unseen_classes():
    # Only class 0 and 1 present in ground truth, but 4 classes defined
    y_true = [0, 0, 1, 1]
    y_prob = [
        [0.9, 0.1, 0.0, 0.0],
        [0.8, 0.2, 0.0, 0.0],
        [0.1, 0.9, 0.0, 0.0],
        [0.2, 0.8, 0.0, 0.0],
    ]
    class_names = ["C0", "C1", "C2", "C3"]
    results = compute_comprehensive_metrics(y_true, y_prob, class_names=class_names)

    # Classes C2 and C3 should have 0 support and zero division safely handled
    assert results["per_class"]["C2"]["support"] == 0
    assert results["per_class"]["C2"]["precision"] == 0.0
    assert results["overall"]["accuracy"] == 1.0

