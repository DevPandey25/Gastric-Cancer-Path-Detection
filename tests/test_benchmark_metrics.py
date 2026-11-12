"""Tests for Comprehensive Multiclass Benchmark Metrics and Statistical Edge Cases."""

import numpy as np
import pytest

from research.benchmark.metrics import (
    compute_comprehensive_metrics,
    compute_expected_calibration_error,
    compute_multiclass_brier_score,
    wilson_score_interval,
    compute_bootstrap_confidence_intervals,
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
    assert overall["accuracy_ci_95"] == [0.5101, 1.0]  # Wilson interval on n=4
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
        assert results["per_class"][c]["sensitivity_ci_95"] == [0.2065, 1.0]


def test_completely_incorrect_predictions():
    # 4 classes, all shifted by 1 (0% correct)
    y_true = [0, 1, 2, 3]
    y_prob = [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
    ]
    class_names = ["C0", "C1", "C2", "C3"]

    results = compute_comprehensive_metrics(y_true, y_prob, class_names=class_names)
    overall = results["overall"]

    assert overall["accuracy"] == 0.0
    assert overall["accuracy_ci_95"] == [0.0, 0.4899]
    assert overall["macro_f1"] == 0.0
    assert overall["balanced_accuracy"] == 0.0
    assert overall["cohen_kappa"] == -0.3333  # Worse than chance
    assert overall["brier_score"] == 2.0      # (1 - 0)^2 + (0 - 1)^2 = 2.0
    assert overall["expected_calibration_error"] == 1.0  # Confident (1.0) but 0% accurate


def test_degenerate_uniform_confidence():
    # Equal uniform probabilities (0.25) across 4 classes
    y_true = [0, 1, 2, 3]
    y_prob = [
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25],
    ]
    class_names = ["C0", "C1", "C2", "C3"]

    results = compute_comprehensive_metrics(y_true, y_prob, class_names=class_names)
    overall = results["overall"]

    # argmax picks first class (index 0) for all, so 1/4 correct
    assert overall["accuracy"] == 0.25
    assert overall["expected_calibration_error"] == 0.0  # acc=0.25, conf=0.25, error=0!


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


def test_metrics_handles_unseen_classes_and_undefined_auroc():
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
    assert results["per_class"]["C2"]["auroc_ovr"] is None  # Undefined AUROC handled as None
    assert results["overall"]["accuracy"] == 1.0
    # Macro AUROC should average only over valid observed classes (C0 and C1)
    assert results["overall"]["macro_auroc"] == 1.0


def test_wilson_score_interval_properties():
    # Bounds check [0, 1]
    lower, upper = wilson_score_interval(0, 100)
    assert lower == 0.0
    assert upper > 0.0

    lower, upper = wilson_score_interval(100, 100)
    assert lower < 1.0
    assert upper == 1.0

    # Symmetric around 0.5
    lower, upper = wilson_score_interval(50, 100)
    assert round(0.5 - lower, 3) == round(upper - 0.5, 3)


def test_bootstrap_confidence_intervals():
    y_true = np.array([0, 1, 2, 3] * 10)
    y_pred = np.array([0, 1, 2, 3] * 10)
    cis = compute_bootstrap_confidence_intervals(y_true, y_pred, n_bootstraps=100, seed=42)

    assert cis["macro_f1_ci_95"] == (1.0, 1.0)
    assert cis["balanced_accuracy_ci_95"] == (1.0, 1.0)
