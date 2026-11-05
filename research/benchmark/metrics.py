"""Comprehensive Multiclass Classification Metrics for GastricPath AI.

Implements accuracy, balanced accuracy, macro/weighted precision, recall, F1, Cohen's kappa,
multiclass confusion matrix, per-class sensitivity, specificity, support, One-vs-Rest AUROC,
macro/weighted AUROC, Brier score, and Expected Calibration Error (ECE).
"""

from __future__ import annotations

from typing import Any
import numpy as np
from sklearn import metrics as sk_metrics

from research.benchmark.dataset import CANONICAL_CLASSES


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, list[dict[str, Any]]]:
    """Calculates Expected Calibration Error (ECE) and bin-wise reliability diagram metrics.

    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)
    bin_details = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            bin_error = abs(bin_acc - bin_conf)
            ece += (bin_count / n_samples) * bin_error

            bin_details.append({
                "bin_index": i,
                "range": [round(float(bin_lower), 2), round(float(bin_upper), 2)],
                "count": bin_count,
                "accuracy": round(bin_acc, 4),
                "confidence": round(bin_conf, 4),
                "error": round(bin_error, 4),
            })

    return round(float(ece), 4), bin_details


def compute_multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int = 8) -> float:
    """Calculates multiclass Brier score: (1/N) * sum_i sum_k (p_ik - y_ik)^2."""
    n_samples = len(y_true)
    if n_samples == 0:
        return 0.0
    one_hot = np.zeros((n_samples, num_classes), dtype=float)
    one_hot[np.arange(n_samples), y_true] = 1.0
    brier = np.mean(np.sum((y_prob - one_hot) ** 2, axis=1))
    return round(float(brier), 4)


def compute_comprehensive_metrics(
    y_true: list[int] | np.ndarray,
    y_prob: list[list[float]] | np.ndarray,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Computes full suite of overall and per-class computational pathology metrics."""
    if class_names is None:
        class_names = CANONICAL_CLASSES

    num_classes = len(class_names)
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    if len(y_true_arr) == 0:
        raise ValueError("Cannot compute metrics on empty ground truth.")

    y_pred_arr = np.argmax(y_prob_arr, axis=1)

    # 1. Overall Metrics
    acc = sk_metrics.accuracy_score(y_true_arr, y_pred_arr)
    bal_acc = sk_metrics.balanced_accuracy_score(y_true_arr, y_pred_arr)
    macro_p, macro_r, macro_f1, _ = sk_metrics.precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = sk_metrics.precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="weighted", zero_division=0
    )
    try:
        kappa = float(sk_metrics.cohen_kappa_score(y_true_arr, y_pred_arr, labels=list(range(num_classes))))
        if np.isnan(kappa):
            kappa = 0.0
    except Exception:
        kappa = 0.0
    cm = sk_metrics.confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(num_classes)))

    # 2. Per-Class Specificity, Precision, Recall, F1, Support
    per_class_p, per_class_r, per_class_f1, per_class_sup = sk_metrics.precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=list(range(num_classes)), zero_division=0
    )

    per_class_metrics: dict[str, dict[str, Any]] = {}
    total_samples = len(y_true_arr)

    # Calculate per-class Specificity = TN / (TN + FP)
    for c_idx, c_name in enumerate(class_names):
        tp = cm[c_idx, c_idx]
        fn = np.sum(cm[c_idx, :]) - tp
        fp = np.sum(cm[:, c_idx]) - tp
        tn = total_samples - (tp + fn + fp)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # One-vs-Rest AUROC for this class
        ovr_auroc: float | None = None
        c_true = (y_true_arr == c_idx).astype(int)
        if len(np.unique(c_true)) > 1:
            try:
                ovr_auroc = round(float(sk_metrics.roc_auc_score(c_true, y_prob_arr[:, c_idx])), 4)
            except Exception:
                ovr_auroc = None

        per_class_metrics[c_name] = {
            "class_index": c_idx,
            "precision": round(float(per_class_p[c_idx]), 4),
            "recall_sensitivity": round(float(per_class_r[c_idx]), 4),
            "specificity": round(float(specificity), 4),
            "f1_score": round(float(per_class_f1[c_idx]), 4),
            "support": int(per_class_sup[c_idx]),
            "auroc_ovr": ovr_auroc,
        }

    # 3. Macro & Weighted AUROC
    macro_auroc: float | None = None
    weighted_auroc: float | None = None
    if len(np.unique(y_true_arr)) > 1:
        try:
            macro_auroc = round(float(sk_metrics.roc_auc_score(y_true_arr, y_prob_arr, multi_class="ovr", average="macro")), 4)
            weighted_auroc = round(float(sk_metrics.roc_auc_score(y_true_arr, y_prob_arr, multi_class="ovr", average="weighted")), 4)
        except Exception:
            pass

    # 4. Calibration & Brier Score
    ece, ece_bins = compute_expected_calibration_error(y_true_arr, y_prob_arr)
    brier = compute_multiclass_brier_score(y_true_arr, y_prob_arr, num_classes=num_classes)

    return {
        "overall": {
            "total_samples": total_samples,
            "accuracy": round(float(acc), 4),
            "balanced_accuracy": round(float(bal_acc), 4),
            "macro_precision": round(float(macro_p), 4),
            "macro_recall": round(float(macro_r), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_precision": round(float(weighted_p), 4),
            "weighted_recall": round(float(weighted_r), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "cohen_kappa": round(float(kappa), 4),
            "macro_auroc": macro_auroc,
            "weighted_auroc": weighted_auroc,
            "expected_calibration_error": ece,
            "brier_score": brier,
        },
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "ece_bins": ece_bins,
        "class_labels": class_names,
    }

