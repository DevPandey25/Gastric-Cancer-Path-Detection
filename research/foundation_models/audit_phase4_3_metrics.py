"""
Independent Audit & Metric Reconciliation Script for Phase 4.3 / 4.3.1.

This script independently recomputes all classification, calibration,
and statistical comparison metrics directly from the saved prediction
and probability CSV artifacts on the Phase 2 test split (N=4,665).

No training, no fine-tuning, no test leakage. Fully deterministic.
"""

import json
import math
import os
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, cohen_kappa_score


CANONICAL_CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CANONICAL_CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CANONICAL_CLASSES)}


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate the Wilson score confidence interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + (z ** 2) / n
    center = (p + (z ** 2) / (2 * n)) / denom
    margin = (z * math.sqrt((p * (1 - p) / n) + (z ** 2) / (4 * (n ** 2)))) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower, upper


def compute_expected_calibration_error(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE) using standard equal-width binning.
    
    Confidence = max probability across all classes.
    Prediction = argmax class.
    Accuracy per bin = fraction of correctly predicted items in bin.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(float)
    
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_count / n) * abs(bin_acc - bin_conf)
            
    return float(ece)


def compute_multiclass_brier_score(
    y_true: np.ndarray,
    probs: np.ndarray,
    num_classes: int = 8,
) -> float:
    """
    Compute the standard multiclass Brier score:
    Brier = (1/N) * sum_i sum_k (p_ik - y_ik)^2
    where y_ik is the one-hot indicator of true class.
    """
    n = len(y_true)
    one_hot = np.zeros((n, num_classes), dtype=float)
    one_hot[np.arange(n), y_true] = 1.0
    squared_diff = (probs - one_hot) ** 2
    brier = float(np.sum(squared_diff) / n)
    return brier


def compute_contingency_and_quadrants(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 8,
) -> Tuple[np.ndarray, Dict[str, Dict[str, int]]]:
    """
    Compute confusion matrix and per-class One-vs-Rest quadrants (TP, FP, FN, TN).
    
    For class k:
    TP_k = C[k, k]
    FP_k = sum_{i != k} C[i, k]
    FN_k = sum_{j != k} C[k, j]
    TN_k = sum_{i != k} sum_{j != k} C[i, j] = N - TP_k - FP_k - FN_k
    """
    n = len(y_true)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
        
    quadrants = {}
    for k in range(num_classes):
        cname = IDX_TO_CLASS[k]
        tp = int(cm[k, k])
        fp = int(np.sum(cm[:, k]) - tp)
        fn = int(np.sum(cm[k, :]) - tp)
        tn = int(n - tp - fp - fn)
        
        # Verify quadrant sum
        assert tp + fp + fn + tn == n, f"Quadrant sum {tp+fp+fn+tn} != {n} for {cname}"
        quadrants[cname] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "support": tp + fn,
        }
        
    return cm, quadrants


def compute_full_metrics_from_predictions(
    df: pd.DataFrame,
    test_manifest_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Independently compute all overall and per-class metrics from prediction DataFrame.
    """
    n = len(df)
    assert n == 4665, f"Expected 4665 rows, got {n}"
    assert len(test_manifest_df) == 4665
    
    # Verify row alignment
    assert (df["relative_path"] == test_manifest_df["relative_path"]).all(), "Path mismatch with manifest"
    assert (df["true_class"] == test_manifest_df["class_name"]).all(), "Class mismatch with manifest"
    
    y_true = np.array([CLASS_TO_IDX[c] for c in df["true_class"]], dtype=int)
    y_pred = np.array([CLASS_TO_IDX[c] for c in df["predicted_class"]], dtype=int)
    
    prob_cols = [f"prob_{c}" for c in CANONICAL_CLASSES]
    probs = df[prob_cols].values.astype(float)
    
    # Verify probability normalization
    prob_sums = np.sum(probs, axis=1)
    max_prob_sum_dev = float(np.max(np.abs(prob_sums - 1.0)))
    assert max_prob_sum_dev <= 0.001, f"Prob sums deviate by {max_prob_sum_dev}"
    
    # Quadrants & Confusion Matrix
    cm, quadrants = compute_contingency_and_quadrants(y_true, y_pred)
    assert np.sum(cm) == n, f"Sum of CM {np.sum(cm)} != {n}"
    
    # Class-level metrics
    per_class = {}
    tp_total = 0
    for k in range(8):
        cname = IDX_TO_CLASS[k]
        q = quadrants[cname]
        tp = q["tp"]
        fp = q["fp"]
        fn = q["fn"]
        tn = q["tn"]
        sup = q["support"]
        tp_total += tp
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        # OvR AUROC
        y_binary = (y_true == k).astype(int)
        class_prob = probs[:, k]
        try:
            auroc_ovr = float(roc_auc_score(y_binary, class_prob))
        except Exception:
            auroc_ovr = 0.5
            
        per_class[cname] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "support": sup,
            "precision": prec,
            "recall": rec,
            "specificity": spec,
            "f1": f1,
            "auroc_ovr": auroc_ovr,
        }
        
    # Overall metrics
    accuracy = tp_total / n
    acc_lower, acc_upper = wilson_score_interval(tp_total, n)
    
    macro_precision = float(np.mean([per_class[c]["precision"] for c in CANONICAL_CLASSES]))
    macro_recall = float(np.mean([per_class[c]["recall"] for c in CANONICAL_CLASSES]))
    macro_specificity = float(np.mean([per_class[c]["specificity"] for c in CANONICAL_CLASSES]))
    macro_f1 = float(np.mean([per_class[c]["f1"] for c in CANONICAL_CLASSES]))
    macro_auroc = float(np.mean([per_class[c]["auroc_ovr"] for c in CANONICAL_CLASSES]))
    
    # Support-weighted metrics
    supports = np.array([per_class[c]["support"] for c in CANONICAL_CLASSES], dtype=float)
    weighted_precision = float(np.sum(supports * [per_class[c]["precision"] for c in CANONICAL_CLASSES]) / n)
    weighted_recall = float(np.sum(supports * [per_class[c]["recall"] for c in CANONICAL_CLASSES]) / n)
    weighted_specificity = float(np.sum(supports * [per_class[c]["specificity"] for c in CANONICAL_CLASSES]) / n)
    weighted_f1 = float(np.sum(supports * [per_class[c]["f1"] for c in CANONICAL_CLASSES]) / n)
    weighted_auroc = float(np.sum(supports * [per_class[c]["auroc_ovr"] for c in CANONICAL_CLASSES]) / n)
    
    balanced_accuracy = macro_recall  # Algebraic identity
    # Verify weighted recall == accuracy
    assert abs(weighted_recall - accuracy) < 1e-9, f"Weighted recall {weighted_recall} != accuracy {accuracy}"
    
    cohen_kappa = float(cohen_kappa_score(y_true, y_pred))
    ece = compute_expected_calibration_error(y_true, probs, n_bins=10)
    brier = compute_multiclass_brier_score(y_true, probs, num_classes=8)
    
    return {
        "n_samples": n,
        "accuracy": accuracy,
        "accuracy_ci_95": [acc_lower, acc_upper],
        "balanced_accuracy": balanced_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_specificity": macro_specificity,
        "macro_f1": macro_f1,
        "macro_auroc": macro_auroc,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_specificity": weighted_specificity,
        "weighted_f1": weighted_f1,
        "weighted_auroc": weighted_auroc,
        "cohen_kappa": cohen_kappa,
        "ece": ece,
        "brier_score": brier,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": probs,
    }


def paired_bootstrap_test(
    model_a_data: Dict[str, Any],
    model_b_data: Dict[str, Any],
    b_iterations: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Perform paired bootstrap resampling over the identical N=4,665 test samples.
    Computes 95% bootstrap percentile CI for Delta Accuracy and Delta Macro-F1.
    """
    rng = np.random.RandomState(seed)
    n = model_a_data["n_samples"]
    y_true = model_a_data["y_true"]
    
    pred_a = model_a_data["y_pred"]
    pred_b = model_b_data["y_pred"]
    
    diff_acc = []
    diff_f1 = []
    
    for _ in range(b_iterations):
        indices = rng.choice(n, size=n, replace=True)
        
        yt_boot = y_true[indices]
        pa_boot = pred_a[indices]
        pb_boot = pred_b[indices]
        
        acc_a = np.mean(pa_boot == yt_boot)
        acc_b = np.mean(pb_boot == yt_boot)
        diff_acc.append(acc_a - acc_b)
        
        # Macro F1
        f1_a_list = []
        f1_b_list = []
        for k in range(8):
            tp_a = np.sum((yt_boot == k) & (pa_boot == k))
            fp_a = np.sum((yt_boot != k) & (pa_boot == k))
            fn_a = np.sum((yt_boot == k) & (pa_boot != k))
            f1_a = (2 * tp_a) / (2 * tp_a + fp_a + fn_a) if (2 * tp_a + fp_a + fn_a) > 0 else 0.0
            f1_a_list.append(f1_a)
            
            tp_b = np.sum((yt_boot == k) & (pb_boot == k))
            fp_b = np.sum((yt_boot != k) & (pb_boot == k))
            fn_b = np.sum((yt_boot == k) & (pb_boot != k))
            f1_b = (2 * tp_b) / (2 * tp_b + fp_b + fn_b) if (2 * tp_b + fp_b + fn_b) > 0 else 0.0
            f1_b_list.append(f1_b)
            
        diff_f1.append(np.mean(f1_a_list) - np.mean(f1_b_list))
        
    diff_acc = np.array(diff_acc)
    diff_f1 = np.array(diff_f1)
    
    # 95% Percentile CI
    acc_ci = [float(np.percentile(diff_acc, 2.5)), float(np.percentile(diff_acc, 97.5))]
    f1_ci = [float(np.percentile(diff_f1, 2.5)), float(np.percentile(diff_f1, 97.5))]
    
    # Two-sided empirical p-value
    p_val_acc = float(2 * min(np.mean(diff_acc <= 0), np.mean(diff_acc >= 0)))
    p_val_f1 = float(2 * min(np.mean(diff_f1 <= 0), np.mean(diff_f1 >= 0)))
    
    # McNemar's test for accuracy
    corr_a = (pred_a == y_true)
    corr_b = (pred_b == y_true)
    n01 = int(np.sum((~corr_a) & corr_b))  # A wrong, B right
    n10 = int(np.sum(corr_a & (~corr_b)))  # A right, B wrong
    
    mcnemar_stat = ((abs(n01 - n10) - 1) ** 2) / (n01 + n10) if (n01 + n10) > 0 else 0.0
    mcnemar_pval = float(stats.chi2.sf(mcnemar_stat, df=1)) if (n01 + n10) > 0 else 1.0
    
    return {
        "diff_acc_mean": float(np.mean(diff_acc)),
        "diff_acc_ci_95": acc_ci,
        "p_val_acc_bootstrap": p_val_acc,
        "diff_f1_mean": float(np.mean(diff_f1)),
        "diff_f1_ci_95": f1_ci,
        "p_val_f1_bootstrap": p_val_f1,
        "mcnemar_discordant": {"A_wrong_B_right": n01, "A_right_B_wrong": n10},
        "mcnemar_stat": float(mcnemar_stat),
        "mcnemar_pval": mcnemar_pval,
    }


def main():
    print("=" * 80)
    print("PHASE 4.3.1 INDEPENDENT METRIC RECONCILIATION AND SCIENTIFIC CERTIFICATION")
    print("=" * 80)
    
    test_manifest_path = "research/splits/manifest_test.csv"
    test_manifest_df = pd.read_csv(test_manifest_path)
    print(f"Loaded test manifest: {len(test_manifest_df)} rows")
    
    prediction_files = {
        "linear_probe_cls": "research/foundation_models/evaluation/linear_probe_cls_test_predictions.csv",
        "linear_probe_mean_patch": "research/foundation_models/evaluation/linear_probe_mean_patch_test_predictions.csv",
        "nonlinear_probe_cls": "research/foundation_models/evaluation/nonlinear_probe_cls_test_predictions.csv",
        "nonlinear_probe_mean_patch": "research/foundation_models/evaluation/nonlinear_probe_mean_patch_test_predictions.csv",
    }
    
    results = {}
    for name, path in prediction_files.items():
        print(f"\n--- Processing {name} ({path}) ---")
        df = pd.read_csv(path)
        metrics = compute_full_metrics_from_predictions(df, test_manifest_df)
        results[name] = metrics
        
        print(f"Accuracy:         {metrics['accuracy']*100:.2f}% (Wilson 95% CI: [{metrics['accuracy_ci_95'][0]*100:.2f}%, {metrics['accuracy_ci_95'][1]*100:.2f}%])")
        print(f"Balanced Acc:     {metrics['balanced_accuracy']*100:.2f}%")
        print(f"Macro Recall:     {metrics['macro_recall']*100:.2f}% (Identity Check: Balanced Acc == Macro Recall: {abs(metrics['balanced_accuracy'] - metrics['macro_recall']) < 1e-9})")
        print(f"Weighted Recall:  {metrics['weighted_recall']*100:.2f}% (Identity Check: Weighted Recall == Accuracy: {abs(metrics['weighted_recall'] - metrics['accuracy']) < 1e-9})")
        print(f"Macro Precision:  {metrics['macro_precision']*100:.2f}%")
        print(f"Macro F1-Score:   {metrics['macro_f1']*100:.2f}%")
        print(f"Macro Specificity:{metrics['macro_specificity']*100:.2f}%")
        print(f"Weighted F1:      {metrics['weighted_f1']*100:.2f}%")
        print(f"Cohen Kappa:      {metrics['cohen_kappa']:.4f}")
        print(f"Macro OvR AUROC:  {metrics['macro_auroc']:.4f}")
        print(f"ECE (10 bins):    {metrics['ece']:.4f}")
        print(f"Brier Score:      {metrics['brier_score']:.4f}")
        
    print("\n" + "=" * 80)
    print("PAIRED STATISTICAL COMPARISONS (B=2,000 paired bootstrap, seed=42)")
    print("=" * 80)
    
    comparisons = [
        ("CLS MLP vs CLS Linear", "nonlinear_probe_cls", "linear_probe_cls"),
        ("Mean Patch MLP vs Mean Patch Linear", "nonlinear_probe_mean_patch", "linear_probe_mean_patch"),
        ("Mean Patch MLP vs CLS MLP", "nonlinear_probe_mean_patch", "nonlinear_probe_cls"),
        ("Mean Patch Linear vs CLS Linear", "linear_probe_mean_patch", "linear_probe_cls"),
    ]
    
    comparison_results = {}
    for comp_name, mod_a, mod_b in comparisons:
        print(f"\n>>> Comparison: {comp_name}")
        comp = paired_bootstrap_test(results[mod_a], results[mod_b])
        comparison_results[comp_name] = comp
        print(f"Delta Accuracy (A - B): {comp['diff_acc_mean']*100:+.2f}% [95% CI: {comp['diff_acc_ci_95'][0]*100:+.2f}%, {comp['diff_acc_ci_95'][1]*100:+.2f}%]")
        print(f"Bootstrap p-value (Acc): {comp['p_val_acc_bootstrap']:.4f}")
        print(f"McNemar p-value (Acc):   {comp['mcnemar_pval']:.4f} (Discordant: A wrong, B right = {comp['mcnemar_discordant']['A_wrong_B_right']}, A right, B wrong = {comp['mcnemar_discordant']['A_right_B_wrong']})")
        print(f"Delta Macro-F1 (A - B): {comp['diff_f1_mean']*100:+.2f}% [95% CI: {comp['diff_f1_ci_95'][0]*100:+.2f}%, {comp['diff_f1_ci_95'][1]*100:+.2f}%]")
        print(f"Bootstrap p-value (F1):  {comp['p_val_f1_bootstrap']:.4f}")

    # Export serialized audit results
    export_payload = {
        "models": {
            name: {
                k: v for k, v in data.items() if k not in ["y_true", "y_pred", "probs"]
            }
            for name, data in results.items()
        },
        "comparisons": comparison_results,
    }
    
    os.makedirs("research/audits/phase4_3_1", exist_ok=True)
    out_json = "research/audits/phase4_3_1/reconciled_metrics.json"
    with open(out_json, "w") as f:
        json.dump(export_payload, f, indent=2)
    print(f"\nSaved reconciled audit metrics to: {out_json}")


if __name__ == "__main__":
    main()

