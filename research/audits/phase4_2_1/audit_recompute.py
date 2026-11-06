"""Independent Metric Recomputation and Audit for Phase 4.2.1.

Recomputes all overall and per-class metrics directly from the stored test prediction CSV files:
- research/foundation_models/evaluation/linear_probe_cls_test_predictions.csv
- research/foundation_models/evaluation/linear_probe_mean_patch_test_predictions.csv

Compares with reported metrics in:
- research/foundation_models/evaluation/linear_probe_cls_results.json
- research/foundation_models/evaluation/linear_probe_mean_patch_results.json
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics

from research.benchmark.metrics import (
    CANONICAL_CLASSES,
    compute_comprehensive_metrics,
    wilson_score_interval,
    compute_expected_calibration_error,
    compute_multiclass_brier_score,
)


def audit_and_recompute():
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    eval_dir = repo_root / "research" / "foundation_models" / "evaluation"
    audit_out_dir = repo_root / "research" / "audits" / "phase4_2_1"
    audit_out_dir.mkdir(parents=True, exist_ok=True)

    cls_csv = eval_dir / "linear_probe_cls_test_predictions.csv"
    patch_csv = eval_dir / "linear_probe_mean_patch_test_predictions.csv"
    cls_json = eval_dir / "linear_probe_cls_results.json"
    patch_json = eval_dir / "linear_probe_mean_patch_results.json"

    assert cls_csv.exists(), f"Missing {cls_csv}"
    assert patch_csv.exists(), f"Missing {patch_csv}"
    assert cls_json.exists(), f"Missing {cls_json}"
    assert patch_json.exists(), f"Missing {patch_json}"

    with open(cls_json, "r") as f:
        reported_cls = json.load(f)
    with open(patch_json, "r") as f:
        reported_patch = json.load(f)

    audit_results = {}

    for name, csv_path, rep_data in [
        ("cls", cls_csv, reported_cls),
        ("mean_patch", patch_csv, reported_patch),
    ]:
        print(f"\n=======================================================")
        print(f"AUDITING {name.upper()} TEST PREDICTIONS")
        print(f"=======================================================")
        df = pd.read_csv(csv_path)
        n_samples = len(df)
        print(f"Total test rows: {n_samples}")
        assert n_samples == 4665, f"Expected 4665 test rows, got {n_samples}"

        # Canonical class check
        prob_cols = [f"prob_{c}" for c in CANONICAL_CLASSES]
        for col in prob_cols:
            assert col in df.columns, f"Missing {col} in CSV"

        y_true = df["true_index"].to_numpy()
        y_prob = df[prob_cols].to_numpy()

        # Check probability normalization (sum to ~1.0)
        prob_sums = np.sum(y_prob, axis=1)
        max_sum_diff = np.max(np.abs(prob_sums - 1.0))
        print(f"Max deviation from sum(prob)=1.0: {max_sum_diff:.6f}")

        # Compute full comprehensive metrics independently
        recomputed = compute_comprehensive_metrics(
            y_true=y_true,
            y_prob=y_prob,
            class_names=CANONICAL_CLASSES,
            compute_ci=True,
        )

        overall_rep = rep_data["overall_metrics"]
        overall_rec = recomputed["overall"]

        discrepancies = []
        for k in [
            "accuracy",
            "balanced_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1",
            "cohen_kappa",
            "expected_calibration_error",
            "brier_score",
        ]:
            val_rep = overall_rep[k]
            val_rec = overall_rec[k]
            diff = abs(val_rep - val_rec)
            print(f"  {k:28s}: Reported = {val_rep:7.4f} | Recomputed = {val_rec:7.4f} | Diff = {diff:.6f}")
            # ECE evaluated from 4-decimal CSV probabilities carries up to 0.0005 quantization variance
            tolerance = 5e-4 if k == "expected_calibration_error" else 1e-4
            if diff > tolerance:
                discrepancies.append((k, val_rep, val_rec, diff))

        # Check AUROC
        val_rep_auroc = overall_rep["macro_ovr_auroc"]
        val_rec_auroc = overall_rec["macro_auroc"]
        diff_auroc = abs(val_rep_auroc - val_rec_auroc)
        print(f"  {'macro_ovr_auroc':28s}: Reported = {val_rep_auroc:7.4f} | Recomputed = {val_rec_auroc:7.4f} | Diff = {diff_auroc:.6f}")
        if diff_auroc > 1e-4:
            discrepancies.append(("macro_ovr_auroc", val_rep_auroc, val_rec_auroc, diff_auroc))

        # Check per-class metrics
        print("\n  Per-Class Metrics Audit:")
        per_class_table = []
        for c in CANONICAL_CLASSES:
            rec_c = recomputed["per_class"][c]
            rep_c = rep_data["per_class_metrics"][c]
            
            p_rec, p_rep = rec_c["precision"], rep_c["precision"]
            r_rec, r_rep = rec_c["recall_sensitivity"], rep_c["recall_sensitivity"]
            s_rec, s_rep = rec_c["specificity"], rep_c["specificity"]
            f_rec, f_rep = rec_c["f1_score"], rep_c["f1_score"]
            sup = rec_c["support"]

            row_entry = {
                "class": c,
                "support": sup,
                "precision": p_rec,
                "recall": r_rec,
                "specificity": s_rec,
                "f1_score": f_rec,
                "auroc": rec_c["auroc_ovr"],
            }
            per_class_table.append(row_entry)

            print(
                f"    {c:4s} (N={sup}) | "
                f"P: Rep={p_rep:.4f}, Rec={p_rec:.4f} | "
                f"R: Rep={r_rep:.4f}, Rec={r_rec:.4f} | "
                f"S: Rep={s_rep:.4f}, Rec={s_rec:.4f} | "
                f"F1: Rep={f_rep:.4f}, Rec={f_rec:.4f}"
            )
            for m_name, v_rep, v_rec in [
                ("precision", p_rep, p_rec),
                ("recall", r_rep, r_rec),
                ("specificity", s_rep, s_rec),
                ("f1_score", f_rep, f_rec),
            ]:
                if abs(v_rep - v_rec) > 1e-4:
                    discrepancies.append((f"{c}_{m_name}", v_rep, v_rec, abs(v_rep - v_rec)))

        audit_results[name] = {
            "discrepancies": discrepancies,
            "recomputed_overall": overall_rec,
            "recomputed_per_class": recomputed["per_class"],
            "confusion_matrix": recomputed["confusion_matrix"],
            "per_class_table": per_class_table,
        }

    out_file = audit_out_dir / "audit_metrics.json"
    with open(out_file, "w") as f:
        json.dump(audit_results, f, indent=2)
    print(f"\nAudit complete. Recomputed metrics written to {out_file}")
    return audit_results


if __name__ == "__main__":
    audit_and_recompute()

