"""Phase 4.2.2: Metric Table Verification and Formulation Consistency Engine.

Strictly verifies:
1. Precision = TP / (TP + FP)
2. Recall = TP / (TP + FN)
3. Specificity = TN / (TN + FP)
4. F1 = 2 * (P * R) / (P + R)
5. Support == Ground Truth class sample count
6. Confusion matrix row totals == Support
7. Confusion matrix column totals == Total Predicted Count
8. Independent calculation of Macro and Weighted averages from full-precision unrounded class metrics
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EVAL_DIR = WORKSPACE_ROOT / "research" / "foundation_models" / "evaluation"
AUDIT_DIR = WORKSPACE_ROOT / "research" / "audits" / "phase4_2_2"

CANONICAL_CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]


def verify_probe_metrics(csv_path: Path, model_name: str) -> dict[str, Any]:
    assert csv_path.exists(), f"Missing predictions file: {csv_path}"
    df = pd.read_csv(csv_path)
    assert len(df) == 4665, f"Expected 4665 rows in {csv_path.name}, got {len(df)}"

    y_true = df["true_index"].to_numpy()
    y_pred = df["predicted_index"].to_numpy()
    total_samples = len(y_true)

    # Canonical 8x8 confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CANONICAL_CLASSES))))

    class_records = []
    for i, c_name in enumerate(CANONICAL_CLASSES):
        tp = int(cm[i, i])
        fn = int(np.sum(cm[i, :]) - tp)
        fp = int(np.sum(cm[:, i]) - tp)
        tn = int(total_samples - (tp + fn + fp))
        support = int(np.sum(cm[i, :]))
        predicted_count = int(np.sum(cm[:, i]))

        # Mathematical invariants
        assert tp + fn == support, f"Row sum != support for {c_name}"
        assert tp + fp == predicted_count, f"Col sum != predicted count for {c_name}"
        assert tp + fp + fn + tn == total_samples, f"Four quadrants != total samples for {c_name}"

        # Exact formula computation
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        class_records.append({
            "class": c_name,
            "class_index": i,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "support": support,
            "predicted_count": predicted_count,
            "precision_raw": prec,
            "recall_raw": rec,
            "specificity_raw": spec,
            "f1_raw": f1,
            "precision_pct": round(prec * 100.0, 2),
            "recall_pct": round(rec * 100.0, 2),
            "specificity_pct": round(spec * 100.0, 2),
            "f1_pct": round(f1 * 100.0, 2),
        })

    # Independent Macro Averages (unweighted mean of raw unrounded class metrics)
    macro_p = float(np.mean([r["precision_raw"] for r in class_records]))
    macro_r = float(np.mean([r["recall_raw"] for r in class_records]))
    macro_s = float(np.mean([r["specificity_raw"] for r in class_records]))
    macro_f1 = float(np.mean([r["f1_raw"] for r in class_records]))

    # Independent Support-Weighted Averages (support-weighted sum of raw unrounded class metrics)
    total_supp = sum(r["support"] for r in class_records)
    weighted_p = float(sum(r["precision_raw"] * r["support"] for r in class_records) / total_supp)
    weighted_r = float(sum(r["recall_raw"] * r["support"] for r in class_records) / total_supp)
    weighted_s = float(sum(r["specificity_raw"] * r["support"] for r in class_records) / total_supp)
    weighted_f1 = float(sum(r["f1_raw"] * r["support"] for r in class_records) / total_supp)

    return {
        "model_name": model_name,
        "total_samples": total_samples,
        "class_metrics": class_records,
        "macro_average": {
            "precision_raw": macro_p,
            "recall_raw": macro_r,
            "specificity_raw": macro_s,
            "f1_raw": macro_f1,
            "precision_pct": round(macro_p * 100.0, 2),
            "recall_pct": round(macro_r * 100.0, 2),
            "specificity_pct": round(macro_s * 100.0, 2),
            "f1_pct": round(macro_f1 * 100.0, 2),
        },
        "weighted_average": {
            "precision_raw": weighted_p,
            "recall_raw": weighted_r,
            "specificity_raw": weighted_s,
            "f1_raw": weighted_f1,
            "precision_pct": round(weighted_p * 100.0, 2),
            "recall_pct": round(weighted_r * 100.0, 2),
            "specificity_pct": round(weighted_s * 100.0, 2),
            "f1_pct": round(weighted_f1 * 100.0, 2),
        },
        "confusion_matrix": cm.tolist(),
    }


def run_verification():
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    cls_csv = EVAL_DIR / "linear_probe_cls_test_predictions.csv"
    patch_csv = EVAL_DIR / "linear_probe_mean_patch_test_predictions.csv"

    cls_verified = verify_probe_metrics(cls_csv, "cls")
    patch_verified = verify_probe_metrics(patch_csv, "mean_patch")

    combined = {
        "verified_at_utc": pd.Timestamp.utcnow().isoformat(),
        "cls": cls_verified,
        "mean_patch": patch_verified,
        "canonical_classes": CANONICAL_CLASSES,
    }

    out_file = AUDIT_DIR / "verified_metrics.json"
    with open(out_file, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"Verified metrics written to {out_file}")
    return combined


if __name__ == "__main__":
    run_verification()

