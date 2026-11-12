"""Downstream Comparative Evaluation on 10% Stratified Subset (Phase 4.4 Reduced).

Evaluates 4 frozen foundation model representations on exactly matching samples:
1. Phikon CLS (d=768)
2. Phikon Mean Spatial Patch (d=768)
3. DINOv2 CLS (d=384)
4. DINOv2 Mean Spatial Patch (d=384)

Protocol:
- Train split (N=2,177): fitting linear probes.
- Validation split (N=466): 12-point C sweep optimizing unweighted Macro-F1.
- Test split (N=467): evaluated strictly once.
- Metrics: Accuracy + Wilson 95% CI, Balanced Accuracy, Macro-F1 + Bootstrap 95% CI,
  Weighted-F1, Cohen's Kappa, Confusion Matrix, Per-class Sensitivity/Recall, Specificity,
  OvR AUROC, ECE (10 bins), multiclass Brier score.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from research.benchmark.metrics import compute_comprehensive_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("subset_evaluator")

CANONICAL_CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]
C_CANDIDATES = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]


class SubsetEvaluator:
    """Evaluates the 4 frozen representations on the 10% stratified subset (N=3,110)."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        if workspace_root is None:
            self.workspace_root = Path(__file__).resolve().parent.parent.parent.parent
        else:
            self.workspace_root = Path(workspace_root).resolve()

        self.subset_dir = self.workspace_root / "research" / "foundation_models" / "cache" / "subset10pct"
        self.manifest_path = self.subset_dir / "manifest_subset.csv"
        self.output_eval_dir = self.subset_dir / "evaluation"
        self.output_eval_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir = self.subset_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing subset manifest: {self.manifest_path}")

        self.df_manifest = pd.read_csv(self.manifest_path)
        self.n_samples = len(self.df_manifest)
        assert self.n_samples == 3110

        self.train_idx = np.where(self.df_manifest["split"] == "train")[0]
        self.val_idx = np.where(self.df_manifest["split"] == "val")[0]
        self.test_idx = np.where(self.df_manifest["split"] == "test")[0]

        self.y_train = self.df_manifest.iloc[self.train_idx]["class_index"].to_numpy()
        self.y_val = self.df_manifest.iloc[self.val_idx]["class_index"].to_numpy()
        self.y_test = self.df_manifest.iloc[self.test_idx]["class_index"].to_numpy()

        logger.info(
            "Loaded 10%% subset manifest: Total = %d | Train = %d | Val = %d | Test = %d",
            self.n_samples, len(self.train_idx), len(self.val_idx), len(self.test_idx),
        )

        self.models_config = {
            "phikon_cls": {
                "name": "Phikon CLS",
                "model_family": "owkin/phikon (ViT-B/16, TCGA iBOT)",
                "dim": 768,
                "path": self.subset_dir / "phikon" / "cls_embeddings.npy",
            },
            "phikon_mean_patch": {
                "name": "Phikon Mean Patch",
                "model_family": "owkin/phikon (ViT-B/16, TCGA iBOT)",
                "dim": 768,
                "path": self.subset_dir / "phikon" / "mean_patch_embeddings.npy",
            },
            "dinov2_cls": {
                "name": "DINOv2 CLS",
                "model_family": "facebookresearch/dinov2 (ViT-S/14, LVD-142M)",
                "dim": 384,
                "path": self.subset_dir / "dinov2_vits14" / "cls_embeddings.npy",
            },
            "dinov2_mean_patch": {
                "name": "DINOv2 Mean Patch",
                "model_family": "facebookresearch/dinov2 (ViT-S/14, LVD-142M)",
                "dim": 384,
                "path": self.subset_dir / "dinov2_vits14" / "mean_patch_embeddings.npy",
            },
        }

    def run_evaluation(self) -> dict[str, Any]:
        logger.info("=" * 80)
        logger.info("GASTRICPATH AI — 10%% STRATIFIED SUBSET COMPARATIVE EVALUATION")
        logger.info("Evaluating 4 Representations: Phikon vs DINOv2 (Train: 2,177, Val: 466, Test: 467)")
        logger.info("=" * 80)

        all_results = {}
        all_test_preds = {}

        for key, cfg in self.models_config.items():
            logger.info("Evaluating representation: %s (%s, d=%d)...", cfg["name"], cfg["model_family"], cfg["dim"])
            mm = np.memmap(cfg["path"], dtype=np.float32, mode="r", shape=(self.n_samples, cfg["dim"]))
            X_tr = mm[self.train_idx]
            X_va = mm[self.val_idx]
            X_te = mm[self.test_idx]

            # 1. Hyperparameter sweep on Validation ONLY
            best_c = C_CANDIDATES[0]
            best_val_f1 = -1.0
            tuning_grid = []

            for c in C_CANDIDATES:
                t0 = time.time()
                clf = LogisticRegression(C=c, max_iter=1000, solver="lbfgs", random_state=42, tol=1e-4)
                clf.fit(X_tr, self.y_train)
                dur = time.time() - t0
                val_preds = clf.predict(X_va)
                val_acc = float(accuracy_score(self.y_val, val_preds))
                val_f1 = float(f1_score(self.y_val, val_preds, average="macro", zero_division=0))

                tuning_grid.append({
                    "C": float(c), "fit_time_sec": round(dur, 3),
                    "val_accuracy": round(val_acc, 4), "val_macro_f1": round(val_f1, 4),
                })
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_c = c

            logger.info("  [%s] Selected C* = %.3f (Val Macro-F1 = %.2f%%)", key.upper(), best_c, best_val_f1 * 100)

            # 2. Final Test evaluation (strictly once)
            t_fit_0 = time.time()
            final_clf = LogisticRegression(C=best_c, max_iter=1000, solver="lbfgs", random_state=42)
            final_clf.fit(X_tr, self.y_train)
            fit_time = time.time() - t_fit_0

            t_inf_0 = time.time()
            test_probs = final_clf.predict_proba(X_te)
            test_preds = final_clf.predict(X_te)
            inf_time = time.time() - t_inf_0
            all_test_preds[key] = test_preds

            # 3. Comprehensive metrics
            metrics = compute_comprehensive_metrics(
                y_true=self.y_test,
                y_prob=test_probs,
                class_names=CANONICAL_CLASSES,
                compute_ci=True,
            )

            # 4. Save predictions CSV
            pred_records = []
            test_df_subset = self.df_manifest.iloc[self.test_idx].reset_index(drop=True)
            for i, row in test_df_subset.iterrows():
                rec = {
                    "subset_test_index": int(i),
                    "master_row_index": int(row["row_index"]),
                    "relative_path": row["relative_path"],
                    "true_class": row["class_name"],
                    "true_index": int(row["class_index"]),
                    "predicted_class": CANONICAL_CLASSES[int(test_preds[i])],
                    "predicted_index": int(test_preds[i]),
                    "confidence": round(float(np.max(test_probs[i])), 4),
                    "is_correct": bool(row["class_index"] == test_preds[i]),
                }
                for c_idx, c_name in enumerate(CANONICAL_CLASSES):
                    rec[f"prob_{c_name}"] = round(float(test_probs[i, c_idx]), 4)
                pred_records.append(rec)

            df_preds = pd.DataFrame(pred_records)
            pred_csv_path = self.output_eval_dir / f"{key}_test_predictions.csv"
            df_preds.to_csv(pred_csv_path, index=False)

            overall = metrics["overall"]
            result_item = {
                "key": key,
                "name": cfg["name"],
                "model_family": cfg["model_family"],
                "embedding_dim": cfg["dim"],
                "optimal_C": float(best_c),
                "val_macro_f1": round(float(best_val_f1), 4),
                "fit_time_seconds": round(fit_time, 3),
                "test_inference_seconds": round(inf_time, 3),
                "test_sample_count": len(self.test_idx),
                "overall_metrics": {
                    "accuracy": overall["accuracy"],
                    "accuracy_ci_95": overall["accuracy_ci_95"],
                    "balanced_accuracy": overall["balanced_accuracy"],
                    "balanced_accuracy_ci_95": overall["balanced_accuracy_ci_95"],
                    "macro_precision": overall["macro_precision"],
                    "macro_recall": overall["macro_recall"],
                    "macro_f1": overall["macro_f1"],
                    "macro_f1_ci_95": overall["macro_f1_ci_95"],
                    "weighted_precision": overall["weighted_precision"],
                    "weighted_recall": overall["weighted_recall"],
                    "weighted_f1": overall["weighted_f1"],
                    "cohen_kappa": overall["cohen_kappa"],
                    "macro_ovr_auroc": overall["macro_auroc"],
                    "weighted_ovr_auroc": overall["weighted_auroc"],
                    "expected_calibration_error": overall["expected_calibration_error"],
                    "brier_score": overall["brier_score"],
                },
                "per_class_metrics": metrics["per_class"],
                "confusion_matrix": metrics["confusion_matrix"],
                "tuning_grid": tuning_grid,
            }

            all_results[key] = result_item
            logger.info(
                "  [%s] TEST ACC: %6.2f%% | BAL ACC: %6.2f%% | MACRO-F1: %6.2f%% | AUROC: %.4f | ECE: %.4f",
                key.upper(),
                overall["accuracy"] * 100,
                overall["balanced_accuracy"] * 100,
                overall["macro_f1"] * 100,
                overall["macro_auroc"],
                overall["expected_calibration_error"],
            )

        # 5. Generate Figures
        self._generate_visualizations(all_results)

        # 6. Save comprehensive evaluation JSON
        summary_out = {
            "evaluation_scope": "Phase 4.4 Reduced 10% Stratified Subset Benchmark",
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "subset_total_samples": self.n_samples,
            "train_samples": len(self.train_idx),
            "val_samples": len(self.val_idx),
            "test_samples": len(self.test_idx),
            "results": all_results,
        }
        with open(self.output_eval_dir / "evaluation_summary.json", "w") as f:
            json.dump(summary_out, f, indent=2)

        logger.info("Saved complete evaluation summary to %s", (self.output_eval_dir / "evaluation_summary.json").name)
        return summary_out

    def _generate_visualizations(self, results: dict[str, Any]) -> None:
        # Figure 1: Per-Class Recall Comparison
        fig, ax = plt.subplots(figsize=(12, 5.5))
        x = np.arange(len(CANONICAL_CLASSES))
        width = 0.20

        colors = ["#7293CB", "#E1974C", "#84BA5B", "#D35E60"]
        keys = ["dinov2_cls", "dinov2_mean_patch", "phikon_cls", "phikon_mean_patch"]
        labels = ["DINOv2 CLS", "DINOv2 Mean Patch", "Phikon CLS", "Phikon Mean Patch"]

        for i, (k, lbl, col) in enumerate(zip(keys, labels, colors)):
            recalls = [results[k]["per_class_metrics"][c]["recall_sensitivity"] * 100 for c in CANONICAL_CLASSES]
            ax.bar(x + (i - 1.5) * width, recalls, width, label=lbl, color=col)

        ax.set_ylabel("Test Sensitivity / Recall (%)", fontsize=11)
        ax.set_title("Per-Class Recall: Pathology Pretraining (Phikon) vs Generic (DINOv2) on 10% Subset (N=467)", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(CANONICAL_CLASSES, fontsize=10, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")
        ax.set_ylim(0, 100)
        plt.tight_layout()
        fig.savefig(self.viz_dir / "subset_per_class_recall_comparison.png", dpi=300)
        plt.close(fig)

        # Figure 2: Confusion Matrices (2x2 grid)
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for idx, (k, lbl) in enumerate(zip(keys, labels)):
            ax = axes[idx]
            cm = np.array(results[k]["confusion_matrix"])
            cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12) * 100
            im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")

            ax.set(
                xticks=np.arange(len(CANONICAL_CLASSES)),
                yticks=np.arange(len(CANONICAL_CLASSES)),
                xticklabels=CANONICAL_CLASSES,
                yticklabels=CANONICAL_CLASSES,
                title=f"{lbl} Confusion Matrix (Acc: {results[k]['overall_metrics']['accuracy']*100:.1f}%)",
                ylabel="True Class",
                xlabel="Predicted Class",
            )
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            thresh = cm_norm.max() / 2.0
            for r in range(len(CANONICAL_CLASSES)):
                for c in range(len(CANONICAL_CLASSES)):
                    val = cm_norm[r, c]
                    raw = cm[r, c]
                    txt_col = "white" if val > thresh else "black"
                    ax.text(c, r, f"{val:.0f}%\n({raw})", ha="center", va="center", color=txt_col, fontsize=7)

        plt.tight_layout()
        fig.savefig(self.viz_dir / "subset_confusion_matrices.png", dpi=300)
        plt.close(fig)
        logger.info("Saved evaluation visualizations to %s", self.viz_dir)


if __name__ == "__main__":
    evaluator = SubsetEvaluator()
    evaluator.run_evaluation()

