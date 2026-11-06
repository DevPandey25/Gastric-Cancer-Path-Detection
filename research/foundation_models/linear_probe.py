"""Downstream Linear Probing & Evaluation Engine for GastricPath AI (Phase 4.2).

Evaluates the classification utility of frozen DINOv2 ViT-S/14 representations:
A. CLS token embeddings (shape [31096, 384])
B. Mean spatial patch-token embeddings (shape [31096, 384])

Methodology:
- Encoder is strictly frozen; zero encoder parameters are loaded for gradient updates.
- Train split (N=21,767) is used exclusively for fitting linear probes.
- Validation split (N=4,664) is used exclusively for hyperparameter tuning & model selection.
- Test split (N=4,665) is evaluated exactly once on the final selected model.
- Fixed random seed = 42 for strict determinism.
- Computes comprehensive multiclass classification metrics with Wilson and bootstrap CIs.
- Explicitly documents that comparisons to historical baselines carry split confounding caveats.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from research.benchmark.metrics import compute_comprehensive_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CANONICAL_CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]


class PyTorchLinearSoftmax(nn.Module):
    """Linear softmax probe implemented natively in PyTorch."""

    def __init__(self, in_features: int = 384, num_classes: int = 8) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class LinearProbingEngine:
    """Manages frozen linear probe training, tuning, and rigorous test evaluation."""

    def __init__(
        self,
        workspace_root: Path | str | None = None,
        model_name: str = "dinov2_vits14",
    ) -> None:
        if workspace_root is None:
            self.workspace_root = Path(__file__).resolve().parent.parent.parent
        else:
            self.workspace_root = Path(workspace_root).resolve()

        self.cache_dir = self.workspace_root / "research" / "foundation_models" / "cache" / "full" / model_name
        self.eval_dir = self.workspace_root / "research" / "foundation_models" / "evaluation"
        self.viz_dir = self.workspace_root / "research" / "foundation_models" / "visualizations"
        self.ckpt_dir = self.workspace_root / "research" / "foundation_models" / "checkpoints"

        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.cache_dir / "manifest.csv"
        self.cls_path = self.cache_dir / "cls_embeddings.npy"
        self.patch_path = self.cache_dir / "mean_patch_embeddings.npy"

        logger.info("Loading manifest from %s...", self.manifest_path.name)
        self.df_manifest = pd.read_csv(self.manifest_path)
        self.total_samples = len(self.df_manifest)

        # Split indices
        self.train_idx = np.where(self.df_manifest["split"] == "train")[0]
        self.val_idx = np.where(self.df_manifest["split"] == "val")[0]
        self.test_idx = np.where(self.df_manifest["split"] == "test")[0]

        self.y_train = self.df_manifest.iloc[self.train_idx]["class_index"].to_numpy()
        self.y_val = self.df_manifest.iloc[self.val_idx]["class_index"].to_numpy()
        self.y_test = self.df_manifest.iloc[self.test_idx]["class_index"].to_numpy()

        logger.info(
            "Splits: Train = %d | Val = %d | Test = %d",
            len(self.train_idx),
            len(self.val_idx),
            len(self.test_idx),
        )

    def tune_hyperparameters(
        self,
        candidate_c_values: list[float] | None = None,
    ) -> dict[str, Any]:
        """Sweeps regularization parameter C on validation split only."""
        if candidate_c_values is None:
            candidate_c_values = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]

        logger.info("=" * 70)
        logger.info("HYPERPARAMETER TUNING ON VALIDATION SPLIT (TRAIN-ONLY FITTING)")
        logger.info("=" * 70)

        tuning_results: dict[str, Any] = {"candidate_c_values": candidate_c_values, "cls": {}, "mean_patch": {}}

        for repr_type, path in [("cls", self.cls_path), ("mean_patch", self.patch_path)]:
            logger.info("Tuning on %s embeddings...", repr_type.upper())
            mm = np.memmap(path, dtype=np.float32, mode="r", shape=(self.total_samples, 384))
            X_tr = mm[self.train_idx]
            X_va = mm[self.val_idx]

            best_c = candidate_c_values[0]
            best_f1 = -1.0
            grid_records = []

            for c in candidate_c_values:
                t0 = time.time()
                clf = LogisticRegression(
                    C=c,
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=42,
                    tol=1e-4,
                )
                clf.fit(X_tr, self.y_train)
                dur = time.time() - t0
                n_iter = int(clf.n_iter_[0])
                converged = n_iter < 1000

                val_preds = clf.predict(X_va)
                val_acc = float(accuracy_score(self.y_val, val_preds))
                val_f1 = float(f1_score(self.y_val, val_preds, average="macro", zero_division=0))

                rec = {
                    "C": float(c),
                    "fit_duration_sec": round(dur, 3),
                    "iterations": n_iter,
                    "converged": converged,
                    "val_accuracy": round(val_acc, 4),
                    "val_macro_f1": round(val_f1, 4),
                }
                grid_records.append(rec)
                logger.info(
                    "  [%s] C=%8.3f | Time: %5.2fs | Iter: %3d (Conv: %s) | Val Acc: %6.2f%% | Val F1: %6.2f%%",
                    repr_type.upper(),
                    c,
                    dur,
                    n_iter,
                    converged,
                    val_acc * 100,
                    val_f1 * 100,
                )

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_c = c

            tuning_results[repr_type] = {
                "grid": grid_records,
                "best_C": float(best_c),
                "best_val_macro_f1": round(float(best_f1), 4),
            }
            logger.info("Selected optimal C* for %s: C = %.2f (Val Macro-F1 = %.2f%%)", repr_type.upper(), best_c, best_f1 * 100)

        # Save tuning log
        log_path = self.eval_dir / "linear_probe_tuning_log.json"
        with open(log_path, "w") as f:
            json.dump(tuning_results, f, indent=2)
        logger.info("Saved hyperparameter tuning log to %s", log_path.name)
        return tuning_results

    def evaluate_test_split(
        self,
        tuning_results: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Trains final linear probe on Train and evaluates Test split exactly once."""
        logger.info("=" * 70)
        logger.info("FINAL TEST EVALUATION (EXACTLY ONCE PER REPRESENTATION)")
        logger.info("=" * 70)

        test_results = {}

        for repr_type, path in [("cls", self.cls_path), ("mean_patch", self.patch_path)]:
            best_c = tuning_results[repr_type]["best_C"]
            logger.info("Training final %s probe on Train (N=%d) with C*=%.2f...", repr_type.upper(), len(self.train_idx), best_c)

            mm = np.memmap(path, dtype=np.float32, mode="r", shape=(self.total_samples, 384))
            X_tr = mm[self.train_idx]
            X_te = mm[self.test_idx]

            t0 = time.time()
            clf = LogisticRegression(
                C=best_c,
                max_iter=1000,
                solver="lbfgs",
                random_state=42,
            )
            clf.fit(X_tr, self.y_train)
            fit_time = time.time() - t0

            # Predict on Test exactly once
            t_inf_0 = time.time()
            test_probs = clf.predict_proba(X_te)
            test_preds = clf.predict(X_te)
            inf_time = time.time() - t_inf_0

            # Save model checkpoint
            model_ckpt_path = self.ckpt_dir / f"linear_probe_{repr_type}.joblib"
            joblib.dump(clf, model_ckpt_path)
            logger.info("Saved model weights to %s", model_ckpt_path.name)

            # Compute comprehensive metrics
            metrics = compute_comprehensive_metrics(
                y_true=self.y_test,
                y_prob=test_probs,
                class_names=CANONICAL_CLASSES,
                compute_ci=True,
            )

            # Save detailed test predictions CSV
            pred_records = []
            test_df_subset = self.df_manifest.iloc[self.test_idx].reset_index(drop=True)
            for i, row in test_df_subset.iterrows():
                rec = {
                    "row_index": int(i),
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

            pred_df = pd.DataFrame(pred_records)
            pred_csv_path = self.eval_dir / f"linear_probe_{repr_type}_test_predictions.csv"
            pred_df.to_csv(pred_csv_path, index=False)

            overall = metrics["overall"]
            result_summary = {
                "representation": repr_type,
                "model_name": "dinov2_vits14",
                "embedding_dim": 384,
                "optimal_C": float(best_c),
                "fit_time_seconds": round(fit_time, 3),
                "inference_time_seconds": round(inf_time, 3),
                "total_test_samples": len(self.test_idx),
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
                "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            }

            res_path = self.eval_dir / f"linear_probe_{repr_type}_results.json"
            with open(res_path, "w") as f:
                json.dump(result_summary, f, indent=2)
            logger.info("Saved final test results to %s", res_path.name)

            test_results[repr_type] = result_summary

        return test_results["cls"], test_results["mean_patch"]

    def generate_visualizations(
        self,
        cls_results: dict[str, Any],
        patch_results: dict[str, Any],
        tuning_results: dict[str, Any],
    ) -> None:
        """Generates publication-quality charts for downstream probing."""
        logger.info("Generating probe visualization figures in %s...", self.viz_dir)

        # 1. Hyperparameter Tuning Curves (C vs Validation Macro-F1)
        plt.figure(figsize=(8, 5))
        c_vals_cls = [r["C"] for r in tuning_results["cls"]["grid"]]
        f1_cls = [r["val_macro_f1"] * 100 for r in tuning_results["cls"]["grid"]]
        f1_patch = [r["val_macro_f1"] * 100 for r in tuning_results["mean_patch"]["grid"]]

        plt.plot(c_vals_cls, f1_cls, marker="o", linewidth=2, label="CLS Token Probe", color="#1d3557")
        plt.plot(c_vals_cls, f1_patch, marker="s", linewidth=2, linestyle="--", label="Mean Patch Token Probe", color="#e63946")
        plt.xscale("log")
        plt.title("GastricPath AI — Linear Probe Regularization Sweep (Validation Macro-F1)", fontsize=12, fontweight="bold")
        plt.xlabel("Regularization Parameter C (log scale)", fontsize=11)
        plt.ylabel("Validation Macro-F1 (%)", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(frameon=True)
        plt.tight_layout()
        fig_tuning_path = self.viz_dir / "linear_probe_tuning_curve.png"
        plt.savefig(fig_tuning_path, dpi=200)
        plt.close()
        logger.info("Saved: %s", fig_tuning_path.name)

        # 2. Confusion Matrix Heatmap (CLS Probe on Test Split)
        cm = np.array(cls_results["confusion_matrix"])
        cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100

        plt.figure(figsize=(8, 6.5))
        plt.imshow(cm_pct, interpolation="nearest", cmap="Blues")
        plt.title("GastricPath AI — CLS Linear Probe Confusion Matrix (% Test)", fontsize=12, fontweight="bold")
        plt.colorbar(label="Classification Accuracy (%)")
        ticks = np.arange(len(CANONICAL_CLASSES))
        plt.xticks(ticks, CANONICAL_CLASSES, rotation=45, fontsize=10)
        plt.yticks(ticks, CANONICAL_CLASSES, fontsize=10)

        for i in range(len(CANONICAL_CLASSES)):
            for j in range(len(CANONICAL_CLASSES)):
                val = cm_pct[i, j]
                color = "white" if val > 50 else "black"
                plt.text(j, i, f"{val:.1f}%", horizontalalignment="center", verticalalignment="center", color=color, fontsize=9)

        plt.ylabel("Ground Truth Class", fontsize=11)
        plt.xlabel("Predicted Class (Linear Probe)", fontsize=11)
        plt.tight_layout()
        fig_cm_path = self.viz_dir / "linear_probe_confusion_matrix.png"
        plt.savefig(fig_cm_path, dpi=200)
        plt.close()
        logger.info("Saved: %s", fig_cm_path.name)

        # 3. Per-Class F1 Comparison: Retrieval vs CLS Probe vs Mean Patch Probe
        f1_cls_classes = [cls_results["per_class_metrics"][c]["f1_score"] * 100 for c in CANONICAL_CLASSES]
        f1_patch_classes = [patch_results["per_class_metrics"][c]["f1_score"] * 100 for c in CANONICAL_CLASSES]

        # Load retrieval Hit@1 if available
        retrieval_file = self.eval_dir / "retrieval_evaluation_cls.json"
        if retrieval_file.exists():
            with open(retrieval_file, "r") as f:
                ret_data = json.load(f)
            ret_hit1 = [ret_data["metrics_by_k"]["k_1"]["per_class_consistency"][c] * 100 for c in CANONICAL_CLASSES]
        else:
            ret_hit1 = [0.0] * len(CANONICAL_CLASSES)

        x = np.arange(len(CANONICAL_CLASSES))
        width = 0.25

        plt.figure(figsize=(11, 5.5))
        plt.bar(x - width, ret_hit1, width, label="Frozen Retrieval Hit@1", color="#8d99ae")
        plt.bar(x, f1_cls_classes, width, label="CLS Linear Probe F1", color="#2a9d8f")
        plt.bar(x + width, f1_patch_classes, width, label="Mean Patch Probe F1", color="#e76f51")

        plt.title("GastricPath AI — Per-Class Discrimination: Retrieval vs. Linear Probes", fontsize=12, fontweight="bold")
        plt.xlabel("Histopathological Tissue Class", fontsize=11)
        plt.ylabel("Metric Score (%)", fontsize=11)
        plt.xticks(x, CANONICAL_CLASSES, fontsize=10, fontweight="bold")
        plt.ylim(0, 105)
        plt.legend(frameon=True)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig_f1_path = self.viz_dir / "linear_probe_per_class_f1.png"
        plt.savefig(fig_f1_path, dpi=200)
        plt.close()
        logger.info("Saved: %s", fig_f1_path.name)


def run_linear_probing(force_retune: bool = False):
    engine = LinearProbingEngine()
    tuning_log_path = engine.eval_dir / "linear_probe_tuning_log.json"
    if tuning_log_path.exists() and not force_retune:
        logger.info("Loading existing tuning log from %s...", tuning_log_path.name)
        with open(tuning_log_path, "r") as f:
            tuning_results = json.load(f)
    else:
        tuning_results = engine.tune_hyperparameters()

    cls_results, patch_results = engine.evaluate_test_split(tuning_results)
    engine.generate_visualizations(cls_results, patch_results, tuning_results)
    print("\n" + "=" * 80)
    print("PHASE 4.2 LINEAR PROBING & EVALUATION COMPLETE")
    print(f"  CLS Token Probe Test Accuracy:        {cls_results['overall_metrics']['accuracy']*100:.2f}% (Macro-F1: {cls_results['overall_metrics']['macro_f1']*100:.2f}%)")
    print(f"  Mean Patch Probe Test Accuracy:       {patch_results['overall_metrics']['accuracy']*100:.2f}% (Macro-F1: {patch_results['overall_metrics']['macro_f1']*100:.2f}%)")
    print("=" * 80)


if __name__ == "__main__":
    run_linear_probing()

