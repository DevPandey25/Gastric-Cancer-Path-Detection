"""Downstream Linear Probing & Evaluation Engine for Phikon (Phase 4.4.3).

Evaluates the classification utility of frozen owkin/phikon ViT-B/16 representations:
A. CLS token embeddings (shape [31096, 768])
B. Mean spatial patch-token embeddings (shape [31096, 768])

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
logger = logging.getLogger("phikon_linear_probe")

CANONICAL_CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]
EMBEDDING_DIM = 768


class PhikonLinearProbingEngine:
    """Manages frozen Phikon linear probe training, tuning, and rigorous test evaluation."""

    def __init__(
        self,
        workspace_root: Path | str | None = None,
    ) -> None:
        if workspace_root is None:
            self.workspace_root = Path(__file__).resolve().parent.parent.parent.parent
        else:
            self.workspace_root = Path(workspace_root).resolve()

        self.cache_dir = self.workspace_root / "research" / "foundation_models" / "cache" / "full" / "phikon"
        self.encoder_dir = self.workspace_root / "research" / "foundation_models" / "pathology_encoder"
        self.eval_dir = self.encoder_dir / "evaluation"
        self.viz_dir = self.encoder_dir / "visualizations"
        self.ckpt_dir = self.encoder_dir / "checkpoints"

        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.cache_dir / "manifest.csv"
        self.cls_path = self.cache_dir / "cls_embeddings.npy"
        self.patch_path = self.cache_dir / "mean_patch_embeddings.npy"

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing Phikon manifest: {self.manifest_path}")

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
        logger.info("PHIKON HYPERPARAMETER TUNING ON VALIDATION SPLIT (TRAIN-ONLY FITTING)")
        logger.info("=" * 70)

        tuning_results: dict[str, Any] = {"candidate_c_values": candidate_c_values, "cls": {}, "mean_patch": {}}

        for repr_type, path in [("cls", self.cls_path), ("mean_patch", self.patch_path)]:
            logger.info("Tuning on %s embeddings...", repr_type.upper())
            mm = np.memmap(path, dtype=np.float32, mode="r", shape=(self.total_samples, EMBEDDING_DIM))
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
        log_path = self.eval_dir / "phikon_linear_probe_tuning_log.json"
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
        logger.info("PHIKON FINAL TEST EVALUATION (EXACTLY ONCE PER REPRESENTATION)")
        logger.info("=" * 70)

        test_results = {}

        for repr_type, path in [("cls", self.cls_path), ("mean_patch", self.patch_path)]:
            best_c = tuning_results[repr_type]["best_C"]
            logger.info("Training final Phikon %s probe on Train (N=%d) with C*=%.2f...", repr_type.upper(), len(self.train_idx), best_c)

            mm = np.memmap(path, dtype=np.float32, mode="r", shape=(self.total_samples, EMBEDDING_DIM))
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
            model_ckpt_path = self.ckpt_dir / f"phikon_linear_probe_{repr_type}.joblib"
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
            pred_csv_path = self.eval_dir / f"phikon_linear_probe_{repr_type}_test_predictions.csv"
            pred_df.to_csv(pred_csv_path, index=False)

            overall = metrics["overall"]
            result_summary = {
                "representation": repr_type,
                "model_name": "owkin/phikon",
                "architecture": "ViT-B/16 (iBOT TCGA Pan-Cancer)",
                "embedding_dim": EMBEDDING_DIM,
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

            res_path = self.eval_dir / f"phikon_linear_probe_{repr_type}_results.json"
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
        """Generates publication-quality charts for downstream Phikon probing."""
        logger.info("Generating probe visualization figures in %s...", self.viz_dir)

        # 1. Validation Tuning Curves
        fig, ax = plt.subplots(figsize=(8, 5))
        c_vals = [rec["C"] for rec in tuning_results["cls"]["grid"]]
        cls_f1s = [rec["val_macro_f1"] * 100 for rec in tuning_results["cls"]["grid"]]
        patch_f1s = [rec["val_macro_f1"] * 100 for rec in tuning_results["mean_patch"]["grid"]]

        ax.plot(c_vals, cls_f1s, marker="o", linewidth=2, label="Phikon CLS Token", color="#2b5c8f")
        ax.plot(c_vals, patch_f1s, marker="s", linewidth=2, label="Phikon Mean Patch Token", color="#d95f02")
        ax.set_xscale("log")
        ax.set_xlabel("Regularization Parameter C (log scale)", fontsize=11)
        ax.set_ylabel("Validation Macro-F1 (%)", fontsize=11)
        ax.set_title("Phikon Linear Probe Hyperparameter Sweep on Validation Split", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=10)
        plt.tight_layout()
        fig.savefig(self.viz_dir / "phikon_linear_probe_tuning_curve.png", dpi=300)
        plt.close(fig)

        # 2. Confusion Matrices
        for repr_type, res in [("cls", cls_results), ("mean_patch", patch_results)]:
            cm = np.array(res["confusion_matrix"])
            cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12) * 100

            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
            ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax.set(
                xticks=np.arange(len(CANONICAL_CLASSES)),
                yticks=np.arange(len(CANONICAL_CLASSES)),
                xticklabels=CANONICAL_CLASSES,
                yticklabels=CANONICAL_CLASSES,
                title=f"Phikon {repr_type.upper()} Probe Confusion Matrix (Normalized %)",
                ylabel="True Class",
                xlabel="Predicted Class",
            )
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            thresh = cm_norm.max() / 2.0
            for i in range(len(CANONICAL_CLASSES)):
                for j in range(len(CANONICAL_CLASSES)):
                    val = cm_norm[i, j]
                    raw = cm[i, j]
                    text_color = "white" if val > thresh else "black"
                    ax.text(j, i, f"{val:.1f}%\n({raw})", ha="center", va="center", color=text_color, fontsize=8)

            plt.tight_layout()
            fig.savefig(self.viz_dir / f"phikon_linear_probe_{repr_type}_confusion_matrix.png", dpi=300)
            plt.close(fig)

        logger.info("Visualizations generated successfully.")

    def run_pipeline(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Executes full hyperparameter tuning, test evaluation, and figure generation."""
        tuning = self.tune_hyperparameters()
        cls_res, patch_res = self.evaluate_test_split(tuning)
        self.generate_visualizations(cls_res, patch_res, tuning)
        return cls_res, patch_res


if __name__ == "__main__":
    engine = PhikonLinearProbingEngine()
    engine.run_pipeline()

