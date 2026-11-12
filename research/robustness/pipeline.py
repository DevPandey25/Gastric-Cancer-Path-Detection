"""Phase 6: Controlled Robustness Analysis Pipeline.

Executes systematic stress-testing on the winning Phase 4.4 Phikon Mean Patch classifier
using 4 controlled perturbation families at a single predeclared severity on the entire
held-out test split (N=467 images across 8 canonical classes).
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch

from research.foundation_models.pathology_encoder.compare_representations import (
    mcnemar_paired_test,
    paired_bootstrap_test,
)
from research.foundation_models.pathology_encoder.subset_extractor import PhikonEncoderWrapper
from research.foundation_models.preprocessing import DeterministicPathologyPreprocessor
from research.robustness.perturbations import (
    apply_brightness_contrast,
    apply_gaussian_blur,
    apply_hed_stain_variation,
    apply_jpeg_compression,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase6_robustness")

CANONICAL_CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]
CONDITIONS = [
    "clean",
    "brightness_contrast",
    "gaussian_blur",
    "jpeg_compression",
    "stain_hed_variation",
]
CONDITION_LABELS = {
    "clean": "Clean (Original)",
    "brightness_contrast": "Brightness & Contrast (x1.1)",
    "gaussian_blur": "Gaussian Blur (5x5, s=1.0)",
    "jpeg_compression": "JPEG Quality 50",
    "stain_hed_variation": "H&E Stain Variation (H,E x1.1)",
}


class RobustnessPipeline:
    """Manages perturbation execution, inference, metric computation, and visualizations."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self.workspace_root = Path(workspace_root or WORKSPACE_ROOT).resolve()
        self.output_dir = self.workspace_root / "research" / "robustness"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.subset_dir = self.workspace_root / "research" / "foundation_models" / "cache" / "subset10pct"
        self.dataset_root = self.workspace_root / "data" / "raw" / "HMU-GC-HE-30K" / "all_image"
        self.manifest_path = self.subset_dir / "manifest_subset.csv"

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing subset manifest: {self.manifest_path}")

        self.df_manifest = pd.read_csv(self.manifest_path)
        self.test_mask = self.df_manifest["split"] == "test"
        self.df_test = self.df_manifest[self.test_mask].copy().reset_index()
        self.df_test.rename(columns={"index": "master_subset_index"}, inplace=True)
        self.n_test = len(self.df_test)
        assert self.n_test == 467, f"Expected 467 test images, got {self.n_test}"

        self.y_true = self.df_test["class_index"].to_numpy().astype(int)

        # Save test evaluation manifest
        self.test_manifest_path = self.output_dir / "robustness_subset_manifest.csv"
        self.df_test.to_csv(self.test_manifest_path, index=False)
        logger.info("Saved robustness evaluation manifest: %s (N=%d)", self.test_manifest_path.name, self.n_test)

        # Load linear probe
        probe_path = self.workspace_root / "research" / "explainability" / "fitted_phikon_linear_probe.joblib"
        if not probe_path.exists():
            probe_path = self._fit_and_save_probe()
        self.clf: LogisticRegression = joblib.load(probe_path)
        logger.info("Loaded linear probe from %s", probe_path.name)

        # Preprocessor and encoder
        self.preprocessor = DeterministicPathologyPreprocessor(image_size=224)
        self.model: PhikonEncoderWrapper | None = None  # Lazy loaded

    def _fit_and_save_probe(self) -> Path:
        probe_path = self.output_dir / "fitted_phikon_linear_probe.joblib"
        train_mask = self.df_manifest["split"] == "train"
        train_idx = np.where(train_mask)[0]
        y_train = self.df_manifest.iloc[train_idx]["class_index"].to_numpy()
        mm = np.memmap(
            self.subset_dir / "phikon" / "mean_patch_embeddings.npy",
            dtype=np.float32,
            mode="r",
            shape=(len(self.df_manifest), 768),
        )
        X_tr = mm[train_idx]
        logger.info("Fitting linear probe (C=10.0) on Phikon train embeddings (N=%d)...", len(train_idx))
        clf = LogisticRegression(C=10.0, max_iter=1000, solver="lbfgs", random_state=42)
        clf.fit(X_tr, y_train)
        joblib.dump(clf, probe_path)
        return probe_path

    def _load_model(self) -> PhikonEncoderWrapper:
        if self.model is None:
            logger.info("Instantiating frozen owkin/phikon encoder...")
            torch.set_num_threads(10)
            self.model = PhikonEncoderWrapper()
        return self.model

    def verify_clean_predictions(self) -> np.ndarray:
        """Verifies clean test embeddings reproduce Phase 4.4 predictions 100%."""
        mm = np.memmap(
            self.subset_dir / "phikon" / "mean_patch_embeddings.npy",
            dtype=np.float32,
            mode="r",
            shape=(len(self.df_manifest), 768),
        )
        test_indices = self.df_test["master_subset_index"].to_numpy()
        X_clean = mm[test_indices]
        clean_probs = self.clf.predict_proba(X_clean)
        clean_preds = self.clf.predict(X_clean)

        ref_path = self.subset_dir / "evaluation" / "phikon_mean_patch_test_predictions.csv"
        if ref_path.exists():
            df_ref = pd.read_csv(ref_path)
            ref_preds = df_ref["predicted_index"].to_numpy()
            mismatches = int(np.sum(clean_preds != ref_preds))
            if mismatches > 0:
                raise ValueError(f"Clean prediction mismatch! {mismatches} differences with Phase 4.4 reference.")
            logger.info("Clean predictions verified: 100%% exact match (%d / %d) with Phase 4.4 reference.", self.n_test, self.n_test)
        return X_clean

    def extract_perturbed_embeddings(
        self, condition: str, batch_size: int = 16
    ) -> np.ndarray:
        """Applies transformation and extracts L2-normalized mean patch embeddings for all 467 test images."""
        model = self._load_model()
        embeddings = np.zeros((self.n_test, 768), dtype=np.float32)

        logger.info("Starting feature extraction for condition '%s' (N=%d)...", condition, self.n_test)
        t0 = time.time()

        for b_start in range(0, self.n_test, batch_size):
            b_end = min(b_start + batch_size, self.n_test)
            batch_rows = self.df_test.iloc[b_start:b_end]

            tensors = []
            for _, row in batch_rows.iterrows():
                img_path = self.dataset_root / row["relative_path"]
                with Image.open(img_path) as raw_img:
                    pil_img = raw_img.convert("RGB")

                # Apply perturbation
                if condition == "brightness_contrast":
                    trans_img = apply_brightness_contrast(pil_img, brightness_factor=1.1, contrast_factor=1.1)
                elif condition == "gaussian_blur":
                    trans_img = apply_gaussian_blur(pil_img, ksize=(5, 5), sigma=1.0)
                elif condition == "jpeg_compression":
                    trans_img = apply_jpeg_compression(pil_img, quality=50)
                elif condition == "stain_hed_variation":
                    trans_img = apply_hed_stain_variation(pil_img, h_factor=1.1, e_factor=1.1)
                else:
                    raise ValueError(f"Unknown perturbation condition: {condition}")

                tensor = self.preprocessor.preprocess_image(trans_img)
                tensors.append(tensor)

            batch_tensor = torch.stack(tensors)
            with torch.no_grad():
                _, patch_norm = model(batch_tensor)

            embeddings[b_start:b_end] = patch_norm.cpu().numpy()

            if (b_end % 96 == 0) or (b_end == self.n_test):
                elapsed = time.time() - t0
                speed = b_end / max(elapsed, 1e-3)
                logger.info("  [%s] Processed %d / %d images (%.2f img/s, elapsed: %.1fs)", condition, b_end, self.n_test, speed, elapsed)

        return embeddings

    def run_robustness_evaluation(self) -> Dict[str, Any]:
        """Runs full robustness evaluation across all 5 conditions."""
        start_time = time.time()

        # Save config
        config = {
            "phase": "Phase 6: Controlled Robustness Analysis",
            "model_name": "owkin/phikon",
            "classifier": "LogisticRegression(C=10.0, solver='lbfgs', random_state=42)",
            "evaluation_set": "Held-out GCHTID test split from 10% stratified subset",
            "n_samples": self.n_test,
            "classes": CANONICAL_CLASSES,
            "conditions": {
                "clean": {"description": "Original uncorrupted H&E biopsy tiles"},
                "brightness_contrast": {"brightness_factor": 1.1, "contrast_factor": 1.1, "library": "PIL.ImageEnhance"},
                "gaussian_blur": {"ksize": [5, 5], "sigma": 1.0, "library": "cv2.GaussianBlur"},
                "jpeg_compression": {"quality": 50, "library": "Pillow in-memory JPEG encode/decode"},
                "stain_hed_variation": {
                    "h_factor": 1.1,
                    "e_factor": 1.1,
                    "dab_factor": 1.0,
                    "method": "Optical density HED decomposition using Ruifrok-Johnston matrix",
                },
            },
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # 1. Evaluate clean condition
        X_clean = self.verify_clean_predictions()
        clean_probs = self.clf.predict_proba(X_clean)
        clean_preds = self.clf.predict(X_clean)
        clean_conf = np.max(clean_probs, axis=1)

        condition_data = {
            "clean": {
                "embeddings": X_clean,
                "probs": clean_probs,
                "preds": clean_preds,
                "conf": clean_conf,
            }
        }

        # 2. Evaluate perturbed conditions
        for cond in CONDITIONS[1:]:
            emb = self.extract_perturbed_embeddings(cond, batch_size=16)
            probs = self.clf.predict_proba(emb)
            preds = self.clf.predict(emb)
            conf = np.max(probs, axis=1)
            condition_data[cond] = {
                "embeddings": emb,
                "probs": probs,
                "preds": preds,
                "conf": conf,
            }

        # 3. Build predictions DataFrame & JSON records
        all_pred_records = []
        for cond in CONDITIONS:
            preds = condition_data[cond]["preds"]
            probs = condition_data[cond]["probs"]
            conf = condition_data[cond]["conf"]

            for i, row in self.df_test.iterrows():
                pred_idx = int(preds[i])
                true_idx = int(row["class_index"])
                rec = {
                    "sample_id": row["relative_path"],
                    "relative_path": row["relative_path"],
                    "master_row_index": int(row["row_index"]),
                    "master_subset_index": int(row["master_subset_index"]),
                    "test_index": int(i),
                    "true_class": row["class_name"],
                    "true_index": true_idx,
                    "condition": cond,
                    "condition_label": CONDITION_LABELS[cond],
                    "predicted_class": CANONICAL_CLASSES[pred_idx],
                    "predicted_index": pred_idx,
                    "confidence": round(float(conf[i]), 4),
                    "is_correct": bool(pred_idx == true_idx),
                    "is_consistent_with_clean": bool(pred_idx == int(clean_preds[i])),
                    "is_label_flipped": bool(pred_idx != int(clean_preds[i])),
                }
                for c_idx, c_name in enumerate(CANONICAL_CLASSES):
                    rec[f"prob_{c_name}"] = round(float(probs[i, c_idx]), 4)
                all_pred_records.append(rec)

        df_preds = pd.DataFrame(all_pred_records)
        csv_path = self.output_dir / "predictions_robustness.csv"
        df_preds.to_csv(csv_path, index=False)
        logger.info("Saved robustness predictions CSV: %s (%d rows)", csv_path.name, len(df_preds))

        json_path = self.output_dir / "predictions_robustness.json"
        with open(json_path, "w") as f:
            json.dump(all_pred_records, f, indent=2)
        logger.info("Saved robustness predictions JSON: %s", json_path.name)

        # 4. Compute metrics for each condition
        metrics_summary = {}
        clean_acc = accuracy_score(self.y_true, clean_preds)
        clean_bacc = balanced_accuracy_score(self.y_true, clean_preds)
        clean_f1 = f1_score(self.y_true, clean_preds, average="macro", zero_division=0)
        clean_mean_conf = float(np.mean(clean_conf))

        for cond in CONDITIONS:
            preds = condition_data[cond]["preds"]
            probs = condition_data[cond]["probs"]
            conf = condition_data[cond]["conf"]

            acc = float(accuracy_score(self.y_true, preds))
            bacc = float(balanced_accuracy_score(self.y_true, preds))
            macro_f1 = float(f1_score(self.y_true, preds, average="macro", zero_division=0))
            weighted_f1 = float(f1_score(self.y_true, preds, average="weighted", zero_division=0))
            err_rate = float(1.0 - acc)

            # Consistency and label-flip relative to clean
            consistency = float(np.mean(preds == clean_preds))
            flip_rate = float(np.mean(preds != clean_preds))
            mean_conf = float(np.mean(conf))

            # Performance degradation (pp)
            delta_acc_pp = round((acc - clean_acc) * 100, 2)
            delta_bacc_pp = round((bacc - clean_bacc) * 100, 2)
            delta_f1_pp = round((macro_f1 - clean_f1) * 100, 2)
            delta_conf = round(mean_conf - clean_mean_conf, 4)

            # Per-class metrics
            prec_per_class = precision_score(self.y_true, preds, average=None, zero_division=0)
            rec_per_class = recall_score(self.y_true, preds, average=None, zero_division=0)
            f1_per_class = f1_score(self.y_true, preds, average=None, zero_division=0)

            clean_rec_per_class = recall_score(self.y_true, clean_preds, average=None, zero_division=0)

            per_class_dict = {}
            for c_idx, c_name in enumerate(CANONICAL_CLASSES):
                class_mask = self.y_true == c_idx
                class_n = int(np.sum(class_mask))
                flips_in_class = int(np.sum(preds[class_mask] != clean_preds[class_mask]))
                class_flip_rate = round(flips_in_class / class_n, 4)
                delta_rec_pp = round((float(rec_per_class[c_idx]) - float(clean_rec_per_class[c_idx])) * 100, 2)

                per_class_dict[c_name] = {
                    "precision": round(float(prec_per_class[c_idx]), 4),
                    "recall": round(float(rec_per_class[c_idx]), 4),
                    "f1_score": round(float(f1_per_class[c_idx]), 4),
                    "support": class_n,
                    "delta_recall_pp": delta_rec_pp,
                    "label_flips": flips_in_class,
                    "flip_rate": class_flip_rate,
                }

            # Statistical paired testing against clean (for perturbed conditions)
            stat_tests = None
            if cond != "clean":
                bootstrap_res = paired_bootstrap_test(
                    y_true=self.y_true,
                    preds_a=clean_preds,
                    preds_b=preds,
                    n_bootstraps=2000,
                    seed=42,
                )
                clean_correct = clean_preds == self.y_true
                cond_correct = preds == self.y_true
                mcnemar_res = mcnemar_paired_test(clean_correct, cond_correct)
                stat_tests = {
                    "mcnemar": mcnemar_res,
                    "paired_bootstrap": bootstrap_res,
                }

            metrics_summary[cond] = {
                "condition_label": CONDITION_LABELS[cond],
                "accuracy": round(acc, 4),
                "balanced_accuracy": round(bacc, 4),
                "macro_f1": round(macro_f1, 4),
                "weighted_f1": round(weighted_f1, 4),
                "ground_truth_error_rate": round(err_rate, 4),
                "prediction_consistency": round(consistency, 4),
                "label_flip_rate": round(flip_rate, 4),
                "mean_max_probability": round(mean_conf, 4),
                "degradation_pp": {
                    "accuracy": delta_acc_pp,
                    "balanced_accuracy": delta_bacc_pp,
                    "macro_f1": delta_f1_pp,
                },
                "delta_mean_max_prob": delta_conf,
                "per_class": per_class_dict,
                "statistical_tests": stat_tests,
            }

        total_runtime = time.time() - start_time
        summary_payload = {
            "metadata": {
                "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
                "n_samples": self.n_test,
                "total_runtime_sec": round(total_runtime, 2),
                "encoder": "owkin/phikon",
                "probe_c": 10.0,
            },
            "summary_metrics": metrics_summary,
        }

        metrics_json_path = self.output_dir / "metrics_robustness.json"
        with open(metrics_json_path, "w") as f:
            json.dump(summary_payload, f, indent=2)
        logger.info("Saved robustness metrics JSON: %s", metrics_json_path.name)

        # 5. Generate Visualizations
        self.generate_visualizations(condition_data, metrics_summary)

        logger.info("Phase 6 Robustness Evaluation successfully finished in %.1f seconds.", total_runtime)
        return summary_payload

    def generate_visualizations(
        self,
        condition_data: Dict[str, Any],
        metrics_summary: Dict[str, Any],
    ) -> None:
        """Generates contact sheet, overall comparison plot, and per-class recall plot."""
        logger.info("Generating publication-ready robustness visualizations...")

        # 1. Contact Sheet (8 classes x 5 conditions)
        # Select first sample of each class deterministically from test manifest
        exemplar_indices = []
        for c in CANONICAL_CLASSES:
            idx = self.df_test[self.df_test["class_name"] == c].index[0]
            exemplar_indices.append(idx)

        fig, axes = plt.subplots(nrows=8, ncols=5, figsize=(15, 22))
        col_titles = [
            "Original (Clean)",
            "Brightness/Contrast\n(x1.1 / x1.1)",
            "Gaussian Blur\n(5x5, s=1.0)",
            "JPEG Quality 50\n(Pillow)",
            "H&E Stain Var.\n(H,E x1.1 HED)",
        ]

        for col, title in enumerate(col_titles):
            axes[0, col].set_title(title, fontsize=12, fontweight="bold", pad=10)

        for row_i, idx in enumerate(exemplar_indices):
            row_data = self.df_test.iloc[idx]
            c_name = row_data["class_name"]
            img_path = self.dataset_root / row_data["relative_path"]

            with Image.open(img_path) as raw_img:
                clean_img = raw_img.convert("RGB")

            imgs_to_show = [
                clean_img,
                apply_brightness_contrast(clean_img, 1.1, 1.1),
                apply_gaussian_blur(clean_img, (5, 5), 1.0),
                apply_jpeg_compression(clean_img, 50),
                apply_hed_stain_variation(clean_img, 1.1, 1.1),
            ]

            for col_i, img in enumerate(imgs_to_show):
                ax = axes[row_i, col_i]
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])

                # Show true label and prediction
                cond_name = CONDITIONS[col_i]
                pred_idx = condition_data[cond_name]["preds"][idx]
                pred_name = CANONICAL_CLASSES[pred_idx]
                conf = condition_data[cond_name]["conf"][idx]
                is_corr = pred_idx == row_data["class_index"]

                status_color = "#2ca02c" if is_corr else "#d62728"
                ax.set_xlabel(f"Pred: {pred_name} ({conf:.2f})", fontsize=10, color=status_color, fontweight="semibold")

                if col_i == 0:
                    ax.set_ylabel(f"Class: {c_name}\n({row_data['relative_path'].split('/')[-1]})", fontsize=10, fontweight="bold")

        plt.suptitle("GastricPath AI — Phase 6: Controlled Robustness Perturbations Across 8 Tissue Classes", fontsize=15, fontweight="bold", y=0.995)
        plt.tight_layout()
        contact_path = self.output_dir / "contact_sheet_robustness.png"
        plt.savefig(contact_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info("Saved contact sheet: %s", contact_path.name)

        # 2. Overall Robustness Plot
        cond_names = CONDITIONS
        display_names = [CONDITION_LABELS[c] for c in cond_names]
        accs = [metrics_summary[c]["accuracy"] * 100 for c in cond_names]
        baccs = [metrics_summary[c]["balanced_accuracy"] * 100 for c in cond_names]
        f1s = [metrics_summary[c]["macro_f1"] * 100 for c in cond_names]

        x = np.arange(len(cond_names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6.5))
        rects1 = ax.bar(x - width, accs, width, label="Accuracy", color="#1f77b4")
        rects2 = ax.bar(x, baccs, width, label="Balanced Accuracy", color="#ff7f0e")
        rects3 = ax.bar(x + width, f1s, width, label="Macro-F1", color="#2ca02c")

        ax.set_ylabel("Metric Score (%)", fontsize=12, fontweight="bold")
        ax.set_title("GastricPath AI — Phase 6: Overall Performance Across Controlled Perturbations (N=467)", fontsize=13, fontweight="bold", pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, fontsize=10, fontweight="semibold", rotation=15, ha="right")
        ax.set_ylim(0, 85)
        ax.axhline(accs[0], color="#1f77b4", linestyle="--", alpha=0.5, label="Clean Accuracy Baseline")
        ax.axhline(f1s[0], color="#2ca02c", linestyle="--", alpha=0.5, label="Clean Macro-F1 Baseline")
        ax.legend(loc="lower left", frameon=True)
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        # Add values on top of bars
        for rects in [rects1, rects2, rects3]:
            for rect in rects:
                height = rect.get_height()
                ax.annotate(
                    f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="bold",
                )

        plt.tight_layout()
        overall_path = self.output_dir / "overall_robustness_plot.png"
        plt.savefig(overall_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info("Saved overall robustness plot: %s", overall_path.name)

        # 3. Per-Class Robustness Plot (Recall across conditions)
        fig, ax = plt.subplots(figsize=(14, 7))
        bar_w = 0.15
        x_cls = np.arange(len(CANONICAL_CLASSES))
        colors = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#1b9e77"]

        for i, cond in enumerate(CONDITIONS):
            recalls = [metrics_summary[cond]["per_class"][c]["recall"] * 100 for c in CANONICAL_CLASSES]
            offset = (i - 2) * bar_w
            ax.bar(x_cls + offset, recalls, bar_w, label=CONDITION_LABELS[cond], color=colors[i], alpha=0.9)

        ax.set_ylabel("Per-Class Recall / Sensitivity (%)", fontsize=12, fontweight="bold")
        ax.set_title("GastricPath AI — Phase 6: Per-Class Sensitivity Degradation Under Perturbations", fontsize=13, fontweight="bold", pad=15)
        ax.set_xticks(x_cls)
        ax.set_xticklabels(CANONICAL_CLASSES, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.legend(loc="upper right", frameon=True, fontsize=9.5)
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        plt.tight_layout()
        per_class_path = self.output_dir / "per_class_robustness.png"
        plt.savefig(per_class_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info("Saved per-class robustness plot: %s", per_class_path.name)


if __name__ == "__main__":
    pipeline = RobustnessPipeline()
    pipeline.run_robustness_evaluation()

