"""Phase 9: Research Error Analysis and Class-Level Findings Pipeline.

Executes consolidated error analysis on the winning Phikon Mean Patch classifier (N=467):
1. Verifies prediction data integrity and manifest alignment.
2. Computes raw and row-normalized confusion matrices.
3. Computes class-level metrics (Support, TP, FP, FN, TN, Precision, Recall/Sensitivity, Specificity, F1-score, Macro & Weighted averages).
4. Identifies and quantifies frequent off-diagonal confusion pairs.
5. Selects a deterministic 48-image gallery (3 correct + 3 incorrect per class) with cross-phase metadata (Phases 5-8).
6. Generates publication-ready visualizations:
   - confusion_matrix_counts.png
   - confusion_matrix_normalized.png
   - top_confusion_pairs.png
   - correct_prediction_gallery.png
   - incorrect_prediction_gallery.png
   - confidence_error_analysis.png
7. Exports error_analysis_manifest.csv, class_metrics.csv, confusion_pairs.csv, and error_analysis_summary.json.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase9_error_analysis")

CANONICAL_CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]


class ErrorAnalysisPipeline:
    """Consolidated error analysis and class-level findings for Phikon Mean Patch."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self.workspace_root = Path(workspace_root or WORKSPACE_ROOT).resolve()
        self.output_dir = self.workspace_root / "research" / "error_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_root = self.workspace_root / "data" / "raw" / "HMU-GC-HE-30K" / "all_image"
        self.pred_path = (
            self.workspace_root
            / "research"
            / "foundation_models"
            / "cache"
            / "subset10pct"
            / "evaluation"
            / "phikon_mean_patch_test_predictions.csv"
        )
        self.manifest_path = (
            self.workspace_root
            / "research"
            / "foundation_models"
            / "cache"
            / "subset10pct"
            / "manifest_subset.csv"
        )

        # Cross-phase artifact paths
        self.phase5_manifest_path = (
            self.workspace_root / "research" / "explainability" / "manifest_explainability_32.csv"
        )
        self.phase6_preds_path = (
            self.workspace_root / "research" / "robustness" / "predictions_robustness.csv"
        )
        self.phase7_ood_path = (
            self.workspace_root / "research" / "ood" / "ood_scores_all.csv"
        )
        self.phase8_spatial_path = (
            self.workspace_root / "research" / "spatial_analysis" / "spatial_descriptors.csv"
        )

    def verify_prediction_data(self) -> pd.DataFrame:
        """Verifies Phase 4.4 test prediction file against manifest."""
        logger.info("Loading and verifying Phase 4.4 test predictions...")
        if not self.pred_path.exists():
            raise FileNotFoundError(f"Missing test predictions: {self.pred_path}")
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing subset manifest: {self.manifest_path}")

        df_pred = pd.read_csv(self.pred_path)
        df_manifest = pd.read_csv(self.manifest_path)
        test_manifest = df_manifest[df_manifest["split"] == "test"].copy()

        assert len(df_pred) == 467, f"Expected 467 test predictions, got {len(df_pred)}"
        assert len(test_manifest) == 467, f"Expected 467 test manifest samples, got {len(test_manifest)}"

        # Check unique relative paths
        assert df_pred["relative_path"].nunique() == 467, "Duplicate paths found in test predictions"

        # Verify alignment with test manifest
        merged = pd.merge(df_pred, test_manifest, left_on="master_row_index", right_on="row_index")
        assert len(merged) == 467, "Mismatch between master_row_index in predictions and manifest"
        assert (merged["relative_path_x"] == merged["relative_path_y"]).all(), "Path mismatch in alignment"
        assert (merged["true_class"] == merged["class_name"]).all(), "Ground truth class mismatch"

        # Verify probabilities
        prob_cols = [f"prob_{c}" for c in CANONICAL_CLASSES]
        for col in prob_cols:
            assert col in df_pred.columns, f"Missing probability column: {col}"
        prob_sums = df_pred[prob_cols].sum(axis=1)
        assert np.allclose(prob_sums, 1.0, atol=1e-3), "Probability sums do not equal 1.0"

        # Verify confidence matches max probability
        max_probs = df_pred[prob_cols].max(axis=1)
        assert np.allclose(df_pred["confidence"], max_probs, atol=1e-3), "Confidence does not match max probability"

        # Add image_id
        df_pred["image_id"] = merged["image_id"]

        logger.info("Verified 467 clean test predictions with exact manifest alignment.")
        return df_pred

    def compute_confusion_and_metrics(
        self, df_pred: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
        """Computes raw/normalized confusion matrix, class-level metrics, and confusion pairs."""
        logger.info("Computing confusion matrices and class-level metrics...")
        cm = confusion_matrix(df_pred["true_class"], df_pred["predicted_class"], labels=CANONICAL_CLASSES)
        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

        # Class-level metrics table
        metric_rows = []
        for i, c in enumerate(CANONICAL_CLASSES):
            tp = int(cm[i, i])
            fn = int(cm[i, :].sum() - tp)
            fp = int(cm[:, i].sum() - tp)
            tn = int(cm.sum() - (tp + fp + fn))
            support = tp + fn
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            metric_rows.append({
                "class_name": c,
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "specificity": round(spec, 4),
            })

        df_metrics = pd.DataFrame(metric_rows)

        # Macro and weighted averages
        macro_prec = float(df_metrics["precision"].mean())
        macro_rec = float(df_metrics["recall"].mean())
        macro_f1 = float(df_metrics["f1_score"].mean())
        macro_spec = float(df_metrics["specificity"].mean())

        weights = df_metrics["support"] / df_metrics["support"].sum()
        wt_prec = float((df_metrics["precision"] * weights).sum())
        wt_rec = float((df_metrics["recall"] * weights).sum())
        wt_f1 = float((df_metrics["f1_score"] * weights).sum())
        wt_spec = float((df_metrics["specificity"] * weights).sum())

        summary_rows = [
            {
                "class_name": "Macro Average",
                "support": int(df_metrics["support"].sum()),
                "tp": int(df_metrics["tp"].sum()),
                "fp": int(df_metrics["fp"].sum()),
                "fn": int(df_metrics["fn"].sum()),
                "tn": int(df_metrics["tn"].sum()),
                "precision": round(macro_prec, 4),
                "recall": round(macro_rec, 4),
                "f1_score": round(macro_f1, 4),
                "specificity": round(macro_spec, 4),
            },
            {
                "class_name": "Weighted Average",
                "support": int(df_metrics["support"].sum()),
                "tp": int(df_metrics["tp"].sum()),
                "fp": int(df_metrics["fp"].sum()),
                "fn": int(df_metrics["fn"].sum()),
                "tn": int(df_metrics["tn"].sum()),
                "precision": round(wt_prec, 4),
                "recall": round(wt_rec, 4),
                "f1_score": round(wt_f1, 4),
                "specificity": round(wt_spec, 4),
            },
        ]
        df_metrics_full = pd.concat([df_metrics, pd.DataFrame(summary_rows)], ignore_index=True)

        # Save class metrics CSV
        metrics_csv_path = self.output_dir / "class_metrics.csv"
        df_metrics_full.to_csv(metrics_csv_path, index=False)
        logger.info("Saved class-level metrics CSV: %s", metrics_csv_path.name)

        # Frequent confusion pairs (off-diagonal)
        errors_df = df_pred[df_pred["true_class"] != df_pred["predicted_class"]]
        total_errors = len(errors_df)
        assert total_errors == 157, f"Expected 157 test errors, got {total_errors}"

        pair_counts = (
            errors_df.groupby(["true_class", "predicted_class"])
            .size()
            .reset_index(name="count")
            .sort_values(by="count", ascending=False)
            .reset_index(drop=True)
        )

        support_map = df_metrics.set_index("class_name")["support"].to_dict()
        pair_counts["pct_of_true_class"] = pair_counts.apply(
            lambda r: round(r["count"] / support_map[r["true_class"]] * 100, 2), axis=1
        )
        pair_counts["pct_of_all_errors"] = pair_counts.apply(
            lambda r: round(r["count"] / total_errors * 100, 2), axis=1
        )

        pairs_csv_path = self.output_dir / "confusion_pairs.csv"
        pair_counts.to_csv(pairs_csv_path, index=False)
        logger.info("Saved frequent confusion pairs CSV: %s (%d pairs)", pairs_csv_path.name, len(pair_counts))

        return cm, cm_norm, df_metrics_full, pair_counts

    def load_cross_phase_data(self) -> Dict[str, Any]:
        """Loads metadata from Phases 5, 6, 7, and 8 to enrich error analysis."""
        logger.info("Loading cross-phase datasets (Phases 5-8)...")
        cross_data: Dict[str, Any] = {}

        # Phase 5: 32 explainability samples
        if self.phase5_manifest_path.exists():
            df_p5 = pd.read_csv(self.phase5_manifest_path)
            cross_data["phase5_paths"] = set(df_p5["relative_path"])
        else:
            cross_data["phase5_paths"] = set()

        # Phase 6: 467 samples across 5 perturbation conditions
        if self.phase6_preds_path.exists():
            df_p6 = pd.read_csv(self.phase6_preds_path)
            piv = df_p6.pivot(index="relative_path", columns="condition", values="predicted_class")
            clean_preds = piv["clean"]
            flips = pd.DataFrame(index=piv.index)
            flips["flipped_under_bc"] = piv["brightness_contrast"] != clean_preds
            flips["flipped_under_blur"] = piv["gaussian_blur"] != clean_preds
            flips["flipped_under_jpeg"] = piv["jpeg_compression"] != clean_preds
            flips["flipped_under_hed"] = piv["stain_hed_variation"] != clean_preds
            flips["total_flips"] = (
                flips["flipped_under_bc"].astype(int)
                + flips["flipped_under_blur"].astype(int)
                + flips["flipped_under_jpeg"].astype(int)
                + flips["flipped_under_hed"].astype(int)
            )
            cross_data["phase6_flips"] = flips
        else:
            cross_data["phase6_flips"] = pd.DataFrame()

        # Phase 7: OOD scores and flags on clean test samples
        if self.phase7_ood_path.exists():
            df_p7 = pd.read_csv(self.phase7_ood_path)
            p7_clean = df_p7[df_p7["condition"] == "clean"].set_index("sample_id")
            cross_data["phase7_clean"] = p7_clean
        else:
            cross_data["phase7_clean"] = pd.DataFrame()

        # Phase 8: Spatial descriptors for 96 test samples
        if self.phase8_spatial_path.exists():
            df_p8 = pd.read_csv(self.phase8_spatial_path)
            p8_sub = df_p8.set_index("relative_path")
            cross_data["phase8_spatial"] = p8_sub
        else:
            cross_data["phase8_spatial"] = pd.DataFrame()

        return cross_data

    def select_deterministic_gallery(
        self, df_pred: pd.DataFrame, cross_data: Dict[str, Any]
    ) -> pd.DataFrame:
        """Selects up to 3 correct and 3 incorrect images per class using deterministic rules."""
        logger.info("Selecting deterministic 48-image error & success gallery...")
        p5_paths = cross_data["phase5_paths"]
        p6_flips = cross_data["phase6_flips"]
        p7_clean = cross_data["phase7_clean"]
        p8_spatial = cross_data["phase8_spatial"]

        errors_df = df_pred[df_pred["true_class"] != df_pred["predicted_class"]]
        top_conf_by_class = {
            c: errors_df[errors_df["true_class"] == c]["predicted_class"].value_counts().index[0]
            for c in CANONICAL_CLASSES
        }

        selected_records: List[Dict[str, Any]] = []

        for c in CANONICAL_CLASSES:
            # -------------------------------------------------------------
            # Selection for Correct Predictions (3 tiles)
            # -------------------------------------------------------------
            c_correct = df_pred[(df_pred["true_class"] == c) & (df_pred["is_correct"] == True)].sort_values(
                "confidence"
            )

            # 1. High-confidence representative correct
            r1 = c_correct.iloc[-1].to_dict()
            r1["gallery_type"] = "correct"
            r1["selection_reason"] = "Representative high-confidence correct prediction"

            # 2. Borderline low-confidence correct
            r2 = c_correct.iloc[0].to_dict()
            r2["gallery_type"] = "correct"
            r2["selection_reason"] = "Borderline low-confidence correct prediction"

            # 3. Cross-phase verified or median correct
            mid_candidates = c_correct.iloc[1:-1]
            p8_paths = set(p8_spatial.index) if not p8_spatial.empty else set()
            cp_cands = mid_candidates[mid_candidates["relative_path"].isin(p5_paths | p8_paths)]
            if len(cp_cands) > 0:
                r3 = cp_cands.iloc[len(cp_cands) // 2].to_dict()
                r3["selection_reason"] = "Cross-phase verified correct prediction"
            else:
                r3 = mid_candidates.iloc[len(mid_candidates) // 2].to_dict()
                r3["selection_reason"] = "Median-confidence typical correct prediction"
            r3["gallery_type"] = "correct"

            selected_records.extend([r1, r2, r3])

            # -------------------------------------------------------------
            # Selection for Incorrect Predictions (3 tiles)
            # -------------------------------------------------------------
            c_incorrect = df_pred[(df_pred["true_class"] == c) & (df_pred["is_correct"] == False)].sort_values(
                "confidence", ascending=False
            )

            # 1. High-confidence error (model overconfidence)
            e1 = c_incorrect.iloc[0].to_dict()
            e1["gallery_type"] = "incorrect"
            e1["selection_reason"] = "High-confidence misclassification (model overconfidence)"

            # 2. Frequent confusion error (True Class -> Top Confusion Target)
            target_class = top_conf_by_class[c]
            conf_cands = c_incorrect[c_incorrect["predicted_class"] == target_class]
            conf_cands = conf_cands[conf_cands["relative_path"] != e1["relative_path"]]
            if len(conf_cands) > 0:
                e2 = conf_cands.iloc[0].to_dict()
                e2["selection_reason"] = f"Frequent confusion error ({c} -> {target_class})"
            else:
                rem = c_incorrect[c_incorrect["relative_path"] != e1["relative_path"]]
                e2 = rem.iloc[0].to_dict()
                e2["selection_reason"] = f"Secondary misclassification ({c} -> {e2['predicted_class']})"
            e2["gallery_type"] = "incorrect"

            # 3. Cross-phase documented error or low-confidence ambiguous error
            chosen_paths = {e1["relative_path"], e2["relative_path"]}
            rem = c_incorrect[~c_incorrect["relative_path"].isin(chosen_paths)]
            cp_errs = rem[rem["relative_path"].isin(p5_paths | p8_paths)]
            if len(cp_errs) > 0:
                e3 = cp_errs.iloc[0].to_dict()
                e3["selection_reason"] = "Cross-phase documented misclassification"
            else:
                e3 = rem.iloc[-1].to_dict()
                e3["selection_reason"] = "Low-confidence ambiguous misclassification"
            e3["gallery_type"] = "incorrect"

            selected_records.extend([e1, e2, e3])

        df_gallery = pd.DataFrame(selected_records)
        assert len(df_gallery) == 48, f"Expected 48 gallery images, got {len(df_gallery)}"

        # Attach Cross-Phase Metadata
        df_gallery["in_phase5_explainability"] = df_gallery["relative_path"].isin(p5_paths)
        df_gallery["in_phase6_robustness"] = True  # All 467 are evaluated in Phase 6

        # Phase 6 flips
        if not p6_flips.empty:
            df_gallery["flips_bc"] = df_gallery["relative_path"].map(p6_flips["flipped_under_bc"]).fillna(False)
            df_gallery["flips_blur"] = df_gallery["relative_path"].map(p6_flips["flipped_under_blur"]).fillna(False)
            df_gallery["flips_jpeg"] = df_gallery["relative_path"].map(p6_flips["flipped_under_jpeg"]).fillna(False)
            df_gallery["flips_hed"] = df_gallery["relative_path"].map(p6_flips["flipped_under_hed"]).fillna(False)
            df_gallery["total_perturbation_flips"] = df_gallery["relative_path"].map(p6_flips["total_flips"]).fillna(0).astype(int)
        else:
            df_gallery["total_perturbation_flips"] = 0

        # Phase 7 OOD metadata
        if not p7_clean.empty:
            df_gallery["mahalanobis_score"] = (
                df_gallery["image_id"].map(p7_clean["mahalanobis_score"]).round(2)
            )
            df_gallery["mahal_flagged_ood"] = (
                df_gallery["image_id"].map(p7_clean["mahal_flagged_ood"]).fillna(False)
            )
            df_gallery["msp_ood_score"] = (
                df_gallery["image_id"].map(p7_clean["msp_ood_score"]).round(4)
            )
            df_gallery["msp_flagged_ood"] = (
                df_gallery["image_id"].map(p7_clean["msp_flagged_ood"]).fillna(False)
            )
        else:
            df_gallery["mahalanobis_score"] = np.nan
            df_gallery["mahal_flagged_ood"] = False
            df_gallery["msp_ood_score"] = np.nan
            df_gallery["msp_flagged_ood"] = False

        # Phase 8 Spatial Metadata
        p8_paths = set(p8_spatial.index) if not p8_spatial.empty else set()
        df_gallery["in_phase8_spatial"] = df_gallery["relative_path"].isin(p8_paths)
        if not p8_spatial.empty:
            df_gallery["spatial_regional_cos_sim"] = (
                df_gallery["relative_path"].map(p8_spatial["regional_cos_sim_mean"]).round(4)
            )
            df_gallery["spatial_regional_dispersion"] = (
                df_gallery["relative_path"].map(p8_spatial["regional_dispersion"]).round(4)
            )
            df_gallery["spatial_token_norm_dispersion"] = (
                df_gallery["relative_path"].map(p8_spatial["token_norm_dispersion"]).round(4)
            )
        else:
            df_gallery["spatial_regional_cos_sim"] = np.nan
            df_gallery["spatial_regional_dispersion"] = np.nan
            df_gallery["spatial_token_norm_dispersion"] = np.nan

        # Save Gallery Manifest
        manifest_path = self.output_dir / "error_analysis_manifest.csv"
        df_gallery.to_csv(manifest_path, index=False)
        logger.info("Saved error analysis gallery manifest: %s (48 rows)", manifest_path.name)
        return df_gallery

    def generate_visualizations(
        self,
        cm: np.ndarray,
        cm_norm: np.ndarray,
        df_pred: pd.DataFrame,
        pair_counts: pd.DataFrame,
        df_gallery: pd.DataFrame,
    ) -> None:
        """Renders publication-ready figures for Phase 9."""
        logger.info("Generating publication-ready figures for Phase 9...")

        # -------------------------------------------------------------
        # 1. Confusion Matrix (Raw Counts)
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_xticks(np.arange(len(CANONICAL_CLASSES)))
        ax.set_yticks(np.arange(len(CANONICAL_CLASSES)))
        ax.set_xticklabels(CANONICAL_CLASSES, fontsize=10, fontweight="bold")
        ax.set_yticklabels(CANONICAL_CLASSES, fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold", labelpad=8)
        ax.set_ylabel("True Ground-Truth Class", fontsize=11, fontweight="bold", labelpad=8)
        ax.set_title(
            "Phikon Mean Patch — Confusion Matrix (Raw Counts, N=467)",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )

        thresh = cm.max() / 2.0
        for i in range(len(CANONICAL_CLASSES)):
            for j in range(len(CANONICAL_CLASSES)):
                val = cm[i, j]
                color = "white" if val > thresh else "black"
                ax.text(
                    j, i, f"{val}", ha="center", va="center", color=color, fontsize=10, fontweight="semibold"
                )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        cm_counts_path = self.output_dir / "confusion_matrix_counts.png"
        plt.savefig(cm_counts_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info("Saved %s", cm_counts_path.name)

        # -------------------------------------------------------------
        # 2. Confusion Matrix (Row-Normalized Proportions)
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8.5, 7.5))
        im = ax.imshow(cm_norm, cmap="Blues", interpolation="nearest", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(CANONICAL_CLASSES)))
        ax.set_yticks(np.arange(len(CANONICAL_CLASSES)))
        ax.set_xticklabels(CANONICAL_CLASSES, fontsize=10, fontweight="bold")
        ax.set_yticklabels(CANONICAL_CLASSES, fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold", labelpad=8)
        ax.set_ylabel("True Ground-Truth Class", fontsize=11, fontweight="bold", labelpad=8)
        ax.set_title(
            "Phikon Mean Patch — Row-Normalized Confusion Matrix (Recall / Sensitivity)",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )

        thresh = 0.5
        for i in range(len(CANONICAL_CLASSES)):
            for j in range(len(CANONICAL_CLASSES)):
                val = cm_norm[i, j]
                color = "white" if val > thresh else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=9.5,
                    fontweight="semibold",
                )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Row Proportion (Sensitivity)", fontsize=10)
        plt.tight_layout()
        cm_norm_path = self.output_dir / "confusion_matrix_normalized.png"
        plt.savefig(cm_norm_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info("Saved %s", cm_norm_path.name)

        # -------------------------------------------------------------
        # 3. Top Confusion Pairs (Horizontal Bar Chart)
        # -------------------------------------------------------------
        top_pairs = pair_counts.head(10).iloc[::-1]  # Reverse for bottom-to-top plotting
        fig, ax = plt.subplots(figsize=(9, 6))
        pair_labels = [f"{r['true_class']} → {r['predicted_class']}" for _, r in top_pairs.iterrows()]
        bars = ax.barh(pair_labels, top_pairs["count"], color="#4a7bb0", edgecolor="navy", alpha=0.85)

        ax.set_xlabel("Number of Misclassified Test Tiles (Total Errors = 157)", fontsize=11, fontweight="bold")
        ax.set_title(
            "Top 10 Off-Diagonal Confusion Pairs (Phikon Mean Patch on N=467)",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        ax.grid(axis="x", linestyle=":", alpha=0.6)

        # Annotate percentages
        for bar, (_, r) in zip(bars, top_pairs.iterrows()):
            width = bar.get_width()
            ax.text(
                width + 0.15,
                bar.get_y() + bar.get_height() / 2,
                f"{int(width)} errors ({r['pct_of_true_class']:.1f}% of true, {r['pct_of_all_errors']:.1f}% of all)",
                va="center",
                fontsize=9,
                fontweight="semibold",
            )

        ax.set_xlim(0, max(top_pairs["count"]) + 3.8)
        plt.tight_layout()
        pairs_plot_path = self.output_dir / "top_confusion_pairs.png"
        plt.savefig(pairs_plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info("Saved %s", pairs_plot_path.name)

        # -------------------------------------------------------------
        # 4. Correct Prediction Gallery (8 classes x 3 columns)
        # -------------------------------------------------------------
        fig, axes = plt.subplots(nrows=8, ncols=3, figsize=(12, 24))
        plt.suptitle(
            "GastricPath AI — Phase 9: Correct Prediction Gallery (24 Representative Exemplars)",
            fontsize=14,
            fontweight="bold",
            y=0.995,
        )
        col_headers = [
            "High-Confidence Canonical",
            "Borderline Low-Confidence",
            "Cross-Phase / Median",
        ]
        for col, h in enumerate(col_headers):
            axes[0, col].set_title(h, fontsize=11, fontweight="bold", pad=8)

        for row_i, c in enumerate(CANONICAL_CLASSES):
            c_gallery = df_gallery[
                (df_gallery["true_class"] == c) & (df_gallery["gallery_type"] == "correct")
            ]
            for col_i in range(3):
                ax = axes[row_i, col_i]
                if col_i < len(c_gallery):
                    item = c_gallery.iloc[col_i]
                    img_file = self.dataset_root / item["relative_path"]
                    if img_file.exists():
                        img = Image.open(img_file).convert("RGB")
                        ax.imshow(img)
                    else:
                        ax.text(0.5, 0.5, "Image Missing", ha="center", va="center")

                    ax.set_xticks([])
                    ax.set_yticks([])
                    status_lbl = f"True: {c} | Conf: {item['confidence']:.2f}"
                    ax.set_xlabel(status_lbl, fontsize=8.5, color="#1b7837", fontweight="semibold")
                    if col_i == 0:
                        ax.set_ylabel(f"{c}", fontsize=11, fontweight="bold")
                else:
                    ax.axis("off")

        plt.tight_layout()
        corr_gallery_path = self.output_dir / "correct_prediction_gallery.png"
        plt.savefig(corr_gallery_path, dpi=180, bbox_inches="tight")
        plt.close()
        logger.info("Saved %s", corr_gallery_path.name)

        # -------------------------------------------------------------
        # 5. Incorrect Prediction Gallery (8 classes x 3 columns)
        # -------------------------------------------------------------
        fig, axes = plt.subplots(nrows=8, ncols=3, figsize=(12, 24))
        plt.suptitle(
            "GastricPath AI — Phase 9: Misclassification Gallery (24 Error Exemplars)",
            fontsize=14,
            fontweight="bold",
            y=0.995,
        )
        col_headers = [
            "High-Confidence Error",
            "Frequent Confusion Pair",
            "Cross-Phase / Ambiguous",
        ]
        for col, h in enumerate(col_headers):
            axes[0, col].set_title(h, fontsize=11, fontweight="bold", pad=8)

        for row_i, c in enumerate(CANONICAL_CLASSES):
            c_gallery = df_gallery[
                (df_gallery["true_class"] == c) & (df_gallery["gallery_type"] == "incorrect")
            ]
            for col_i in range(3):
                ax = axes[row_i, col_i]
                if col_i < len(c_gallery):
                    item = c_gallery.iloc[col_i]
                    img_file = self.dataset_root / item["relative_path"]
                    if img_file.exists():
                        img = Image.open(img_file).convert("RGB")
                        ax.imshow(img)
                    else:
                        ax.text(0.5, 0.5, "Image Missing", ha="center", va="center")

                    ax.set_xticks([])
                    ax.set_yticks([])
                    err_lbl = f"True: {c} → Pred: {item['predicted_class']}\nConf: {item['confidence']:.2f}"
                    ax.set_xlabel(err_lbl, fontsize=8.5, color="#d73027", fontweight="bold")
                    if col_i == 0:
                        ax.set_ylabel(f"{c}", fontsize=11, fontweight="bold")
                else:
                    ax.axis("off")

        plt.tight_layout()
        incorr_gallery_path = self.output_dir / "incorrect_prediction_gallery.png"
        plt.savefig(incorr_gallery_path, dpi=180, bbox_inches="tight")
        plt.close()
        logger.info("Saved %s", incorr_gallery_path.name)

        # -------------------------------------------------------------
        # 6. Confidence & Error Analysis (Dual Subplots)
        # -------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        # Panel A: Confidence distribution histogram
        ax1 = axes[0]
        correct_confs = df_pred[df_pred["is_correct"] == True]["confidence"]
        incorrect_confs = df_pred[df_pred["is_correct"] == False]["confidence"]

        bins = np.linspace(0.2, 1.0, 17)
        ax1.hist(
            correct_confs,
            bins=bins,
            alpha=0.65,
            color="#2ca02c",
            label=f"Correct Predictions (N={len(correct_confs)}, Mean={correct_confs.mean():.2f})",
            edgecolor="black",
        )
        ax1.hist(
            incorrect_confs,
            bins=bins,
            alpha=0.65,
            color="#d62728",
            label=f"Misclassifications (N={len(incorrect_confs)}, Mean={incorrect_confs.mean():.2f})",
            edgecolor="black",
        )
        ax1.axvline(correct_confs.mean(), color="#1b7837", linestyle="--", linewidth=1.8)
        ax1.axvline(incorrect_confs.mean(), color="#b2182b", linestyle="--", linewidth=1.8)
        ax1.set_xlabel("Prediction Confidence (Max Softmax Probability)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Number of Test Tiles", fontsize=11, fontweight="bold")
        ax1.set_title("Panel A: Confidence Distribution (Correct vs Errors)", fontsize=12, fontweight="bold")
        ax1.legend(loc="upper left", frameon=True)
        ax1.grid(axis="y", linestyle=":", alpha=0.6)

        # Panel B: Class Accuracy vs Mean Confidence
        ax2 = axes[1]
        class_accs = [
            df_pred[df_pred["true_class"] == c]["is_correct"].mean() for c in CANONICAL_CLASSES
        ]
        class_confs = [
            df_pred[df_pred["true_class"] == c]["confidence"].mean() for c in CANONICAL_CLASSES
        ]

        x = np.arange(len(CANONICAL_CLASSES))
        width = 0.35
        ax2.bar(x - width / 2, class_accs, width, label="Recall / Sensitivity", color="#4575b4", edgecolor="black")
        ax2.bar(x + width / 2, class_confs, width, label="Mean Confidence", color="#fdae61", edgecolor="black")

        ax2.set_xticks(x)
        ax2.set_xticklabels(CANONICAL_CLASSES, fontsize=10, fontweight="bold")
        ax2.set_ylabel("Proportion", fontsize=11, fontweight="bold")
        ax2.set_title("Panel B: Accuracy vs Mean Confidence per Class", fontsize=12, fontweight="bold")
        ax2.set_ylim(0, 1.05)
        ax2.legend(loc="lower right", frameon=True)
        ax2.grid(axis="y", linestyle=":", alpha=0.6)

        plt.suptitle(
            "GastricPath AI — Phase 9: Model Confidence and Error Characteristics",
            fontsize=13,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        conf_plot_path = self.output_dir / "confidence_error_analysis.png"
        plt.savefig(conf_plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info("Saved %s", conf_plot_path.name)

    def export_summary(
        self,
        df_pred: pd.DataFrame,
        df_metrics: pd.DataFrame,
        pair_counts: pd.DataFrame,
        df_gallery: pd.DataFrame,
        cross_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compiles and exports error_analysis_summary.json."""
        logger.info("Exporting error_analysis_summary.json...")
        total_samples = len(df_pred)
        correct_samples = int(df_pred["is_correct"].sum())
        error_samples = int((~df_pred["is_correct"]).sum())
        overall_acc = float(correct_samples / total_samples)

        correct_confs = df_pred[df_pred["is_correct"] == True]["confidence"]
        error_confs = df_pred[df_pred["is_correct"] == False]["confidence"]

        # Top confusion pairs list
        top_pairs_list = []
        for _, r in pair_counts.head(10).iterrows():
            top_pairs_list.append({
                "true_class": r["true_class"],
                "predicted_class": r["predicted_class"],
                "error_count": int(r["count"]),
                "pct_of_true_class": float(r["pct_of_true_class"]),
                "pct_of_all_errors": float(r["pct_of_all_errors"]),
            })

        # Cross-phase correlation metrics
        p6_flips = cross_data["phase6_flips"]
        p7_clean = cross_data["phase7_clean"]
        p8_spatial = cross_data["phase8_spatial"]

        cross_phase_summary = {
            "phase5_overlap_count": 32,
            "phase5_accuracy": float(df_pred[df_pred["relative_path"].isin(cross_data["phase5_paths"])]["is_correct"].mean())
            if cross_data["phase5_paths"] else None,
            "phase6_perturbation_flips": {
                "brightness_contrast_flips": int(p6_flips["flipped_under_bc"].sum()) if not p6_flips.empty else 0,
                "gaussian_blur_flips": int(p6_flips["flipped_under_blur"].sum()) if not p6_flips.empty else 0,
                "jpeg_compression_flips": int(p6_flips["flipped_under_jpeg"].sum()) if not p6_flips.empty else 0,
                "stain_hed_variation_flips": int(p6_flips["flipped_under_hed"].sum()) if not p6_flips.empty else 0,
            },
            "phase7_mahalanobis_scores": {
                "correct_mean": float(p7_clean.loc[df_pred[df_pred["is_correct"] == True]["relative_path"], "mahalanobis_score"].mean())
                if not p7_clean.empty else None,
                "error_mean": float(p7_clean.loc[df_pred[df_pred["is_correct"] == False]["relative_path"], "mahalanobis_score"].mean())
                if not p7_clean.empty else None,
            },
            "phase8_spatial_overlap_count": 96,
            "phase8_spatial_accuracy": float(p8_spatial["is_correct"].mean()) if not p8_spatial.empty else None,
        }

        summary_payload = {
            "metadata": {
                "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
                "model_name": "owkin/phikon (ViT-B/16, Mean Patch Pooling)",
                "linear_probe_C": 10.0,
                "evaluation_population": total_samples,
                "total_correct": correct_samples,
                "total_errors": error_samples,
                "overall_accuracy": round(overall_acc, 4),
            },
            "confidence_statistics": {
                "overall_mean_confidence": round(float(df_pred["confidence"].mean()), 4),
                "correct_mean_confidence": round(float(correct_confs.mean()), 4),
                "correct_std_confidence": round(float(correct_confs.std()), 4),
                "error_mean_confidence": round(float(error_confs.mean()), 4),
                "error_std_confidence": round(float(error_confs.std()), 4),
            },
            "class_metrics": df_metrics.to_dict(orient="records"),
            "top_confusion_pairs": top_pairs_list,
            "cross_phase_summary": cross_phase_summary,
            "gallery_manifest": {
                "total_selected": len(df_gallery),
                "correct_selected": int((df_gallery["gallery_type"] == "correct").sum()),
                "incorrect_selected": int((df_gallery["gallery_type"] == "incorrect").sum()),
                "samples_per_class": 6,
            },
        }

        summary_path = self.output_dir / "error_analysis_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary_payload, f, indent=2)
        logger.info("Saved %s", summary_path.name)
        return summary_payload

    def run_pipeline(self) -> Dict[str, Any]:
        """Executes the full Phase 9 Error Analysis pipeline."""
        df_pred = self.verify_prediction_data()
        cm, cm_norm, df_metrics, pair_counts = self.compute_confusion_and_metrics(df_pred)
        cross_data = self.load_cross_phase_data()
        df_gallery = self.select_deterministic_gallery(df_pred, cross_data)
        self.generate_visualizations(cm, cm_norm, df_pred, pair_counts, df_gallery)
        summary = self.export_summary(df_pred, df_metrics, pair_counts, df_gallery, cross_data)
        logger.info("Phase 9 Error Analysis Pipeline completed successfully.")
        return summary


if __name__ == "__main__":
    pipeline = ErrorAnalysisPipeline()
    pipeline.run_pipeline()

