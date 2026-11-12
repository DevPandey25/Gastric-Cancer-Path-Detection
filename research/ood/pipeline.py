"""Phase 7: Lightweight OOD / Uncertainty Screening Pipeline.

Implements a Mahalanobis-distance-based feature-space novelty score and a Maximum
Softmax Probability (MSP) baseline for out-of-distribution proxy detection on the
frozen Phase 4.4 Phikon Mean Patch representation.

Methodology follows Lee et al. (NeurIPS 2018):
    "A Simple Unified Framework for Detecting Out-of-Distribution Samples
     and Adversarial Attacks."

Key design:
- Class-conditional Gaussian model fitted on TRAIN embeddings only.
- Shared (tied) covariance estimated via Ledoit-Wolf shrinkage for numerical stability.
- Rejection threshold calibrated from VALIDATION in-distribution scores only (95th percentile).
- Test evaluation and synthetic proxy-OOD assessment with frozen parameters.
- Phase 6 controlled perturbations used as synthetic OOD proxies (not genuine unseen tissue classes).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg as sp_linalg
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase7_ood")

CANONICAL_CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]
PERTURBATION_CONDITIONS = [
    "brightness_contrast",
    "gaussian_blur",
    "jpeg_compression",
    "stain_hed_variation",
]
PERTURBATION_LABELS = {
    "brightness_contrast": "Brightness & Contrast (x1.1)",
    "gaussian_blur": "Gaussian Blur (5x5, σ=1.0)",
    "jpeg_compression": "JPEG Quality 50",
    "stain_hed_variation": "H&E Stain Variation (H,E x1.1)",
    "combined_proxy": "Combined Proxy-OOD (All 4 Perturbations)",
}


class OODScreeningPipeline:
    """Mahalanobis distance and MSP-based OOD / uncertainty screening."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self.workspace_root = Path(workspace_root or WORKSPACE_ROOT).resolve()
        self.output_dir = self.workspace_root / "research" / "ood"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.subset_dir = self.workspace_root / "research" / "foundation_models" / "cache" / "subset10pct"
        self.manifest_path = self.subset_dir / "manifest_subset.csv"

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing subset manifest: {self.manifest_path}")

        self.df_manifest = pd.read_csv(self.manifest_path)
        self.n_total = len(self.df_manifest)

        # Split indices
        self.train_idx = np.where(self.df_manifest["split"] == "train")[0]
        self.val_idx = np.where(self.df_manifest["split"] == "val")[0]
        self.test_idx = np.where(self.df_manifest["split"] == "test")[0]

        self.y_train = self.df_manifest.iloc[self.train_idx]["class_index"].to_numpy().astype(int)
        self.y_val = self.df_manifest.iloc[self.val_idx]["class_index"].to_numpy().astype(int)
        self.y_test = self.df_manifest.iloc[self.test_idx]["class_index"].to_numpy().astype(int)

        logger.info(
            "Loaded manifest: Total=%d, Train=%d, Val=%d, Test=%d",
            self.n_total, len(self.train_idx), len(self.val_idx), len(self.test_idx),
        )

        # Load embeddings
        mm = np.memmap(
            self.subset_dir / "phikon" / "mean_patch_embeddings.npy",
            dtype=np.float32, mode="r", shape=(self.n_total, 768),
        )
        self.X_train = np.array(mm[self.train_idx], dtype=np.float64)
        self.X_val = np.array(mm[self.val_idx], dtype=np.float64)
        self.X_test = np.array(mm[self.test_idx], dtype=np.float64)

        # Load fitted linear probe
        probe_path = self.workspace_root / "research" / "explainability" / "fitted_phikon_linear_probe.joblib"
        self.clf: LogisticRegression = joblib.load(probe_path)
        logger.info("Loaded fitted linear probe from %s", probe_path.name)

        # Phase 6 robustness predictions (for proxy OOD)
        self.robustness_preds_path = self.workspace_root / "research" / "robustness" / "predictions_robustness.csv"

        # Fitted model parameters (populated by fit())
        self.class_means: Optional[np.ndarray] = None  # (8, 768)
        self.precision_matrix: Optional[np.ndarray] = None  # (768, 768) = Sigma^{-1}
        self.cholesky_L: Optional[np.ndarray] = None  # Lower Cholesky factor of covariance
        self.lw_shrinkage: Optional[float] = None
        self.cov_condition_number: Optional[float] = None
        self.mahal_threshold: Optional[float] = None
        self.msp_threshold: Optional[float] = None

    # ------------------------------------------------------------------ #
    #  Step 1: Fit class-conditional Gaussian model (TRAIN only)
    # ------------------------------------------------------------------ #
    def fit_mahalanobis_model(self) -> Dict[str, Any]:
        """Fits class means and shared Ledoit-Wolf covariance on train embeddings."""
        logger.info("Fitting class-conditional Gaussian model on train embeddings (N=%d)...", len(self.train_idx))

        # Class means
        self.class_means = np.zeros((8, 768), dtype=np.float64)
        within_class_residuals = []

        for c in range(8):
            mask = self.y_train == c
            X_c = self.X_train[mask]
            mu_c = X_c.mean(axis=0)
            self.class_means[c] = mu_c
            within_class_residuals.append(X_c - mu_c)
            logger.info(
                "  Class %d (%s): N=%d, ||mu||=%.4f",
                c, CANONICAL_CLASSES[c], mask.sum(), np.linalg.norm(mu_c),
            )

        residuals_all = np.vstack(within_class_residuals)
        logger.info("Total within-class residuals: %s", residuals_all.shape)

        # Ledoit-Wolf shrinkage estimator
        lw = LedoitWolf()
        lw.fit(residuals_all)
        covariance = lw.covariance_
        self.lw_shrinkage = float(lw.shrinkage_)
        logger.info("Ledoit-Wolf shrinkage coefficient: %.6f", self.lw_shrinkage)

        # Eigenvalue analysis for conditioning
        eigvals = np.linalg.eigvalsh(covariance)
        self.cov_condition_number = float(eigvals.max() / eigvals.min())
        all_positive = bool(np.all(eigvals > 0))
        logger.info(
            "Covariance conditioning: min_eigval=%.4e, max_eigval=%.4e, condition=%.2e, all_positive=%s",
            eigvals.min(), eigvals.max(), self.cov_condition_number, all_positive,
        )

        # Cholesky factorization for numerically stable solves
        # Sigma = L @ L^T, so Sigma^{-1} x = L^{-T} L^{-1} x
        self.cholesky_L = sp_linalg.cholesky(covariance, lower=True)
        # Also store precision matrix for reference
        self.precision_matrix = sp_linalg.cho_solve(
            (self.cholesky_L, True), np.eye(768, dtype=np.float64)
        )
        logger.info("Cholesky decomposition and precision matrix computed successfully.")

        # Save fitted parameters
        np.save(self.output_dir / "class_means.npy", self.class_means.astype(np.float32))
        np.save(self.output_dir / "precision_matrix.npy", self.precision_matrix.astype(np.float32))
        np.save(self.output_dir / "covariance_matrix.npy", covariance.astype(np.float32))
        np.save(self.output_dir / "cholesky_lower.npy", self.cholesky_L.astype(np.float32))

        fit_info = {
            "estimator": "Ledoit-Wolf (sklearn.covariance.LedoitWolf)",
            "shrinkage_coefficient": self.lw_shrinkage,
            "covariance_shape": list(covariance.shape),
            "min_eigenvalue": float(eigvals.min()),
            "max_eigenvalue": float(eigvals.max()),
            "condition_number": self.cov_condition_number,
            "all_eigenvalues_positive": all_positive,
            "n_train_samples": int(len(self.train_idx)),
            "n_classes": 8,
            "embedding_dim": 768,
        }

        with open(self.output_dir / "covariance_fit_info.json", "w") as f:
            json.dump(fit_info, f, indent=2)

        return fit_info

    # ------------------------------------------------------------------ #
    #  Mahalanobis score computation
    # ------------------------------------------------------------------ #
    def compute_mahalanobis_scores(self, X: np.ndarray) -> np.ndarray:
        """Computes min-class Mahalanobis distance for each sample.

        score(x) = min_c (x - mu_c)^T Sigma^{-1} (x - mu_c)

        Uses Cholesky solve for stability: Sigma^{-1} d = cho_solve(L, d).
        """
        n = X.shape[0]
        scores = np.full(n, np.inf, dtype=np.float64)

        for c in range(8):
            delta = X - self.class_means[c]  # (n, 768)
            # Solve L @ z = delta^T for z, then score = ||z||^2 per sample
            z = sp_linalg.solve_triangular(self.cholesky_L, delta.T, lower=True)  # (768, n)
            class_scores = np.sum(z ** 2, axis=0)  # (n,)
            scores = np.minimum(scores, class_scores)

        return scores

    # ------------------------------------------------------------------ #
    #  MSP score computation
    # ------------------------------------------------------------------ #
    def compute_msp_scores(self, X: np.ndarray) -> np.ndarray:
        """Computes OOD score as 1 - max_softmax_probability."""
        probs = self.clf.predict_proba(X.astype(np.float32))
        msp = np.max(probs, axis=1)
        return 1.0 - msp  # Higher = more OOD-like

    # ------------------------------------------------------------------ #
    #  Step 2: Calibrate thresholds (VALIDATION only)
    # ------------------------------------------------------------------ #
    def calibrate_thresholds(self, percentile: float = 95.0) -> Dict[str, Any]:
        """Calibrates rejection thresholds from validation in-distribution scores."""
        logger.info("Calibrating rejection thresholds on validation set (N=%d, percentile=%.1f)...",
                     len(self.val_idx), percentile)

        val_mahal_scores = self.compute_mahalanobis_scores(self.X_val)
        val_msp_scores = self.compute_msp_scores(self.X_val)

        self.mahal_threshold = float(np.percentile(val_mahal_scores, percentile))
        self.msp_threshold = float(np.percentile(val_msp_scores, percentile))

        val_mahal_reject_rate = float(np.mean(val_mahal_scores > self.mahal_threshold))
        val_msp_reject_rate = float(np.mean(val_msp_scores > self.msp_threshold))

        logger.info("  Mahalanobis threshold (%.0fth %%ile): %.4f (val rejection rate: %.4f)",
                     percentile, self.mahal_threshold, val_mahal_reject_rate)
        logger.info("  MSP threshold (%.0fth %%ile): %.4f (val rejection rate: %.4f)",
                     percentile, self.msp_threshold, val_msp_reject_rate)

        # Save scores
        np.save(self.output_dir / "val_mahalanobis_scores.npy", val_mahal_scores.astype(np.float32))
        np.save(self.output_dir / "val_msp_scores.npy", val_msp_scores.astype(np.float32))

        calibration_info = {
            "calibration_percentile": percentile,
            "calibration_split": "validation",
            "n_validation_samples": int(len(self.val_idx)),
            "mahalanobis": {
                "threshold": self.mahal_threshold,
                "val_mean": float(val_mahal_scores.mean()),
                "val_std": float(val_mahal_scores.std()),
                "val_min": float(val_mahal_scores.min()),
                "val_max": float(val_mahal_scores.max()),
                "val_median": float(np.median(val_mahal_scores)),
                "val_rejection_rate": val_mahal_reject_rate,
            },
            "msp": {
                "threshold": self.msp_threshold,
                "val_mean": float(val_msp_scores.mean()),
                "val_std": float(val_msp_scores.std()),
                "val_min": float(val_msp_scores.min()),
                "val_max": float(val_msp_scores.max()),
                "val_median": float(np.median(val_msp_scores)),
                "val_rejection_rate": val_msp_reject_rate,
            },
        }

        with open(self.output_dir / "calibration_info.json", "w") as f:
            json.dump(calibration_info, f, indent=2)

        return calibration_info

    # ------------------------------------------------------------------ #
    #  Step 3: Score test set and evaluate ID false-rejection
    # ------------------------------------------------------------------ #
    def score_test_set(self) -> Dict[str, Any]:
        """Scores clean test set with frozen thresholds, computes ID false-rejection rate."""
        logger.info("Scoring clean test set (N=%d) with frozen thresholds...", len(self.test_idx))

        test_mahal = self.compute_mahalanobis_scores(self.X_test)
        test_msp = self.compute_msp_scores(self.X_test)

        np.save(self.output_dir / "test_mahalanobis_scores.npy", test_mahal.astype(np.float32))
        np.save(self.output_dir / "test_msp_scores.npy", test_msp.astype(np.float32))

        # ID false-rejection rate: proportion of clean test samples that exceed threshold
        mahal_id_reject = float(np.mean(test_mahal > self.mahal_threshold))
        msp_id_reject = float(np.mean(test_msp > self.msp_threshold))

        logger.info("  Mahalanobis ID false-rejection rate: %.4f (%d / %d)",
                     mahal_id_reject, int(np.sum(test_mahal > self.mahal_threshold)), len(self.test_idx))
        logger.info("  MSP ID false-rejection rate: %.4f (%d / %d)",
                     msp_id_reject, int(np.sum(test_msp > self.msp_threshold)), len(self.test_idx))

        # Build per-sample CSV
        test_df = self.df_manifest.iloc[self.test_idx].reset_index()
        test_df.rename(columns={"index": "master_subset_index"}, inplace=True)

        # Get classifier predictions
        test_probs = self.clf.predict_proba(self.X_test.astype(np.float32))
        test_preds = self.clf.predict(self.X_test.astype(np.float32))
        test_conf = np.max(test_probs, axis=1)

        records = []
        for i in range(len(self.test_idx)):
            records.append({
                "sample_id": test_df.iloc[i]["relative_path"],
                "split": "test",
                "condition": "clean",
                "true_class": test_df.iloc[i]["class_name"],
                "true_index": int(test_df.iloc[i]["class_index"]),
                "predicted_class": CANONICAL_CLASSES[int(test_preds[i])],
                "predicted_index": int(test_preds[i]),
                "confidence": round(float(test_conf[i]), 4),
                "msp_ood_score": round(float(test_msp[i]), 4),
                "mahalanobis_score": round(float(test_mahal[i]), 4),
                "msp_flagged_ood": bool(test_msp[i] > self.msp_threshold),
                "mahal_flagged_ood": bool(test_mahal[i] > self.mahal_threshold),
            })

        test_scores_info = {
            "mahalanobis": {
                "test_mean": float(test_mahal.mean()),
                "test_std": float(test_mahal.std()),
                "test_min": float(test_mahal.min()),
                "test_max": float(test_mahal.max()),
                "test_median": float(np.median(test_mahal)),
                "id_false_rejection_rate": mahal_id_reject,
                "id_false_rejection_count": int(np.sum(test_mahal > self.mahal_threshold)),
            },
            "msp": {
                "test_mean": float(test_msp.mean()),
                "test_std": float(test_msp.std()),
                "test_min": float(test_msp.min()),
                "test_max": float(test_msp.max()),
                "test_median": float(np.median(test_msp)),
                "id_false_rejection_rate": msp_id_reject,
                "id_false_rejection_count": int(np.sum(test_msp > self.msp_threshold)),
            },
        }

        return test_scores_info, records

    # ------------------------------------------------------------------ #
    #  Step 4: Evaluate synthetic proxy-OOD from Phase 6 perturbations
    # ------------------------------------------------------------------ #
    def evaluate_proxy_ood(self, test_records: List[Dict]) -> Dict[str, Any]:
        """Evaluates proxy-OOD detection using Phase 6 perturbed embeddings."""
        logger.info("Loading Phase 6 perturbed embeddings and predictions for proxy-OOD evaluation...")

        # Load Phase 6 predictions for MSP scores
        rob_df = pd.read_csv(self.robustness_preds_path)
        prob_cols = [f"prob_{c}" for c in CANONICAL_CLASSES]

        # Clean test scores (already computed)
        test_mahal = np.load(self.output_dir / "test_mahalanobis_scores.npy").astype(np.float64)
        test_msp = np.load(self.output_dir / "test_msp_scores.npy").astype(np.float64)

        results = {}
        all_perturbed_records = []

        for cond in PERTURBATION_CONDITIONS:
            logger.info("Evaluating proxy-OOD for condition: %s", cond)
            cond_df = rob_df[rob_df["condition"] == cond].reset_index(drop=True)
            assert len(cond_df) == 467, f"Expected 467 for {cond}, got {len(cond_df)}"

            # MSP scores from saved probabilities
            cond_probs = cond_df[prob_cols].to_numpy(dtype=np.float64)
            cond_msp_scores = 1.0 - np.max(cond_probs, axis=1)

            # Mahalanobis scores from cached perturbed embeddings
            emb_file = self.output_dir / f"perturbed_embeddings_{cond}.npy"
            if not emb_file.exists():
                raise FileNotFoundError(f"Missing perturbed embeddings: {emb_file}")
            cond_emb = np.load(emb_file).astype(np.float64)
            cond_mahal_scores = self.compute_mahalanobis_scores(cond_emb)

            # Binary labels: 0 = ID (clean test), 1 = Proxy-OOD (perturbed)
            y_true_binary = np.concatenate([np.zeros(467), np.ones(467)])

            # MSP-based proxy-OOD detection
            msp_combined = np.concatenate([test_msp, cond_msp_scores])
            msp_auroc = float(roc_auc_score(y_true_binary, msp_combined))
            fpr_vals_msp, tpr_vals_msp, _ = roc_curve(y_true_binary, msp_combined)
            idx_95_msp = np.argmin(np.abs(tpr_vals_msp - 0.95))
            msp_fpr95 = float(fpr_vals_msp[idx_95_msp])
            msp_ood_detected = float(np.mean(cond_msp_scores > self.msp_threshold))

            # Mahalanobis-based proxy-OOD detection
            mahal_combined = np.concatenate([test_mahal, cond_mahal_scores])
            mahal_auroc = float(roc_auc_score(y_true_binary, mahal_combined))
            fpr_vals_m, tpr_vals_m, _ = roc_curve(y_true_binary, mahal_combined)
            idx_95_m = np.argmin(np.abs(tpr_vals_m - 0.95))
            mahal_fpr95 = float(fpr_vals_m[idx_95_m])
            mahal_ood_detected = float(np.mean(cond_mahal_scores > self.mahal_threshold))

            # Per-sample records for this condition
            for i in range(467):
                all_perturbed_records.append({
                    "sample_id": cond_df.iloc[i]["sample_id"] if "sample_id" in cond_df else cond_df.iloc[i]["relative_path"],
                    "split": "test",
                    "condition": cond,
                    "true_class": cond_df.iloc[i]["true_class"],
                    "true_index": int(cond_df.iloc[i]["true_index"]),
                    "predicted_class": cond_df.iloc[i]["predicted_class"],
                    "predicted_index": int(cond_df.iloc[i]["predicted_index"]),
                    "confidence": round(float(cond_df.iloc[i]["confidence"]), 4),
                    "msp_ood_score": round(float(cond_msp_scores[i]), 4),
                    "mahalanobis_score": round(float(cond_mahal_scores[i]), 4),
                    "msp_flagged_ood": bool(cond_msp_scores[i] > self.msp_threshold),
                    "mahal_flagged_ood": bool(cond_mahal_scores[i] > self.mahal_threshold),
                })

            results[cond] = {
                "condition_label": PERTURBATION_LABELS[cond],
                "n_samples": 467,
                "msp": {
                    "auroc": round(msp_auroc, 4),
                    "fpr_at_95tpr": round(msp_fpr95, 4),
                    "ood_detection_rate_at_threshold": round(msp_ood_detected, 4),
                    "ood_mean_score": round(float(cond_msp_scores.mean()), 4),
                    "ood_std_score": round(float(cond_msp_scores.std()), 4),
                },
                "mahalanobis": {
                    "auroc": round(mahal_auroc, 4),
                    "fpr_at_95tpr": round(mahal_fpr95, 4),
                    "ood_detection_rate_at_threshold": round(mahal_ood_detected, 4),
                    "ood_mean_score": round(float(cond_mahal_scores.mean()), 4),
                    "ood_std_score": round(float(cond_mahal_scores.std()), 4),
                },
            }
            logger.info(
                "  [%s] MSP AUROC=%.4f (FPR95=%.4f, Det=%.4f) | Mahal AUROC=%.4f (FPR95=%.4f, Det=%.4f)",
                cond, msp_auroc, msp_fpr95, msp_ood_detected, mahal_auroc, mahal_fpr95, mahal_ood_detected,
            )

        # Combined proxy-OOD (pool all 4 perturbations)
        all_perturbed_msp = []
        all_perturbed_mahal = []
        for cond in PERTURBATION_CONDITIONS:
            cond_df = rob_df[rob_df["condition"] == cond].reset_index(drop=True)
            cond_probs = cond_df[prob_cols].to_numpy(dtype=np.float64)
            all_perturbed_msp.append(1.0 - np.max(cond_probs, axis=1))

            emb_file = self.output_dir / f"perturbed_embeddings_{cond}.npy"
            cond_emb = np.load(emb_file).astype(np.float64)
            all_perturbed_mahal.append(self.compute_mahalanobis_scores(cond_emb))

        combined_ood_msp = np.concatenate(all_perturbed_msp)
        combined_ood_mahal = np.concatenate(all_perturbed_mahal)

        y_combined = np.concatenate([np.zeros(467), np.ones(467 * 4)])

        # MSP combined
        msp_combined_all = np.concatenate([test_msp, combined_ood_msp])
        combined_msp_auroc = float(roc_auc_score(y_combined, msp_combined_all))
        fpr_mc, tpr_mc, _ = roc_curve(y_combined, msp_combined_all)
        idx_95_mc = np.argmin(np.abs(tpr_mc - 0.95))
        combined_msp_fpr95 = float(fpr_mc[idx_95_mc])
        combined_msp_det = float(np.mean(combined_ood_msp > self.msp_threshold))

        # Mahalanobis combined
        mahal_combined_all = np.concatenate([test_mahal, combined_ood_mahal])
        combined_mahal_auroc = float(roc_auc_score(y_combined, mahal_combined_all))
        fpr_mac, tpr_mac, _ = roc_curve(y_combined, mahal_combined_all)
        idx_95_mac = np.argmin(np.abs(tpr_mac - 0.95))
        combined_mahal_fpr95 = float(fpr_mac[idx_95_mac])
        combined_mahal_det = float(np.mean(combined_ood_mahal > self.mahal_threshold))

        results["combined_proxy"] = {
            "condition_label": PERTURBATION_LABELS["combined_proxy"],
            "n_id_samples": 467,
            "n_ood_samples": 467 * 4,
            "pooling_note": "All 4 Phase 6 perturbation conditions pooled as a single proxy-OOD set.",
            "msp": {
                "auroc": round(combined_msp_auroc, 4),
                "fpr_at_95tpr": round(combined_msp_fpr95, 4),
                "ood_detection_rate_at_threshold": round(combined_msp_det, 4),
            },
            "mahalanobis": {
                "auroc": round(combined_mahal_auroc, 4),
                "fpr_at_95tpr": round(combined_mahal_fpr95, 4),
                "ood_detection_rate_at_threshold": round(combined_mahal_det, 4),
            },
        }

        return results, all_perturbed_records

    # ------------------------------------------------------------------ #
    #  Step 5: Visualizations
    # ------------------------------------------------------------------ #
    def generate_visualizations(
        self,
        test_scores_info: Dict,
        proxy_ood_results: Dict,
    ) -> None:
        """Generates OOD score distribution plots and ROC comparison."""
        logger.info("Generating OOD visualizations...")

        # Load scores
        test_mahal = np.load(self.output_dir / "test_mahalanobis_scores.npy").astype(np.float64)
        test_msp = np.load(self.output_dir / "test_msp_scores.npy").astype(np.float64)
        val_mahal = np.load(self.output_dir / "val_mahalanobis_scores.npy").astype(np.float64)
        val_msp = np.load(self.output_dir / "val_msp_scores.npy").astype(np.float64)

        # Load Phase 6 perturbed MSP and Mahalanobis scores
        rob_df = pd.read_csv(self.robustness_preds_path)
        prob_cols = [f"prob_{c}" for c in CANONICAL_CLASSES]
        perturbed_msp = {}
        perturbed_mahal = {}
        for cond in PERTURBATION_CONDITIONS:
            cond_df = rob_df[rob_df["condition"] == cond].reset_index(drop=True)
            cond_probs = cond_df[prob_cols].to_numpy(dtype=np.float64)
            perturbed_msp[cond] = 1.0 - np.max(cond_probs, axis=1)

            cond_emb = np.load(self.output_dir / f"perturbed_embeddings_{cond}.npy").astype(np.float64)
            perturbed_mahal[cond] = self.compute_mahalanobis_scores(cond_emb)

        # ---- Plot 1: Mahalanobis and MSP Score Distributions (Train, Val, Test) ----
        train_mahal = self.compute_mahalanobis_scores(self.X_train)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Mahalanobis
        ax = axes[0]
        ax.hist(train_mahal, bins=60, alpha=0.5, density=True, label="Train (ID, N=2,177)", color="#2b5c8f")
        ax.hist(val_mahal, bins=40, alpha=0.5, density=True, label="Validation (ID, N=466)", color="#ff7f0e")
        ax.hist(test_mahal, bins=40, alpha=0.5, density=True, label="Test (ID, N=467)", color="#2ca02c")
        ax.axvline(self.mahal_threshold, color="red", linestyle="--", linewidth=2,
                   label=f"95th %ile Threshold ({self.mahal_threshold:.2f})")
        ax.set_xlabel("Mahalanobis Distance Score", fontsize=12, fontweight="bold")
        ax.set_ylabel("Density", fontsize=12, fontweight="bold")
        ax.set_title("Mahalanobis Score Distribution (In-Distribution)", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.4)

        # MSP
        ax = axes[1]
        train_msp = 1.0 - self.clf.predict_proba(self.X_train.astype(np.float32)).max(axis=1)
        ax.hist(train_msp, bins=60, alpha=0.5, density=True, label="Train (ID, N=2,177)", color="#2b5c8f")
        ax.hist(val_msp, bins=40, alpha=0.5, density=True, label="Validation (ID, N=466)", color="#ff7f0e")
        ax.hist(test_msp, bins=40, alpha=0.5, density=True, label="Test (ID, N=467)", color="#2ca02c")
        ax.axvline(self.msp_threshold, color="red", linestyle="--", linewidth=2,
                   label=f"95th %ile Threshold ({self.msp_threshold:.2f})")
        ax.set_xlabel("MSP OOD Score (1 - max softmax prob)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Density", fontsize=12, fontweight="bold")
        ax.set_title("MSP Score Distribution (In-Distribution)", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.4)

        plt.suptitle("Phase 7: In-Distribution Calibration Distributions — Mahalanobis vs MSP",
                      fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / "id_score_distributions.png", dpi=200, bbox_inches="tight")
        plt.close()
        logger.info("Saved id_score_distributions.png")

        # ---- Plot 2: ID vs Proxy-OOD Score Distributions (Mahalanobis & MSP) ----
        fig, axes = plt.subplots(4, 2, figsize=(16, 16))
        colors = ["#d95f02", "#7570b3", "#e7298a", "#1b9e77"]

        for idx, cond in enumerate(PERTURBATION_CONDITIONS):
            # Left column: Mahalanobis
            ax_m = axes[idx, 0]
            ax_m.hist(test_mahal, bins=35, alpha=0.5, density=True, label="Clean Test (ID)", color="#2b5c8f")
            ax_m.hist(perturbed_mahal[cond], bins=35, alpha=0.5, density=True,
                      label=f"Proxy-OOD: {PERTURBATION_LABELS[cond]}", color=colors[idx])
            ax_m.axvline(self.mahal_threshold, color="red", linestyle="--", linewidth=1.5,
                         label=f"Threshold ({self.mahal_threshold:.1f})")
            ax_m.set_xlabel("Mahalanobis Score", fontsize=10)
            ax_m.set_ylabel("Density", fontsize=10)
            ax_m.set_title(f"Mahalanobis: {PERTURBATION_LABELS[cond]}", fontsize=11, fontweight="bold")
            ax_m.legend(fontsize=8)
            ax_m.grid(axis="y", alpha=0.4)

            # Right column: MSP
            ax_p = axes[idx, 1]
            ax_p.hist(test_msp, bins=35, alpha=0.5, density=True, label="Clean Test (ID)", color="#2b5c8f")
            ax_p.hist(perturbed_msp[cond], bins=35, alpha=0.5, density=True,
                      label=f"Proxy-OOD: {PERTURBATION_LABELS[cond]}", color=colors[idx])
            ax_p.axvline(self.msp_threshold, color="red", linestyle="--", linewidth=1.5,
                         label=f"Threshold ({self.msp_threshold:.3f})")
            ax_p.set_xlabel("MSP OOD Score", fontsize=10)
            ax_p.set_ylabel("Density", fontsize=10)
            ax_p.set_title(f"MSP: {PERTURBATION_LABELS[cond]}", fontsize=11, fontweight="bold")
            ax_p.legend(fontsize=8)
            ax_p.grid(axis="y", alpha=0.4)

        plt.suptitle("Phase 7: ID vs Proxy-OOD Score Distributions (Mahalanobis vs MSP)",
                      fontsize=14, fontweight="bold", y=1.002)
        plt.tight_layout()
        plt.savefig(self.output_dir / "proxy_ood_score_distributions.png", dpi=200, bbox_inches="tight")
        plt.close()
        logger.info("Saved proxy_ood_score_distributions.png")

        # ---- Plot 3: Dual-Panel ROC Curves (Mahalanobis vs MSP) ----
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Panel A: Mahalanobis ROC
        ax = axes[0]
        for idx, cond in enumerate(PERTURBATION_CONDITIONS):
            y_bin = np.concatenate([np.zeros(467), np.ones(467)])
            scores = np.concatenate([test_mahal, perturbed_mahal[cond]])
            fpr, tpr, _ = roc_curve(y_bin, scores)
            auroc = roc_auc_score(y_bin, scores)
            ax.plot(fpr, tpr, label=f"{PERTURBATION_LABELS[cond]} (AUROC={auroc:.3f})",
                    color=colors[idx], linewidth=2)

        # Combined Mahalanobis
        all_p_mahal = [perturbed_mahal[c] for c in PERTURBATION_CONDITIONS]
        y_comb = np.concatenate([np.zeros(467), np.ones(467 * 4)])
        scores_comb_m = np.concatenate([test_mahal] + all_p_mahal)
        fpr_cm, tpr_cm, _ = roc_curve(y_comb, scores_comb_m)
        auroc_cm = roc_auc_score(y_comb, scores_comb_m)
        ax.plot(fpr_cm, tpr_cm, label=f"Combined Proxy-OOD (AUROC={auroc_cm:.3f})",
                color="black", linewidth=2.5, linestyle="--")

        ax.plot([0, 1], [0, 1], "k:", alpha=0.4, label="Random Guess (0.500)")
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
        ax.set_title("Panel A: Mahalanobis Distance ROC Curves", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.4)

        # Panel B: MSP ROC
        ax = axes[1]
        for idx, cond in enumerate(PERTURBATION_CONDITIONS):
            y_bin = np.concatenate([np.zeros(467), np.ones(467)])
            scores = np.concatenate([test_msp, perturbed_msp[cond]])
            fpr, tpr, _ = roc_curve(y_bin, scores)
            auroc = roc_auc_score(y_bin, scores)
            ax.plot(fpr, tpr, label=f"{PERTURBATION_LABELS[cond]} (AUROC={auroc:.3f})",
                    color=colors[idx], linewidth=2)

        # Combined MSP
        all_p_msp = [perturbed_msp[c] for c in PERTURBATION_CONDITIONS]
        scores_comb_p = np.concatenate([test_msp] + all_p_msp)
        fpr_cp, tpr_cp, _ = roc_curve(y_comb, scores_comb_p)
        auroc_cp = roc_auc_score(y_comb, scores_comb_p)
        ax.plot(fpr_cp, tpr_cp, label=f"Combined Proxy-OOD (AUROC={auroc_cp:.3f})",
                color="black", linewidth=2.5, linestyle="--")

        ax.plot([0, 1], [0, 1], "k:", alpha=0.4, label="Random Guess (0.500)")
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
        ax.set_title("Panel B: Maximum Softmax Probability (MSP) ROC Curves", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.4)

        plt.suptitle("Phase 7: Proxy-OOD Detection ROC Curves — Mahalanobis Distance vs MSP Baseline",
                      fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / "roc_proxy_ood.png", dpi=200, bbox_inches="tight")
        plt.close()
        logger.info("Saved roc_proxy_ood.png")

    # ------------------------------------------------------------------ #
    #  Main execution
    # ------------------------------------------------------------------ #
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Executes the complete Phase 7 pipeline."""
        start_time = time.time()

        # Step 1: Fit Mahalanobis model
        fit_info = self.fit_mahalanobis_model()

        # Step 2: Calibrate thresholds
        cal_info = self.calibrate_thresholds(percentile=95.0)

        # Step 3: Score test set
        test_info, test_records = self.score_test_set()

        # Step 4: Proxy-OOD evaluation
        proxy_results, perturbed_records = self.evaluate_proxy_ood(test_records)

        # Step 5: Visualizations
        self.generate_visualizations(test_info, proxy_results)

        total_runtime = time.time() - start_time

        # Consolidate and save all results
        all_records = test_records + perturbed_records
        df_all = pd.DataFrame(all_records)
        df_all.to_csv(self.output_dir / "ood_scores_all.csv", index=False)
        with open(self.output_dir / "ood_scores_all.json", "w") as f:
            json.dump(all_records, f, indent=2)

        # Save config
        config = {
            "phase": "Phase 7: Lightweight OOD / Uncertainty Screening",
            "model": "owkin/phikon ViT-B/16 Mean Patch (768-d, L2-normalized)",
            "classifier": "LogisticRegression(C=10.0, solver='lbfgs', random_state=42)",
            "mahalanobis_method": "Class-conditional Gaussian with tied Ledoit-Wolf covariance",
            "msp_method": "Maximum Softmax Probability (1 - max_prob as OOD score)",
            "threshold_calibration": "95th percentile of validation in-distribution scores",
            "proxy_ood_source": "Phase 6 controlled perturbation families",
            "ood_positive_definition": "Higher score = more OOD-like",
            "score_direction": "Rejection if score > threshold",
            "data_isolation": {
                "train": "Class means and covariance estimation ONLY",
                "validation": "Threshold calibration ONLY (95th percentile)",
                "test": "Final evaluation with frozen parameters ONLY",
            },
            "executed_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_runtime_sec": round(total_runtime, 2),
        }
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        summary = {
            "metadata": config,
            "covariance_fit": fit_info,
            "calibration": cal_info,
            "test_scores": test_info,
            "proxy_ood_results": proxy_results,
        }

        with open(self.output_dir / "ood_evaluation_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("Phase 7 OOD Screening pipeline completed in %.1f seconds.", total_runtime)
        return summary


if __name__ == "__main__":
    pipeline = OODScreeningPipeline()
    pipeline.run_full_pipeline()

