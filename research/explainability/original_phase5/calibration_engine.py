"""Original Phase 5: Calibration, Uncertainty, and Abstention Engine.

Implements rigorous post-hoc probability calibration and selective classification:
1. Computes logits for both the Phikon Mean Patch linear probe and GastricCancerRNN baseline.
2. Fits scalar temperature T* strictly on the validation split (N=466) minimizing NLL.
   Test-set data is strictly excluded from temperature fitting.
3. Evaluates ECE (Expected Calibration Error, 10 bins), Brier score, and NLL before & after calibration.
4. Generates publication-ready reliability diagrams:
   - reliability_diagrams_phikon.png
   - reliability_diagrams_cnn.png
5. Implements validation-calibrated selective classification (abstention):
   - Computes Coverage vs Selective Accuracy trade-offs.
   - Calibrates operating thresholds on validation data targeting 90% coverage and 80% accuracy.
   - Evaluates on held-out test data (N=467).
   - Generates abstention_tradeoff_curves.png.
6. Exports calibration_metrics.json and abstention_results.csv.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from PIL import Image
import torch
import torch.nn.functional as F

from app.core.config import settings
from app.ml.model import GastricCancerRNN
from app.ml.preprocessing import image_to_patch_sequence, load_rgb_image

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("original_phase5_calibration")

CANONICAL_CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable softmax computation."""
    shifted = z - np.max(z, axis=axis, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> Tuple[float, Dict[str, Any]]:
    """Computes Expected Calibration Error and bin-level statistics."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_stats: List[Dict[str, Any]] = []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = float(np.mean(in_bin))
        if prop_in_bin > 0:
            acc_in_bin = float(np.mean(accuracies[in_bin]))
            conf_in_bin = float(np.mean(confidences[in_bin]))
            ece += abs(conf_in_bin - acc_in_bin) * prop_in_bin
            bin_stats.append({
                "bin_idx": i,
                "range": [round(bin_boundaries[i], 2), round(bin_boundaries[i + 1], 2)],
                "count": int(np.sum(in_bin)),
                "proportion": round(prop_in_bin, 4),
                "accuracy": round(acc_in_bin, 4),
                "confidence": round(conf_in_bin, 4),
                "calibration_gap": round(conf_in_bin - acc_in_bin, 4),
            })
        else:
            bin_stats.append({
                "bin_idx": i,
                "range": [round(bin_boundaries[i], 2), round(bin_boundaries[i + 1], 2)],
                "count": 0,
                "proportion": 0.0,
                "accuracy": 0.0,
                "confidence": 0.0,
                "calibration_gap": 0.0,
            })

    return float(ece), {"bins": bin_stats, "overall_ece": round(float(ece), 4)}


def compute_brier_score(probs: np.ndarray, labels: np.ndarray, n_classes: int = 8) -> float:
    """Computes multi-class Brier score (lower is better)."""
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def compute_nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """Computes negative log-likelihood."""
    eps = 1e-12
    nll_vals = -np.log(probs[np.arange(len(labels)), labels] + eps)
    return float(np.mean(nll_vals))


class CalibrationEngine:
    """Calibration, uncertainty, and abstention engine for GastricPath AI."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self.workspace_root = Path(workspace_root or WORKSPACE_ROOT).resolve()
        self.output_dir = self.workspace_root / "research" / "explainability" / "original_phase5"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_path = (
            self.workspace_root
            / "research"
            / "foundation_models"
            / "cache"
            / "subset10pct"
            / "manifest_subset.csv"
        )
        self.df_manifest = pd.read_csv(self.manifest_path)

        # Phikon Mean Patch Probe
        emb_path = (
            self.workspace_root
            / "research"
            / "foundation_models"
            / "cache"
            / "subset10pct"
            / "phikon"
            / "mean_patch_embeddings.npy"
        )
        self.phikon_embs = np.fromfile(emb_path, dtype=np.float32).reshape(3110, 768)
        probe_path = self.workspace_root / "research" / "explainability" / "fitted_phikon_linear_probe.joblib"
        self.phikon_clf = joblib.load(probe_path)

        # GastricCancerRNN
        self.device = torch.device("cpu")
        self.cnn_model = GastricCancerRNN(num_classes=8).to(self.device)
        ckpt_path = self.workspace_root / "models" / "checkpoints" / "gastric_rnn_classifier.pt"
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.cnn_model.load_state_dict(ckpt["model_state_dict"])
        self.cnn_model.eval()

        # Split masks and labels
        self.val_mask = (self.df_manifest["split"] == "val").values
        self.test_mask = (self.df_manifest["split"] == "test").values
        self.y_val = self.df_manifest.loc[self.val_mask, "class_index"].values
        self.y_test = self.df_manifest.loc[self.test_mask, "class_index"].values

        assert len(self.y_val) == 466, f"Expected 466 validation samples, got {len(self.y_val)}"
        assert len(self.y_test) == 467, f"Expected 467 test samples, got {len(self.y_test)}"

    def get_phikon_logits(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts validation and test logits from the Phikon Mean Patch probe."""
        val_logits = self.phikon_clf.decision_function(self.phikon_embs[self.val_mask])
        test_logits = self.phikon_clf.decision_function(self.phikon_embs[self.test_mask])
        return val_logits, test_logits

    def extract_cnn_logits(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts validation and test logits from the GastricCancerRNN model."""
        logger.info("Extracting GastricCancerRNN logits for val (N=466) and test (N=467)...")
        dataset_root = self.workspace_root / "data" / "raw" / "HMU-GC-HE-30K" / "all_image"

        def _infer_split(mask: np.ndarray) -> np.ndarray:
            sub_df = self.df_manifest.loc[mask].reset_index(drop=True)
            sequences: List[torch.Tensor] = []
            logits_list: List[np.ndarray] = []
            batch_size = 16

            for idx, row in sub_df.iterrows():
                img_path = dataset_root / row["relative_path"]
                img = load_rgb_image(img_path, settings.image_size)
                seq = image_to_patch_sequence(img, settings.patch_size, settings.sequence_length)
                sequences.append(seq)

                if len(sequences) == batch_size or (idx + 1) == len(sub_df):
                    batch = torch.stack(sequences, dim=0).to(self.device)
                    with torch.no_grad():
                        logits, _ = self.cnn_model(batch)
                        logits_list.append(logits.numpy())
                    sequences = []

            return np.concatenate(logits_list, axis=0)

        val_logits = _infer_split(self.val_mask)
        test_logits = _infer_split(self.test_mask)
        return val_logits, test_logits

    def fit_temperature_scaling(self, val_logits: np.ndarray, y_val: np.ndarray) -> float:
        """Fits scalar temperature T* strictly on validation data using L-BFGS-B."""
        val_logits_t = torch.tensor(val_logits, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long)

        def eval_nll(t_val):
            t_scalar = float(t_val[0])
            scaled = val_logits_t / t_scalar
            return F.cross_entropy(scaled, y_val_t).item()

        res = minimize(eval_nll, [1.0], bounds=[(0.05, 10.0)], method="L-BFGS-B")
        t_opt = float(res.x[0])
        logger.info("Optimized temperature T*: %.4f (Val NLL: %.4f -> %.4f)", t_opt, eval_nll([1.0]), eval_nll([t_opt]))
        return t_opt

    def evaluate_model_calibration(
        self,
        name: str,
        val_logits: np.ndarray,
        test_logits: np.ndarray,
    ) -> Dict[str, Any]:
        """Performs complete calibration audit before & after validation-fitted temperature scaling."""
        logger.info("Evaluating calibration for %s...", name)
        t_opt = self.fit_temperature_scaling(val_logits, self.y_val)

        # Validation metrics
        val_probs_uncal = softmax(val_logits)
        val_probs_cal = softmax(val_logits / t_opt)

        val_ece_uncal, val_bins_uncal = compute_ece(val_probs_uncal, self.y_val)
        val_ece_cal, val_bins_cal = compute_ece(val_probs_cal, self.y_val)
        val_brier_uncal = compute_brier_score(val_probs_uncal, self.y_val)
        val_brier_cal = compute_brier_score(val_probs_cal, self.y_val)
        val_nll_uncal = compute_nll(val_probs_uncal, self.y_val)
        val_nll_cal = compute_nll(val_probs_cal, self.y_val)

        # Test metrics (Strictly evaluated using validation-learned T_opt)
        test_probs_uncal = softmax(test_logits)
        test_probs_cal = softmax(test_logits / t_opt)

        test_ece_uncal, test_bins_uncal = compute_ece(test_probs_uncal, self.y_test)
        test_ece_cal, test_bins_cal = compute_ece(test_probs_cal, self.y_test)
        test_brier_uncal = compute_brier_score(test_probs_uncal, self.y_test)
        test_brier_cal = compute_brier_score(test_probs_cal, self.y_test)
        test_nll_uncal = compute_nll(test_probs_uncal, self.y_test)
        test_nll_cal = compute_nll(test_probs_cal, self.y_test)

        test_acc = float((test_probs_uncal.argmax(axis=1) == self.y_test).mean())

        results = {
            "model_name": name,
            "optimal_temperature_T": round(t_opt, 4),
            "test_accuracy": round(test_acc, 4),
            "validation_metrics": {
                "uncalibrated": {
                    "ece": round(val_ece_uncal, 4),
                    "brier_score": round(val_brier_uncal, 4),
                    "nll": round(val_nll_uncal, 4),
                },
                "calibrated": {
                    "ece": round(val_ece_cal, 4),
                    "brier_score": round(val_brier_cal, 4),
                    "nll": round(val_nll_cal, 4),
                },
            },
            "test_metrics": {
                "uncalibrated": {
                    "ece": round(test_ece_uncal, 4),
                    "brier_score": round(test_brier_uncal, 4),
                    "nll": round(test_nll_uncal, 4),
                },
                "calibrated": {
                    "ece": round(test_ece_cal, 4),
                    "brier_score": round(test_brier_cal, 4),
                    "nll": round(test_nll_cal, 4),
                },
            },
            "test_bin_stats_uncalibrated": test_bins_uncal["bins"],
            "test_bin_stats_calibrated": test_bins_cal["bins"],
        }
        return results

    def plot_reliability_diagrams(self, cal_result: Dict[str, Any], filename: str) -> None:
        """Plots dual-panel reliability diagrams before and after temperature scaling."""
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        m_name = cal_result["model_name"]
        t_opt = cal_result["optimal_temperature_T"]

        for idx, (mode, bins_data, ece, brier) in enumerate([
            ("Uncalibrated (T=1.000)", cal_result["test_bin_stats_uncalibrated"], cal_result["test_metrics"]["uncalibrated"]["ece"], cal_result["test_metrics"]["uncalibrated"]["brier_score"]),
            (f"Temperature Scaled (T={t_opt:.3f})", cal_result["test_bin_stats_calibrated"], cal_result["test_metrics"]["calibrated"]["ece"], cal_result["test_metrics"]["calibrated"]["brier_score"]),
        ]):
            ax = axes[idx]
            confs = [b["confidence"] for b in bins_data if b["count"] > 0]
            accs = [b["accuracy"] for b in bins_data if b["count"] > 0]
            counts = [b["count"] for b in bins_data if b["count"] > 0]

            # Bar plot of accuracy vs confidence
            widths = [0.08] * len(confs)
            ax.bar(confs, accs, width=widths, color="#4575b4", alpha=0.75, edgecolor="black", label="Outputs")

            # Diagonal perfect calibration
            ax.plot([0, 1], [0, 1], "r--", linewidth=1.8, label="Perfect Calibration")

            # Annotate bin counts
            for c, a, n in zip(confs, accs, counts):
                ax.text(c, a + 0.02, f"n={n}", ha="center", va="bottom", fontsize=7.5, rotation=45)

            ax.set_xlim(0, 1.0)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Confidence", fontsize=11, fontweight="bold")
            ax.set_ylabel("Empirical Accuracy", fontsize=11, fontweight="bold")
            ax.set_title(f"{mode}\nECE = {ece:.4f} | Brier = {brier:.4f}", fontsize=11, fontweight="bold")
            ax.legend(loc="upper left", frameon=True)
            ax.grid(True, linestyle=":", alpha=0.6)

        plt.suptitle(f"GastricPath AI — Original Phase 5: Reliability Diagrams ({m_name} on Test Split N=467)", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info("Saved %s", filename)

    def evaluate_selective_classification(
        self,
        val_probs: np.ndarray,
        test_probs: np.ndarray,
        thresholds: np.ndarray | None = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Evaluates validation-calibrated selective classification (abstention)."""
        logger.info("Evaluating validation-calibrated selective classification / abstention...")
        if thresholds is None:
            thresholds = np.linspace(0.20, 0.95, 31)

        val_confs = np.max(val_probs, axis=1)
        val_preds = np.argmax(val_probs, axis=1)
        val_correct = (val_preds == self.y_val)

        test_confs = np.max(test_probs, axis=1)
        test_preds = np.argmax(test_probs, axis=1)
        test_correct = (test_preds == self.y_test)

        records: List[Dict[str, Any]] = []
        for t in thresholds:
            # Validation
            v_retained = val_confs >= t
            v_cov = float(np.mean(v_retained))
            v_acc = float(np.mean(val_correct[v_retained])) if np.sum(v_retained) > 0 else 1.0

            # Test
            t_retained = test_confs >= t
            t_cov = float(np.mean(t_retained))
            t_acc = float(np.mean(test_correct[t_retained])) if np.sum(t_retained) > 0 else 1.0
            t_err = 1.0 - t_acc

            records.append({
                "threshold": round(float(t), 4),
                "val_coverage": round(v_cov, 4),
                "val_selective_acc": round(v_acc, 4),
                "test_coverage": round(t_cov, 4),
                "test_selective_acc": round(t_acc, 4),
                "test_selective_error": round(t_err, 4),
                "test_retained_count": int(np.sum(t_retained)),
                "test_abstained_count": int(len(self.y_test) - np.sum(t_retained)),
            })

        df_curve = pd.DataFrame(records)

        # Calibrate two specific operating points on VALIDATION:
        # Operating Point A: Targeting 90% coverage on validation
        row_90cov = df_curve.iloc[(df_curve["val_coverage"] - 0.90).abs().argmin()]
        thresh_90cov = float(row_90cov["threshold"])

        # Operating Point B: Targeting 80% selective accuracy on validation
        row_80acc = df_curve[df_curve["val_selective_acc"] >= 0.80]
        if len(row_80acc) > 0:
            thresh_80acc = float(row_80acc.iloc[0]["threshold"])
        else:
            thresh_80acc = float(df_curve.iloc[-1]["threshold"])

        # Test evaluation at operating point A
        t_ret_a = test_confs >= thresh_90cov
        op_a = {
            "name": "Operating Point A (Target: ~90% Validation Coverage)",
            "calibrated_threshold": thresh_90cov,
            "val_coverage": float(row_90cov["val_coverage"]),
            "val_accuracy": float(row_90cov["val_selective_acc"]),
            "test_coverage": float(np.mean(t_ret_a)),
            "test_retained_samples": int(np.sum(t_ret_a)),
            "test_abstained_samples": int(len(self.y_test) - np.sum(t_ret_a)),
            "test_selective_accuracy": float(np.mean(test_correct[t_ret_a])),
            "accuracy_gain_pp": round(float(np.mean(test_correct[t_ret_a]) - np.mean(test_correct)) * 100, 2),
        }

        # Test evaluation at operating point B
        t_ret_b = test_confs >= thresh_80acc
        op_b = {
            "name": "Operating Point B (Target: >=80% Validation Selective Accuracy)",
            "calibrated_threshold": thresh_80acc,
            "val_coverage": float(df_curve[df_curve["threshold"] == thresh_80acc]["val_coverage"].iloc[0]),
            "val_accuracy": float(df_curve[df_curve["threshold"] == thresh_80acc]["val_selective_acc"].iloc[0]),
            "test_coverage": float(np.mean(t_ret_b)),
            "test_retained_samples": int(np.sum(t_ret_b)),
            "test_abstained_samples": int(len(self.y_test) - np.sum(t_ret_b)),
            "test_selective_accuracy": float(np.mean(test_correct[t_ret_b])),
            "accuracy_gain_pp": round(float(np.mean(test_correct[t_ret_b]) - np.mean(test_correct)) * 100, 2),
        }

        abstention_summary = {
            "operating_point_a_90pct_coverage": op_a,
            "operating_point_b_80pct_accuracy": op_b,
        }

        # Save curve CSV
        csv_path = self.output_dir / "abstention_results.csv"
        df_curve.to_csv(csv_path, index=False)
        logger.info("Saved %s", csv_path.name)

        return df_curve, abstention_summary

    def plot_abstention_curves(self, df_curve: pd.DataFrame, abstention_summary: Dict[str, Any]) -> None:
        """Plots Coverage vs Selective Accuracy trade-off curve."""
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

        # Panel A: Coverage vs Selective Accuracy
        ax1 = axes[0]
        ax1.plot(df_curve["test_coverage"] * 100, df_curve["test_selective_acc"] * 100, "b-o", linewidth=2, label="Phikon Mean Patch (Test Set)")
        ax1.axhline(66.38, color="gray", linestyle="--", label="Baseline (100% Coverage = 66.38%)")

        op_a = abstention_summary["operating_point_a_90pct_coverage"]
        op_b = abstention_summary["operating_point_b_80pct_accuracy"]
        ax1.plot(op_a["test_coverage"] * 100, op_a["test_selective_accuracy"] * 100, "r*", markersize=14, label=f"Op Point A (Cov: {op_a['test_coverage']*100:.1f}%, Acc: {op_a['test_selective_accuracy']*100:.1f}%)")
        ax1.plot(op_b["test_coverage"] * 100, op_b["test_selective_accuracy"] * 100, "g^", markersize=12, label=f"Op Point B (Cov: {op_b['test_coverage']*100:.1f}%, Acc: {op_b['test_selective_accuracy']*100:.1f}%)")

        ax1.set_xlabel("Cohort Coverage (% of Test Tiles Retained)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Selective Diagnostic Accuracy (%)", fontsize=11, fontweight="bold")
        ax1.set_title("Panel A: Coverage vs Selective Accuracy", fontsize=12, fontweight="bold")
        ax1.legend(loc="lower left", frameon=True)
        ax1.grid(True, linestyle=":", alpha=0.6)

        # Panel B: Risk-Coverage Curve (Selective Error vs Coverage)
        ax2 = axes[1]
        ax2.plot(df_curve["test_coverage"] * 100, df_curve["test_selective_error"] * 100, "r-s", linewidth=2, label="Selective Misclassification Rate")
        ax2.axhline(33.62, color="gray", linestyle="--", label="Baseline Error (100% Coverage = 33.62%)")

        ax2.set_xlabel("Cohort Coverage (% of Test Tiles Retained)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Selective Error Rate (%)", fontsize=11, fontweight="bold")
        ax2.set_title("Panel B: Risk-Coverage Curve (Error vs Coverage)", fontsize=12, fontweight="bold")
        ax2.legend(loc="upper left", frameon=True)
        ax2.grid(True, linestyle=":", alpha=0.6)

        plt.suptitle("GastricPath AI — Original Phase 5: Validation-Calibrated Selective Classification (Abstention)", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        save_path = self.output_dir / "abstention_tradeoff_curves.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info("Saved %s", save_path.name)

    def run_pipeline(self) -> Dict[str, Any]:
        """Runs the complete calibration and abstention pipeline."""
        logger.info("Running complete Original Phase 5 calibration engine...")

        # 1. Phikon Mean Patch
        val_logits_phikon, test_logits_phikon = self.get_phikon_logits()
        phikon_cal = self.evaluate_model_calibration("Phikon ViT-B/16 Mean Patch", val_logits_phikon, test_logits_phikon)
        self.plot_reliability_diagrams(phikon_cal, "reliability_diagrams_phikon.png")

        # 2. GastricCancerRNN
        val_logits_cnn, test_logits_cnn = self.extract_cnn_logits()
        cnn_cal = self.evaluate_model_calibration("GastricCancerRNN (CNN-BiLSTM)", val_logits_cnn, test_logits_cnn)
        self.plot_reliability_diagrams(cnn_cal, "reliability_diagrams_cnn.png")

        # 3. Selective Classification / Abstention on Phikon Mean Patch
        t_opt_phikon = phikon_cal["optimal_temperature_T"]
        val_probs_phikon = softmax(val_logits_phikon / t_opt_phikon)
        test_probs_phikon = softmax(test_logits_phikon / t_opt_phikon)
        df_abstention, abstention_summary = self.evaluate_selective_classification(val_probs_phikon, test_probs_phikon)
        self.plot_abstention_curves(df_abstention, abstention_summary)

        # 4. Save compilation JSON
        compilation = {
            "phikon_mean_patch_calibration": phikon_cal,
            "gastric_cancer_rnn_calibration": cnn_cal,
            "abstention_selective_classification": abstention_summary,
        }
        with open(self.output_dir / "calibration_metrics.json", "w") as f:
            json.dump(compilation, f, indent=2)
        logger.info("Saved calibration_metrics.json")
        return compilation


if __name__ == "__main__":
    engine = CalibrationEngine()
    engine.run_pipeline()

