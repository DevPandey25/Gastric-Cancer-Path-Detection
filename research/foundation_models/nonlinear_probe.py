"""Nonlinear Probing & Evaluation Engine for GastricPath AI (Phase 4.3).

Evaluates downstream classification utility of frozen DINOv2 ViT-S/14 representations
using compact non-linear Multi-Layer Perceptron (MLP) probes:
A. CLS token embeddings (shape [31096, 384])
B. Mean spatial patch-token embeddings (shape [31096, 384])

Methodology:
- Encoder is strictly frozen (0 encoder parameters loaded for gradients).
- Train split (N=21,767) is used exclusively for gradient updates.
- Validation split (N=4,664) is used exclusively for hyperparameter tuning & early stopping.
- Test split (N=4,665) is evaluated exactly once on the final selected model.
- Fixed random seed = 42 for strict reproducibility.
- Comprehensive multiclass classification metrics with Wilson and bootstrap CIs.
- Explicitly documents that comparisons to historical baselines carry cohort incommensurability caveats.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

from research.benchmark.metrics import compute_comprehensive_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CANONICAL_CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]


def set_seed(seed: int = 42) -> None:
    """Sets deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PathologyNonlinearProbe(nn.Module):
    """Compact 2-layer MLP probe for frozen foundation model representations."""

    def __init__(
        self,
        in_features: int = 384,
        hidden_dim: int = 256,
        num_classes: int = 8,
        dropout: float = 0.2,
        activation: str = "gelu",
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.activation_name = activation
        self.use_layer_norm = use_layer_norm

        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()

        if activation.lower() == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU()

        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.norm(h)
        h = self.act(h)
        h = self.drop(h)
        return self.fc2(h)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class NonlinearProbingEngine:
    """Manages frozen nonlinear probe training, tuning, and rigorous test evaluation."""

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

        self.y_train = self.df_manifest.iloc[self.train_idx]["class_index"].to_numpy(dtype=np.int64)
        self.y_val = self.df_manifest.iloc[self.val_idx]["class_index"].to_numpy(dtype=np.int64)
        self.y_test = self.df_manifest.iloc[self.test_idx]["class_index"].to_numpy(dtype=np.int64)

        logger.info(
            "Splits: Train = %d | Val = %d | Test = %d",
            len(self.train_idx),
            len(self.val_idx),
            len(self.test_idx),
        )

    def train_single_mlp(
        self,
        X_tr: torch.Tensor,
        y_tr: torch.Tensor,
        X_va: torch.Tensor,
        y_va: np.ndarray,
        config: dict[str, Any],
        max_epochs: int = 35,
        patience: int = 7,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Trains a single MLP candidate with early stopping on validation Macro-F1."""
        set_seed(seed)
        hidden_dim = config["hidden_dim"]
        lr = config["lr"]
        weight_decay = config["weight_decay"]
        dropout = config["dropout"]
        activation = config.get("activation", "gelu")
        batch_size = config.get("batch_size", 256)

        model = PathologyNonlinearProbe(
            in_features=384,
            hidden_dim=hidden_dim,
            num_classes=8,
            dropout=dropout,
            activation=activation,
            use_layer_norm=True,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        train_dataset = TensorDataset(X_tr, y_tr)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )

        best_val_macro_f1 = -1.0
        best_val_acc = -1.0
        best_epoch = 0
        best_state = None
        epochs_without_improvement = 0
        history = []

        t0_train = time.time()
        for epoch in range(1, max_epochs + 1):
            t0_ep = time.time()
            model.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_total = 0

            for bx, by in train_loader:
                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * len(by)
                preds = logits.argmax(dim=-1)
                train_correct += (preds == by).sum().item()
                train_total += len(by)

            train_loss = train_loss_sum / train_total
            train_acc = train_correct / train_total

            # Validation pass
            model.eval()
            with torch.no_grad():
                val_logits = model(X_va)
                val_loss = criterion(val_logits, torch.from_numpy(y_va)).item()
                val_preds = val_logits.argmax(dim=-1).numpy()
                val_acc = float(accuracy_score(y_va, val_preds))
                val_macro_f1 = float(f1_score(y_va, val_preds, average="macro", zero_division=0))

            ep_time = time.time() - t0_ep
            ep_record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
                "val_macro_f1": round(val_macro_f1, 4),
                "epoch_time_sec": round(ep_time, 2),
            }
            history.append(ep_record)

            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                best_val_acc = val_acc
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logger.info(
                    "    Early stopping triggered at epoch %d (Best epoch: %d | Val Macro-F1: %.2f%%)",
                    epoch,
                    best_epoch,
                    best_val_macro_f1 * 100,
                )
                break

        total_train_time = time.time() - t0_train
        return {
            "config": config,
            "parameter_count": model.count_parameters(),
            "total_epochs_trained": len(history),
            "best_epoch": best_epoch,
            "best_val_macro_f1": round(best_val_macro_f1, 4),
            "best_val_acc": round(best_val_acc, 4),
            "total_train_time_sec": round(total_train_time, 2),
            "best_state_dict": best_state,
            "history": history,
        }

    def tune_hyperparameters(
        self,
        candidate_configs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Sweeps hyperparameter grid strictly on Validation Macro-F1 for CLS and Mean Patch."""
        if candidate_configs is None:
            candidate_configs = [
                {"name": "MLP-128-lr1e3-wd1e4-dp0.1", "hidden_dim": 128, "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.1, "activation": "gelu"},
                {"name": "MLP-128-lr3e4-wd1e3-dp0.2", "hidden_dim": 128, "lr": 3e-4, "weight_decay": 1e-3, "dropout": 0.2, "activation": "gelu"},
                {"name": "MLP-256-lr1e3-wd1e4-dp0.2", "hidden_dim": 256, "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.2, "activation": "gelu"},
                {"name": "MLP-256-lr3e4-wd1e3-dp0.3", "hidden_dim": 256, "lr": 3e-4, "weight_decay": 1e-3, "dropout": 0.3, "activation": "gelu"},
            ]

        logger.info("=" * 70)
        logger.info("NONLINEAR PROBE HYPERPARAMETER TUNING (VALIDATION-ONLY SELECTION)")
        logger.info("=" * 70)

        proc = psutil.Process(os.getpid())
        tuning_results: dict[str, Any] = {"candidate_configs": candidate_configs, "cls": {}, "mean_patch": {}}

        for repr_type, path in [("cls", self.cls_path), ("mean_patch", self.patch_path)]:
            logger.info("Tuning nonlinear probe on %s embeddings...", repr_type.upper())
            mm = np.memmap(path, dtype=np.float32, mode="r", shape=(self.total_samples, 384))
            X_tr = torch.from_numpy(mm[self.train_idx])
            y_tr = torch.from_numpy(self.y_train)
            X_va = torch.from_numpy(mm[self.val_idx])

            best_f1 = -1.0
            best_config_idx = 0
            grid_records = []
            best_model_state = None
            best_history = None

            for c_idx, cfg in enumerate(candidate_configs):
                logger.info(
                    "  [%s Config %d/%d] %s (Hidden: %d, lr: %s, wd: %s, dp: %s)...",
                    repr_type.upper(),
                    c_idx + 1,
                    len(candidate_configs),
                    cfg["name"],
                    cfg["hidden_dim"],
                    cfg["lr"],
                    cfg["weight_decay"],
                    cfg["dropout"],
                )
                res = self.train_single_mlp(
                    X_tr=X_tr,
                    y_tr=y_tr,
                    X_va=X_va,
                    y_va=self.y_val,
                    config=cfg,
                    max_epochs=35,
                    patience=7,
                    seed=42,
                )

                rss_mb = proc.memory_info().rss / (1024 * 1024)
                rec = {
                    "config": cfg,
                    "parameters": res["parameter_count"],
                    "best_epoch": res["best_epoch"],
                    "total_epochs": res["total_epochs_trained"],
                    "val_accuracy": res["best_val_acc"],
                    "val_macro_f1": res["best_val_macro_f1"],
                    "train_time_sec": res["total_train_time_sec"],
                    "rss_mb": round(rss_mb, 2),
                    "history": res["history"],
                }
                grid_records.append(rec)
                logger.info(
                    "    -> Peak Val Acc: %6.2f%% | Peak Val Macro-F1: %6.2f%% (Epoch %d in %.1fs)",
                    rec["val_accuracy"] * 100,
                    rec["val_macro_f1"] * 100,
                    rec["best_epoch"],
                    rec["train_time_sec"],
                )

                if res["best_val_macro_f1"] > best_f1:
                    best_f1 = res["best_val_macro_f1"]
                    best_config_idx = c_idx
                    best_model_state = res["best_state_dict"]
                    best_history = res["history"]

            best_cfg = candidate_configs[best_config_idx]
            tuning_results[repr_type] = {
                "grid": grid_records,
                "best_config": best_cfg,
                "best_val_macro_f1": best_f1,
                "best_val_accuracy": grid_records[best_config_idx]["val_accuracy"],
                "best_epoch": grid_records[best_config_idx]["best_epoch"],
                "best_history": best_history,
            }
            logger.info(
                "Selected optimal MLP configuration for %s: %s (Val Macro-F1 = %.2f%%)",
                repr_type.upper(),
                best_cfg["name"],
                best_f1 * 100,
            )

            # Save the best model checkpoint weights
            ckpt_file = self.ckpt_dir / f"nonlinear_probe_{repr_type}.pt"
            torch.save({
                "model_state_dict": best_model_state,
                "config": best_cfg,
                "representation": repr_type,
                "best_val_macro_f1": best_f1,
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            }, ckpt_file)
            logger.info("Saved best MLP checkpoint to %s", ckpt_file.name)

        # Save tuning log
        log_path = self.eval_dir / "nonlinear_probe_tuning_log.json"
        with open(log_path, "w") as f:
            json.dump({
                "candidate_configs": candidate_configs,
                "cls": {k: v for k, v in tuning_results["cls"].items() if k != "best_state_dict"},
                "mean_patch": {k: v for k, v in tuning_results["mean_patch"].items() if k != "best_state_dict"},
            }, f, indent=2)
        logger.info("Saved hyperparameter tuning log to %s", log_path.name)
        return tuning_results

    def evaluate_test_split(
        self,
        tuning_results: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Evaluates the final selected MLP probes on the Test split exactly once."""
        logger.info("=" * 70)
        logger.info("FINAL NONLINEAR PROBE TEST EVALUATION (EXACTLY ONCE PER REPRESENTATION)")
        logger.info("=" * 70)

        test_results = {}

        for repr_type, path in [("cls", self.cls_path), ("mean_patch", self.patch_path)]:
            best_cfg = tuning_results[repr_type]["best_config"]
            ckpt_file = self.ckpt_dir / f"nonlinear_probe_{repr_type}.pt"
            logger.info("Loading selected %s probe from %s...", repr_type.upper(), ckpt_file.name)

            ckpt_data = torch.load(ckpt_file, map_location="cpu", weights_only=True)
            model = PathologyNonlinearProbe(
                in_features=384,
                hidden_dim=best_cfg["hidden_dim"],
                num_classes=8,
                dropout=best_cfg["dropout"],
                activation=best_cfg.get("activation", "gelu"),
                use_layer_norm=True,
            )
            model.load_state_dict(ckpt_data["model_state_dict"])
            model.eval()

            mm = np.memmap(path, dtype=np.float32, mode="r", shape=(self.total_samples, 384))
            X_te = torch.from_numpy(mm[self.test_idx])

            # Deterministic inference on Test split exactly once
            t0_inf = time.time()
            with torch.no_grad():
                logits = model(X_te)
                test_probs = F.softmax(logits, dim=-1).numpy()
                test_preds = np.argmax(test_probs, axis=-1)
            inf_time = time.time() - t0_inf

            # Compute comprehensive metrics with Wilson and bootstrap CIs
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
            pred_csv_path = self.eval_dir / f"nonlinear_probe_{repr_type}_test_predictions.csv"
            pred_df.to_csv(pred_csv_path, index=False)
            logger.info("Saved test predictions to %s", pred_csv_path.name)

            overall = metrics["overall"]
            result_summary = {
                "probe_type": "nonlinear_mlp",
                "representation": repr_type,
                "model_name": "dinov2_vits14",
                "embedding_dim": 384,
                "selected_config": best_cfg,
                "parameter_count": model.count_parameters(),
                "best_val_epoch": tuning_results[repr_type]["best_epoch"],
                "best_val_macro_f1": tuning_results[repr_type]["best_val_macro_f1"],
                "inference_time_seconds": round(inf_time, 4),
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

            res_path = self.eval_dir / f"nonlinear_probe_{repr_type}_results.json"
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
        """Generates publication-quality charts for Phase 4.3 nonlinear probing."""
        logger.info("Generating nonlinear probe visualization figures in %s...", self.viz_dir)

        # 1. Training & Validation Loss / Macro-F1 Curves
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        hist_cls = tuning_results["cls"]["best_history"]
        hist_patch = tuning_results["mean_patch"]["best_history"]

        # Panel A: Loss Curves
        ep_cls = [r["epoch"] for r in hist_cls]
        tr_loss_cls = [r["train_loss"] for r in hist_cls]
        va_loss_cls = [r["val_loss"] for r in hist_cls]

        ep_patch = [r["epoch"] for r in hist_patch]
        tr_loss_patch = [r["train_loss"] for r in hist_patch]
        va_loss_patch = [r["val_loss"] for r in hist_patch]

        axes[0].plot(ep_cls, tr_loss_cls, label="Train Loss (CLS)", color="#1d3557", linestyle=":")
        axes[0].plot(ep_cls, va_loss_cls, label="Val Loss (CLS)", color="#1d3557", linewidth=2)
        axes[0].plot(ep_patch, tr_loss_patch, label="Train Loss (Patch)", color="#e63946", linestyle=":")
        axes[0].plot(ep_patch, va_loss_patch, label="Val Loss (Patch)", color="#e63946", linewidth=2)
        axes[0].set_title("Nonlinear Probe Optimization Loss", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("Epoch", fontsize=10)
        axes[0].set_ylabel("Cross-Entropy Loss", fontsize=10)
        axes[0].grid(True, linestyle="--", alpha=0.5)
        axes[0].legend(frameon=True, fontsize=9)

        # Panel B: Validation Macro-F1 Progression
        va_f1_cls = [r["val_macro_f1"] * 100 for r in hist_cls]
        va_f1_patch = [r["val_macro_f1"] * 100 for r in hist_patch]

        axes[1].plot(ep_cls, va_f1_cls, marker="o", label="Val Macro-F1 (CLS)", color="#1d3557", linewidth=2)
        axes[1].plot(ep_patch, va_f1_patch, marker="s", label="Val Macro-F1 (Patch)", color="#e63946", linewidth=2)
        axes[1].axvline(tuning_results["cls"]["best_epoch"], color="#1d3557", linestyle="--", alpha=0.7, label=f"Best CLS (Ep {tuning_results['cls']['best_epoch']})")
        axes[1].axvline(tuning_results["mean_patch"]["best_epoch"], color="#e63946", linestyle="--", alpha=0.7, label=f"Best Patch (Ep {tuning_results['mean_patch']['best_epoch']})")
        axes[1].set_title("Validation Macro-F1 Early Stopping Trajectory", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Epoch", fontsize=10)
        axes[1].set_ylabel("Macro-F1 (%)", fontsize=10)
        axes[1].grid(True, linestyle="--", alpha=0.5)
        axes[1].legend(frameon=True, fontsize=9)

        plt.suptitle("GastricPath AI — Phase 4.3 Nonlinear MLP Probe Convergence", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fig_loss_path = self.viz_dir / "nonlinear_probe_loss_curves.png"
        plt.savefig(fig_loss_path, dpi=200)
        plt.close()
        logger.info("Saved: %s", fig_loss_path.name)

        # 2. Confusion Matrix Heatmap (Mean Patch MLP on Test Split)
        cm = np.array(patch_results["confusion_matrix"])
        cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100

        plt.figure(figsize=(8, 6.5))
        plt.imshow(cm_pct, interpolation="nearest", cmap="Purples")
        plt.title("GastricPath AI — Mean Patch Nonlinear MLP Probe Confusion Matrix (% Test)", fontsize=11, fontweight="bold")
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
        plt.xlabel("Predicted Class (Nonlinear MLP Probe)", fontsize=11)
        plt.tight_layout()
        fig_cm_path = self.viz_dir / "nonlinear_probe_confusion_matrix.png"
        plt.savefig(fig_cm_path, dpi=200)
        plt.close()
        logger.info("Saved: %s", fig_cm_path.name)

        # 3. Linear Probe vs Nonlinear MLP Probe Per-Class F1 Comparison
        # Load linear probe results if available
        linear_patch_file = self.eval_dir / "linear_probe_mean_patch_results.json"
        if linear_patch_file.exists():
            with open(linear_patch_file, "r") as f:
                lin_data = json.load(f)
            lin_f1 = [lin_data["per_class_metrics"][c]["f1_score"] * 100 for c in CANONICAL_CLASSES]
        else:
            lin_f1 = [0.0] * len(CANONICAL_CLASSES)

        mlp_f1 = [patch_results["per_class_metrics"][c]["f1_score"] * 100 for c in CANONICAL_CLASSES]

        x = np.arange(len(CANONICAL_CLASSES))
        width = 0.35

        plt.figure(figsize=(10, 5))
        plt.bar(x - width / 2, lin_f1, width, label="Linear Probe (L-BFGS)", color="#457b9d")
        plt.bar(x + width / 2, mlp_f1, width, label="Nonlinear MLP Probe", color="#7209b7")

        plt.title("GastricPath AI — Linear vs. Nonlinear Readout (Mean Patch Embeddings)", fontsize=12, fontweight="bold")
        plt.xlabel("Histopathological Tissue Class", fontsize=11)
        plt.ylabel("Test F1-Score (%)", fontsize=11)
        plt.xticks(x, CANONICAL_CLASSES, fontsize=10, fontweight="bold")
        plt.ylim(0, 105)
        plt.legend(frameon=True)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig_comp_path = self.viz_dir / "nonlinear_probe_linear_vs_mlp_comparison.png"
        plt.savefig(fig_comp_path, dpi=200)
        plt.close()
        logger.info("Saved: %s", fig_comp_path.name)


def run_nonlinear_probing(force_retune: bool = False):
    engine = NonlinearProbingEngine()
    tuning_log_path = engine.eval_dir / "nonlinear_probe_tuning_log.json"
    ckpt_cls = engine.ckpt_dir / "nonlinear_probe_cls.pt"
    ckpt_patch = engine.ckpt_dir / "nonlinear_probe_mean_patch.pt"

    if tuning_log_path.exists() and ckpt_cls.exists() and ckpt_patch.exists() and not force_retune:
        logger.info("Loading existing tuning log from %s...", tuning_log_path.name)
        with open(tuning_log_path, "r") as f:
            tuning_results = json.load(f)
    else:
        tuning_results = engine.tune_hyperparameters()

    cls_results, patch_results = engine.evaluate_test_split(tuning_results)
    engine.generate_visualizations(cls_results, patch_results, tuning_results)

    print("\n" + "=" * 80)
    print("PHASE 4.3 NONLINEAR PROBING & EVALUATION COMPLETE")
    print(f"  CLS Token MLP Test Accuracy:        {cls_results['overall_metrics']['accuracy']*100:.2f}% (Macro-F1: {cls_results['overall_metrics']['macro_f1']*100:.2f}%)")
    print(f"  Mean Patch MLP Test Accuracy:       {patch_results['overall_metrics']['accuracy']*100:.2f}% (Macro-F1: {patch_results['overall_metrics']['macro_f1']*100:.2f}%)")
    print("=" * 80)


if __name__ == "__main__":
    run_nonlinear_probing()

