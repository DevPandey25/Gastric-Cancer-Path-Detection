"""Phase 5: Explainable AI Engine for Frozen Phikon Histopathology Representations.

Implements:
1. Token-level Leave-One-Out (LOO) Feature Ablation on the winning Phase 4.4 Phikon Mean Patch model.
2. 32-sample explanation generation across all 8 canonical classes (seed 42).
3. 14x14 token attribution maps upsampled to 224x224 and rendered over H&E tiles.
4. Contact sheet visualization for all 32 samples.
5. Faithfulness deletion experiment on 8 representative test tiles (1 per class)
   comparing Most Important First (MIF) vs Random vs Least Important First (LIF) across
   25%, 50%, and 75% patch masking levels.
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
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import zoom
from sklearn.linear_model import LogisticRegression
import torch
from transformers import AutoImageProcessor, AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase5_explainability")

CANONICAL_CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]
NEUTRAL_MASK_COLOR = (128, 128, 128)  # Documented neutral gray baseline for H&E masking


class ExplainabilityPipeline:
    """End-to-end pipeline for Phase 5 Explainable AI."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self.workspace_root = Path(workspace_root or WORKSPACE_ROOT).resolve()
        self.output_dir = self.workspace_root / "research" / "explainability"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.maps_dir = self.output_dir / "attribution_maps"
        self.maps_dir.mkdir(parents=True, exist_ok=True)

        self.subset_dir = self.workspace_root / "research" / "foundation_models" / "cache" / "subset10pct"
        self.manifest_path = self.subset_dir / "manifest_subset.csv"
        self.image_base_dir = self.workspace_root / "data" / "raw" / "HMU-GC-HE-30K" / "all_image"

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing subset manifest: {self.manifest_path}")

        self.df_manifest = pd.read_csv(self.manifest_path)
        self.train_idx = np.where(self.df_manifest["split"] == "train")[0]
        self.test_idx = np.where(self.df_manifest["split"] == "test")[0]
        self.y_train = self.df_manifest.iloc[self.train_idx]["class_index"].to_numpy()

        # Reconstruct / load linear probe (C=10.0, lbfgs, random_state=42)
        self.clf = self._get_or_train_classifier()

        # Load Phikon model offline
        logger.info("Loading owkin/phikon model and processor...")
        self.processor = AutoImageProcessor.from_pretrained("owkin/phikon", local_files_only=True)
        self.model = AutoModel.from_pretrained("owkin/phikon", local_files_only=True)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def _get_or_train_classifier(self) -> LogisticRegression:
        model_save_path = self.output_dir / "fitted_phikon_linear_probe.joblib"
        if model_save_path.exists():
            logger.info("Loading pre-fitted linear probe from %s", model_save_path.name)
            return joblib.load(model_save_path)

        logger.info("Fitting linear probe (C=10.0) on Phikon train embeddings (N=%d)...", len(self.train_idx))
        mm = np.memmap(
            self.subset_dir / "phikon" / "mean_patch_embeddings.npy",
            dtype=np.float32,
            mode="r",
            shape=(len(self.df_manifest), 768),
        )
        X_tr = mm[self.train_idx]
        clf = LogisticRegression(C=10.0, max_iter=1000, solver="lbfgs", random_state=42)
        clf.fit(X_tr, self.y_train)
        joblib.dump(clf, model_save_path)
        logger.info("Saved fitted linear probe to %s", model_save_path.name)
        return clf

    def select_samples(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Selects 32 deterministic test samples (4 per class) and 8 faithfulness samples."""
        test_df = self.df_manifest[self.df_manifest["split"] == "test"].sort_values("relative_path").reset_index(drop=True)
        
        # 4 per class with seed 42
        samples_32 = (
            test_df.groupby("class_name", group_keys=False)
            .apply(lambda g: g.sample(n=4, random_state=42), include_groups=True)
            .sort_values(["class_index", "relative_path"])
            .reset_index(drop=True)
        )
        samples_32["sample_id"] = [f"EXP_{i:02d}_{row['class_name']}" for i, row in samples_32.iterrows()]

        # 1 per class for faithfulness (first sample of each class from samples_32)
        faith_indices = set(samples_32.groupby("class_name").first().index)
        samples_32["is_faithfulness_subset"] = samples_32["class_name"].isin(faith_indices) & (
            samples_32.groupby("class_name").cumcount() == 0
        )

        samples_32_path = self.output_dir / "manifest_explainability_32.csv"
        samples_32.to_csv(samples_32_path, index=False)
        logger.info("Saved 32-sample explanation manifest to %s", samples_32_path.name)

        samples_8 = samples_32[samples_32["is_faithfulness_subset"]].reset_index(drop=True)
        return samples_32, samples_8

    def extract_patch_tokens(self, image: Image.Image) -> np.ndarray:
        """Runs Phikon forward pass and returns spatial patch tokens shape (196, 768)."""
        inputs = self.processor(image.convert("RGB"), return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            # last_hidden_state: [1, 197, 768]; 0 is CLS, 1:197 are 196 spatial tokens
            patch_tokens = outputs.last_hidden_state[:, 1:, :].squeeze(0).cpu().numpy()
        return patch_tokens

    def compute_loo_attributions(
        self, patch_tokens: np.ndarray, target_class: int
    ) -> tuple[np.ndarray, float, int, np.ndarray]:
        """Computes leave-one-out feature ablation attribution for the target class.

        Returns:
            loo_attributions: shape (196,) - delta probability for each token
            base_prob: unablated probability for target class
            pred_class: unablated predicted class index
            base_all_probs: unablated full probability distribution
        """
        N = 196
        h_base = patch_tokens.mean(axis=0)  # [768]
        h_base_norm = h_base / (np.linalg.norm(h_base) + 1e-12)
        base_probs = self.clf.predict_proba(h_base_norm.reshape(1, -1))[0]
        base_target_prob = float(base_probs[target_class])
        pred_class = int(np.argmax(base_probs))

        # Vectorized leave-one-out means:
        # h_{\setminus p} = (N * h_base - z_p) / (N - 1)
        h_ablated = (N * h_base - patch_tokens) / (N - 1.0)  # [196, 768]
        norms = np.linalg.norm(h_ablated, axis=1, keepdims=True) + 1e-12
        h_ablated_norm = h_ablated / norms  # [196, 768]

        ablated_probs = self.clf.predict_proba(h_ablated_norm)[:, target_class]
        loo_attributions = base_target_prob - ablated_probs  # [196,]

        return loo_attributions, base_target_prob, pred_class, base_probs

    def generate_explanations(self, samples_32: pd.DataFrame) -> dict[str, Any]:
        """Generates LOO attribution maps, overlays, and contact sheet for all 32 samples."""
        logger.info("Generating explanations for 32 test samples...")
        t0 = time.time()
        results = []
        all_grids = []

        for idx, row in samples_32.iterrows():
            img_path = self.image_base_dir / row["relative_path"]
            img = Image.open(img_path).convert("RGB")
            true_cls = int(row["class_index"])

            tokens = self.extract_patch_tokens(img)
            # Evaluate attribution with respect to the predicted class (model's explanation)
            # and record true class score as well
            h_base = tokens.mean(axis=0)
            h_base_norm = h_base / (np.linalg.norm(h_base) + 1e-12)
            base_probs = self.clf.predict_proba(h_base_norm.reshape(1, -1))[0]
            pred_cls = int(np.argmax(base_probs))

            loo_attrs_pred, base_pred_prob, _, _ = self.compute_loo_attributions(tokens, pred_cls)
            loo_attrs_true, base_true_prob, _, _ = self.compute_loo_attributions(tokens, true_cls)

            grid_14x14 = loo_attrs_pred.reshape(14, 14)
            all_grids.append(grid_14x14)

            # Render individual attribution overlay
            overlay_path = self.maps_dir / f"attribution_{idx:02d}_{row['class_name']}_{row['relative_path'].replace('/', '_')}"
            self._render_overlay(img, grid_14x14, row, pred_cls, base_pred_prob, overlay_path)

            results.append({
                "sample_index": int(idx),
                "sample_id": row["sample_id"],
                "relative_path": row["relative_path"],
                "true_class": row["class_name"],
                "true_index": true_cls,
                "predicted_class": CANONICAL_CLASSES[pred_cls],
                "predicted_index": pred_cls,
                "confidence": round(base_pred_prob, 4),
                "true_class_prob": round(base_true_prob, 4),
                "is_correct": bool(pred_cls == true_cls),
                "attribution_min": round(float(loo_attrs_pred.min()), 6),
                "attribution_max": round(float(loo_attrs_pred.max()), 6),
                "attribution_mean": round(float(loo_attrs_pred.mean()), 6),
                "attribution_std": round(float(loo_attrs_pred.std()), 6),
                "overlay_file": overlay_path.name,
            })

        duration = time.time() - t0
        logger.info("Generated 32 explanations in %.2f seconds (%.2f s/sample).", duration, duration / 32)

        # Save attribution arrays
        np.save(self.output_dir / "attributions_grid_32x14x14.npy", np.array(all_grids))

        # Save results metadata
        with open(self.output_dir / "explanations_summary.json", "w") as f:
            json.dump({
                "phase": "Phase 5 Explainable AI",
                "method": "Token-level Leave-One-Out Feature Ablation",
                "encoder": "owkin/phikon (frozen)",
                "classifier": "LogisticRegression(C=10.0, solver='lbfgs')",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "total_samples": len(results),
                "runtime_seconds": round(duration, 3),
                "samples": results,
            }, f, indent=2)

        # Create publication-ready contact sheet
        self._generate_contact_sheet(samples_32, all_grids, results)

        return {"samples": results, "runtime_seconds": duration}

    def _render_overlay(
        self,
        img: Image.Image,
        grid_14x14: np.ndarray,
        row: pd.Series,
        pred_cls: int,
        conf: float,
        save_path: Path,
    ) -> None:
        """Renders side-by-side original tile, attribution heatmap, and blended overlay."""
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), dpi=150)

        # 1. Original H&E tile
        axes[0].imshow(img)
        axes[0].set_title(f"H&E Tile: {row['relative_path']}", fontsize=8)
        axes[0].axis("off")

        # 2. Raw 14x14 Attribution Heatmap
        im1 = axes[1].imshow(grid_14x14, cmap="viridis", interpolation="nearest")
        axes[1].set_title("14×14 Token LOO Attribution", fontsize=8)
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # 3. Blended Overlay (upsampled to 224x224)
        zoom_factor = 224.0 / 14.0
        upsampled = zoom(grid_14x14, zoom_factor, order=1)
        # Normalize to [0, 1] for display
        norm_map = (upsampled - upsampled.min()) / (upsampled.max() - upsampled.min() + 1e-12)

        axes[2].imshow(img)
        axes[2].imshow(norm_map, cmap="inferno", alpha=0.55, interpolation="bilinear")
        status_txt = "CORRECT" if row["class_index"] == pred_cls else "MISCLASSIFIED"
        color_txt = "darkgreen" if row["class_index"] == pred_cls else "darkred"
        axes[2].set_title(
            f"True: {row['class_name']} | Pred: {CANONICAL_CLASSES[pred_cls]} ({conf*100:.1f}%)\n[{status_txt}]",
            fontsize=8,
            fontweight="bold",
            color=color_txt,
        )
        axes[2].axis("off")

        plt.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

    def _generate_contact_sheet(
        self, samples_32: pd.DataFrame, grids: list[np.ndarray], results: list[dict[str, Any]]
    ) -> None:
        """Generates an 8x4 contact sheet displaying all 32 tiles with attribution overlays."""
        logger.info("Generating 8x4 explanation contact sheet...")
        fig, axes = plt.subplots(8, 4, figsize=(16, 28), dpi=150)
        zoom_factor = 224.0 / 14.0

        for idx, row in samples_32.iterrows():
            r = row["class_index"]
            c = idx % 4
            ax = axes[r, c]

            img_path = self.image_base_dir / row["relative_path"]
            img = Image.open(img_path).convert("RGB")
            grid = grids[idx]
            upsampled = zoom(grid, zoom_factor, order=1)
            norm_map = (upsampled - upsampled.min()) / (upsampled.max() - upsampled.min() + 1e-12)

            res = results[idx]
            pred_cls = res["predicted_class"]
            conf = res["confidence"]
            is_corr = res["is_correct"]

            ax.imshow(img)
            ax.imshow(norm_map, cmap="inferno", alpha=0.52, interpolation="bilinear")
            badge = "✓" if is_corr else "✗"
            title_col = "green" if is_corr else "red"
            ax.set_title(
                f"{badge} {row['class_name']} → {pred_cls} ({conf*100:.0f}%)",
                fontsize=9,
                fontweight="bold",
                color=title_col,
            )
            ax.axis("off")

        plt.suptitle(
            "GastricPath AI — Phase 5: Phikon Mean-Patch Attribution Gallery (32 Test Samples)\n"
            "Method: Token-level Leave-One-Out Feature Ablation | Colormap: Inferno Overlay (α=0.52)",
            fontsize=13,
            fontweight="bold",
            y=0.995,
        )
        plt.tight_layout()
        contact_sheet_path = self.output_dir / "explanation_contact_sheet.png"
        fig.savefig(contact_sheet_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved contact sheet to %s", contact_sheet_path.name)

    def run_faithfulness_experiment(self, samples_8: pd.DataFrame) -> dict[str, Any]:
        """Runs the deletion faithfulness experiment on 8 representative tiles (1 per class).

        Tests target-class probability drop across:
        - Most Important First (MIF): masking top 25%, 50%, 75% patches
        - Random (RND): masking random 25%, 50%, 75% patches (averaged across 5 seeds)
        - Least Important First (LIF): masking bottom 25%, 50%, 75% patches
        """
        logger.info("=" * 80)
        logger.info("Running Faithfulness Deletion Experiment on 8 representative test tiles...")
        logger.info("Masking levels: 0%%, 25%% (49 patches), 50%% (98 patches), 75%% (147 patches)")
        logger.info("Masking baseline: Constant Neutral Gray %s", NEUTRAL_MASK_COLOR)
        logger.info("=" * 80)

        levels = [0, 49, 98, 147]  # 0%, 25%, 50%, 75% of 196
        level_pcts = [0, 25, 50, 75]
        random_seeds = [101, 102, 103, 104, 105]

        per_sample_results = []
        curve_data = {
            "levels_pct": level_pcts,
            "mif_mean_probs": [],
            "rnd_mean_probs": [],
            "lif_mean_probs": [],
            "mif_mean_drops": [],
            "rnd_mean_drops": [],
            "lif_mean_drops": [],
        }

        all_mif_probs = {lvl: [] for lvl in level_pcts}
        all_rnd_probs = {lvl: [] for lvl in level_pcts}
        all_lif_probs = {lvl: [] for lvl in level_pcts}

        t0 = time.time()

        for idx, row in samples_8.iterrows():
            img_path = self.image_base_dir / row["relative_path"]
            img = Image.open(img_path).convert("RGB")
            true_cls = int(row["class_index"])

            # 1. Baseline unmasked prediction
            tokens = self.extract_patch_tokens(img)
            h_base = tokens.mean(axis=0)
            h_base_norm = h_base / (np.linalg.norm(h_base) + 1e-12)
            base_probs = self.clf.predict_proba(h_base_norm.reshape(1, -1))[0]
            pred_cls = int(np.argmax(base_probs))
            target_cls = pred_cls  # Faithful to the model's chosen target hypothesis
            base_target_prob = float(base_probs[target_cls])

            # 2. Compute LOO attributions to get patch ranking
            loo_attrs, _, _, _ = self.compute_loo_attributions(tokens, target_cls)
            # Ranked from highest attribution (most positive) to lowest
            ranked_patch_indices = np.argsort(-loo_attrs)

            sample_record = {
                "sample_id": row["sample_id"],
                "relative_path": row["relative_path"],
                "true_class": row["class_name"],
                "target_class": CANONICAL_CLASSES[target_cls],
                "base_prob": round(base_target_prob, 4),
                "is_correct": bool(pred_cls == true_cls),
                "deletions": {},
            }

            for k, pct in zip(levels, level_pcts):
                if k == 0:
                    sample_record["deletions"][pct] = {
                        "mif_prob": round(base_target_prob, 4),
                        "rnd_prob": round(base_target_prob, 4),
                        "lif_prob": round(base_target_prob, 4),
                        "mif_drop": 0.0,
                        "rnd_drop": 0.0,
                        "lif_drop": 0.0,
                    }
                    all_mif_probs[pct].append(base_target_prob)
                    all_rnd_probs[pct].append(base_target_prob)
                    all_lif_probs[pct].append(base_target_prob)
                    continue

                # A. Most Important First (MIF): mask top-k patches
                mif_patches = ranked_patch_indices[:k]
                mif_masked_img = self._apply_patch_mask(img, mif_patches)
                mif_tokens = self.extract_patch_tokens(mif_masked_img)
                mif_h = mif_tokens.mean(axis=0)
                mif_h_norm = mif_h / (np.linalg.norm(mif_h) + 1e-12)
                mif_prob = float(self.clf.predict_proba(mif_h_norm.reshape(1, -1))[0, target_cls])

                # B. Least Important First (LIF): mask bottom-k patches
                lif_patches = ranked_patch_indices[-k:]
                lif_masked_img = self._apply_patch_mask(img, lif_patches)
                lif_tokens = self.extract_patch_tokens(lif_masked_img)
                lif_h = lif_tokens.mean(axis=0)
                lif_h_norm = lif_h / (np.linalg.norm(lif_h) + 1e-12)
                lif_prob = float(self.clf.predict_proba(lif_h_norm.reshape(1, -1))[0, target_cls])

                # C. Random (RND): average across 5 random seeds
                rnd_probs = []
                for seed in random_seeds:
                    rng = np.random.RandomState(seed + idx * 10)
                    rnd_patches = rng.choice(196, size=k, replace=False)
                    rnd_masked_img = self._apply_patch_mask(img, rnd_patches)
                    rnd_tokens = self.extract_patch_tokens(rnd_masked_img)
                    rnd_h = rnd_tokens.mean(axis=0)
                    rnd_h_norm = rnd_h / (np.linalg.norm(rnd_h) + 1e-12)
                    rnd_p = float(self.clf.predict_proba(rnd_h_norm.reshape(1, -1))[0, target_cls])
                    rnd_probs.append(rnd_p)
                rnd_prob_avg = float(np.mean(rnd_probs))

                sample_record["deletions"][pct] = {
                    "mif_prob": round(mif_prob, 4),
                    "rnd_prob": round(rnd_prob_avg, 4),
                    "lif_prob": round(lif_prob, 4),
                    "mif_drop": round(base_target_prob - mif_prob, 4),
                    "rnd_drop": round(base_target_prob - rnd_prob_avg, 4),
                    "lif_drop": round(base_target_prob - lif_prob, 4),
                }

                all_mif_probs[pct].append(mif_prob)
                all_rnd_probs[pct].append(rnd_prob_avg)
                all_lif_probs[pct].append(lif_prob)

            per_sample_results.append(sample_record)
            logger.info(
                "  [%s] Base: %.3f | MIF@50%%: %.3f (Δ-%.3f) | RND@50%%: %.3f (Δ-%.3f) | LIF@50%%: %.3f (Δ-%.3f)",
                row["class_name"],
                base_target_prob,
                sample_record["deletions"][50]["mif_prob"],
                sample_record["deletions"][50]["mif_drop"],
                sample_record["deletions"][50]["rnd_prob"],
                sample_record["deletions"][50]["rnd_drop"],
                sample_record["deletions"][50]["lif_prob"],
                sample_record["deletions"][50]["lif_drop"],
            )

        duration = time.time() - t0
        logger.info("Faithfulness deletion experiment completed in %.2f seconds.", duration)

        # Aggregate curve data
        for pct in level_pcts:
            curve_data["mif_mean_probs"].append(round(float(np.mean(all_mif_probs[pct])), 4))
            curve_data["rnd_mean_probs"].append(round(float(np.mean(all_rnd_probs[pct])), 4))
            curve_data["lif_mean_probs"].append(round(float(np.mean(all_lif_probs[pct])), 4))

            base_mean = float(np.mean(all_mif_probs[0]))
            curve_data["mif_mean_drops"].append(round(base_mean - float(np.mean(all_mif_probs[pct])), 4))
            curve_data["rnd_mean_drops"].append(round(base_mean - float(np.mean(all_rnd_probs[pct])), 4))
            curve_data["lif_mean_drops"].append(round(base_mean - float(np.mean(all_lif_probs[pct])), 4))

        # Save JSON results
        faith_output = {
            "experiment": "Phase 5 Faithfulness Deletion Test",
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "samples_count": len(samples_8),
            "mask_baseline_color": NEUTRAL_MASK_COLOR,
            "deletion_levels_pct": level_pcts,
            "aggregate_curves": curve_data,
            "per_sample_results": per_sample_results,
            "runtime_seconds": round(duration, 3),
        }
        with open(self.output_dir / "deletion_faithfulness_results.json", "w") as f:
            json.dump(faith_output, f, indent=2)

        # Save flat CSV for tabular inspection
        rows_csv = []
        for s in per_sample_results:
            for pct in level_pcts:
                d = s["deletions"][pct]
                rows_csv.append({
                    "sample_id": s["sample_id"],
                    "relative_path": s["relative_path"],
                    "true_class": s["true_class"],
                    "target_class": s["target_class"],
                    "deletion_pct": pct,
                    "mif_prob": d["mif_prob"],
                    "rnd_prob": d["rnd_prob"],
                    "lif_prob": d["lif_prob"],
                    "mif_drop": d["mif_drop"],
                    "rnd_drop": d["rnd_drop"],
                    "lif_drop": d["lif_drop"],
                })
        pd.DataFrame(rows_csv).to_csv(self.output_dir / "deletion_faithfulness_results.csv", index=False)

        # Plot deletion curve figure
        self._plot_faithfulness_curves(curve_data, per_sample_results)

        return faith_output

    def _apply_patch_mask(self, image: Image.Image, patch_indices: list[int] | np.ndarray) -> Image.Image:
        """Replaces designated 16x16 patch regions with neutral gray in the 224x224 image."""
        arr = np.array(image.convert("RGB")).copy()
        for idx in patch_indices:
            r = idx // 14
            c = idx % 14
            y0, y1 = r * 16, (r + 1) * 16
            x0, x1 = c * 16, (c + 1) * 16
            arr[y0:y1, x0:x1, :] = NEUTRAL_MASK_COLOR
        return Image.fromarray(arr)

    def _plot_faithfulness_curves(
        self, curve_data: dict[str, Any], per_sample_results: list[dict[str, Any]]
    ) -> None:
        """Plots the deletion faithfulness comparison curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
        x = curve_data["levels_pct"]

        # Left plot: Mean Remaining Target Probability
        ax1.plot(x, curve_data["mif_mean_probs"], "o-", color="#D35E60", linewidth=2.5, label="Most Important First (MIF)")
        ax1.plot(x, curve_data["rnd_mean_probs"], "s--", color="#7293CB", linewidth=2.0, label="Random Masking (Avg 5 seeds)")
        ax1.plot(x, curve_data["lif_mean_probs"], "^-.", color="#84BA5B", linewidth=2.0, label="Least Important First (LIF)")
        ax1.set_xlabel("Percentage of Image Patches Masked (%)", fontsize=10)
        ax1.set_ylabel("Mean Target Class Probability", fontsize=10)
        ax1.set_title("Faithfulness: Target Probability Deletion Curve\n(Sharp Drop in MIF indicates faithful attribution)", fontsize=11, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_ylim(-0.05, 1.0)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(fontsize=9, loc="upper right")

        # Right plot: Mean Cumulative Score Drop (Delta P)
        ax2.plot(x, curve_data["mif_mean_drops"], "o-", color="#D35E60", linewidth=2.5, label="MIF Score Drop")
        ax2.plot(x, curve_data["rnd_mean_drops"], "s--", color="#7293CB", linewidth=2.0, label="Random Score Drop")
        ax2.plot(x, curve_data["lif_mean_drops"], "^-.", color="#84BA5B", linewidth=2.0, label="LIF Score Drop")
        ax2.set_xlabel("Percentage of Image Patches Masked (%)", fontsize=10)
        ax2.set_ylabel("Mean Target Probability Drop (ΔP)", fontsize=10)
        ax2.set_title("Score Degradation by Masking Strategy\n(MIF Drop >> Random >> LIF demonstrates rank validity)", fontsize=11, fontweight="bold")
        ax2.set_xticks(x)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(fontsize=9, loc="upper left")

        plt.tight_layout()
        plot_path = self.output_dir / "faithfulness_deletion_curves.png"
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved faithfulness deletion curves plot to %s", plot_path.name)


def run_phase5() -> None:
    """Executes the full Phase 5 Explainable AI workflow."""
    logger.info("=" * 80)
    logger.info("STARTING GASTRICPATH AI — PHASE 5: EXPLAINABLE AI IMPLEMENTATION")
    logger.info("=" * 80)
    t_start = time.time()

    pipeline = ExplainabilityPipeline()
    samples_32, samples_8 = pipeline.select_samples()

    # 1. Generate explanations for 32 samples
    pipeline.generate_explanations(samples_32)

    # 2. Run deletion faithfulness experiment on 8 samples
    pipeline.run_faithfulness_experiment(samples_8)

    # 3. Write reproducible configuration
    config = {
        "phase": 5,
        "title": "Explainable AI for Frozen Phikon Representations",
        "method": "Token-level Leave-One-Out Feature Ablation (LOO)",
        "encoder": {
            "model_id": "owkin/phikon",
            "architecture": "ViT-B/16",
            "weights_provenance": "TCGA pan-cancer iBOT self-supervised pretraining",
            "license": "Owkin Non-Commercial Academic License",
            "embedding_dim": 768,
            "patch_size": 16,
            "grid_dimensions": [14, 14],
            "total_tokens": 196,
        },
        "classifier": {
            "type": "Multinomial Logistic Regression",
            "regularization_C": 10.0,
            "solver": "lbfgs",
            "training_split": "Phase 4.4 Stratified Train Split (N=2,177)",
            "test_split_macro_f1": 0.6654,
            "test_split_accuracy": 0.6638,
        },
        "sampling": {
            "seed": 42,
            "total_explanation_samples": 32,
            "samples_per_class": 4,
            "faithfulness_samples": 8,
            "faithfulness_per_class": 1,
            "split_source": "Phase 4.4 10% Stratified Subset Test Split (N=467)",
        },
        "faithfulness_experiment": {
            "deletion_levels_pct": [0, 25, 50, 75],
            "deletion_patch_counts": [0, 49, 98, 147],
            "mask_baseline_color_rgb": [128, 128, 128],
            "random_seeds": [101, 102, 103, 104, 105],
        },
        "execution_telemetry": {
            "total_elapsed_seconds": round(time.time() - t_start, 2),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    with open(pipeline.output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info("=" * 80)
    logger.info("PHASE 5 COMPLETE in %.2f seconds", time.time() - t_start)
    logger.info("All artifacts written to: %s", pipeline.output_dir)
    logger.info("=" * 80)


if __name__ == "__main__":
    run_phase5()

