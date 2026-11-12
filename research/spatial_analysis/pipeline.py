"""Phase 8: Intra-Patch Spatial Representation Analysis Pipeline.

Analyzes within-patch spatial token patterns of the frozen owkin/phikon ViT-B/16
representation on a deterministic, class-balanced held-out test cohort (N=96, 12 per class, seed 42):
1. Verifies 14x14 token grid (196 spatial tokens + 1 CLS token, d=768).
2. Compares 3 representations: CLS, Global Mean Patch, and 2x2 Regional Pooling (4 quadrants).
3. Computes regional spatial descriptors: token norm dispersion, pairwise quadrant cosine similarities,
   regional L2 distances, dispersion around center, and CLS vs. Global Mean similarity.
4. Generates publication-ready visualizations: 14x14 token feature-norm heatmaps, 14x14 LOO attribution overlays,
   2x2 regional pooling quadrant illustrations, and an 8-class contact sheet.
"""

from __future__ import annotations

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
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from research.foundation_models.preprocessing import DeterministicPathologyPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase8_spatial")

CANONICAL_CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]
QUADRANTS = ["TL", "TR", "BL", "BR"]
QUADRANT_SLICES = {
    "TL": (slice(0, 7), slice(0, 7)),     # Top-Left: rows 0..6, cols 0..6 (49 tokens)
    "TR": (slice(0, 7), slice(7, 14)),    # Top-Right: rows 0..6, cols 7..13 (49 tokens)
    "BL": (slice(7, 14), slice(0, 7)),    # Bottom-Left: rows 7..13, cols 0..6 (49 tokens)
    "BR": (slice(7, 14), slice(7, 14)),   # Bottom-Right: rows 7..13, cols 7..13 (49 tokens)
}


class SpatialAnalysisPipeline:
    """Intra-patch spatial token representation analyzer for frozen Phikon ViT-B/16."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self.workspace_root = Path(workspace_root or WORKSPACE_ROOT).resolve()
        self.output_dir = self.workspace_root / "research" / "spatial_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        self.subset_dir = self.workspace_root / "research" / "foundation_models" / "cache" / "subset10pct"
        self.dataset_root = self.workspace_root / "data" / "raw" / "HMU-GC-HE-30K" / "all_image"
        self.manifest_path = self.subset_dir / "manifest_subset.csv"

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing subset manifest: {self.manifest_path}")

        self.df_manifest = pd.read_csv(self.manifest_path)

        # Preprocessor and linear probe
        self.preprocessor = DeterministicPathologyPreprocessor(image_size=224)
        probe_path = self.workspace_root / "research" / "explainability" / "fitted_phikon_linear_probe.joblib"
        self.clf: LogisticRegression = joblib.load(probe_path)
        logger.info("Loaded linear probe from %s", probe_path.name)

        # Verify offline model configuration
        self.config = AutoConfig.from_pretrained("owkin/phikon", local_files_only=True)
        self.model: AutoModel | None = None  # Lazy loaded

    def verify_architecture(self) -> Dict[str, Any]:
        """Verifies ViT-B/16 architecture and spatial token dimensions."""
        cfg = self.config
        assert cfg.image_size == 224, f"Expected image_size 224, got {cfg.image_size}"
        assert cfg.patch_size == 16, f"Expected patch_size 16, got {cfg.patch_size}"
        assert cfg.hidden_size == 768, f"Expected hidden_size 768, got {cfg.hidden_size}"
        grid_h = cfg.image_size // cfg.patch_size
        grid_w = cfg.image_size // cfg.patch_size
        assert grid_h == 14 and grid_w == 14, f"Expected 14x14 grid, got {grid_h}x{grid_w}"
        n_patches = grid_h * grid_w
        assert n_patches == 196, f"Expected 196 patches, got {n_patches}"

        arch_info = {
            "model_type": cfg.model_type,
            "image_size": cfg.image_size,
            "patch_size": cfg.patch_size,
            "hidden_size": cfg.hidden_size,
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "grid_dimensions": [grid_h, grid_w],
            "num_spatial_tokens": n_patches,
            "total_tokens_with_cls": n_patches + 1,
            "quadrant_partition": {
                q: {"row_slice": [s[0].start, s[0].stop], "col_slice": [s[1].start, s[1].stop], "num_tokens": 49}
                for q, s in QUADRANT_SLICES.items()
            },
        }
        logger.info("Architecture verified: ViT-B/16 (224x224, patch 16x16, 14x14=196 tokens, d=768).")
        return arch_info

    def _load_model(self) -> AutoModel:
        if self.model is None:
            logger.info("Instantiating frozen owkin/phikon model on CPU...")
            torch.set_num_threads(10)
            self.model = AutoModel.from_pretrained("owkin/phikon", local_files_only=True)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
        return self.model

    def select_deterministic_cohort(self, seed: int = 42, n_per_class: int = 12) -> pd.DataFrame:
        """Selects a deterministic, class-balanced held-out test cohort (N=96)."""
        test_df = self.df_manifest[self.df_manifest["split"] == "test"].sort_values("relative_path").reset_index(drop=True)
        assert len(test_df) == 467

        # Sample exactly n_per_class per canonical class
        sampled_rows = []
        for c in CANONICAL_CLASSES:
            df_c = test_df[test_df["class_name"] == c]
            sampled_c = df_c.sample(n=n_per_class, random_state=seed)
            sampled_rows.append(sampled_c)

        df_96 = pd.concat(sampled_rows, ignore_index=True)
        df_96 = df_96.sort_values(["class_index", "relative_path"]).reset_index(drop=True)
        df_96["spatial_sample_id"] = [f"SPAT_{i:02d}_{row['class_name']}" for i, row in df_96.iterrows()]

        manifest_path = self.output_dir / "spatial_subset_manifest_96.csv"
        df_96.to_csv(manifest_path, index=False)
        logger.info("Saved deterministic 96-sample spatial manifest: %s (12 per class, 8 classes)", manifest_path.name)
        return df_96

    def compute_loo_attribution_grid(self, patch_tokens: np.ndarray, target_class: int) -> np.ndarray:
        """Computes 14x14 token-level Leave-One-Out (LOO) feature ablation attribution."""
        N = 196
        token_sum = np.sum(patch_tokens, axis=0)  # [768]
        h_base = token_sum / N
        h_base_norm = h_base / (np.linalg.norm(h_base) + 1e-12)
        base_prob = float(self.clf.predict_proba(h_base_norm.reshape(1, -1))[0, target_class])

        loo_deltas = np.zeros(N, dtype=np.float32)
        # Vectorized ablation calculation
        h_ablated = (token_sum - patch_tokens) / (N - 1)  # [196, 768]
        norms = np.linalg.norm(h_ablated, axis=1, keepdims=True) + 1e-12
        h_abl_norm = h_ablated / norms
        abl_probs = self.clf.predict_proba(h_abl_norm)[:, target_class]
        loo_deltas = base_prob - abl_probs  # Positive = token contributed to target class
        return loo_deltas.reshape(14, 14)

    def analyze_sample(
        self,
        image_path: Path,
        true_class_idx: int,
        model: AutoModel,
    ) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
        """Extracts representations, spatial grid, descriptors, and attribution for one image.

        Returns:
            descriptors: dictionary of numerical metrics
            token_norm_grid: (14, 14) token L2 norms
            loo_grid: (14, 14) LOO attribution deltas
            grid_tokens: (14, 14, 768) unnormalized spatial tokens
        """
        tensor = self.preprocessor.preprocess_image(image_path).unsqueeze(0)
        with torch.no_grad():
            outputs = model(tensor)
            last_hidden = outputs.last_hidden_state.squeeze(0).cpu().numpy()  # (197, 768)

        # 1. Separate CLS and Spatial Tokens
        cls_raw = last_hidden[0, :]                       # (768,)
        patch_tokens = last_hidden[1:, :]                 # (196, 768)
        grid_tokens = patch_tokens.reshape(14, 14, 768)   # (14, 14, 768)

        # 2. Representations A & B
        cls_norm = cls_raw / (np.linalg.norm(cls_raw) + 1e-12)
        global_mean_raw = patch_tokens.mean(axis=0)       # (768,)
        global_mean_norm = global_mean_raw / (np.linalg.norm(global_mean_raw) + 1e-12)

        # 3. Representation C: 2x2 Regional Pooling
        regional_raw = {}
        regional_norm = {}
        for q, (r_slice, c_slice) in QUADRANT_SLICES.items():
            q_tokens = grid_tokens[r_slice, c_slice, :].reshape(-1, 768)  # (49, 768)
            q_mean_raw = q_tokens.mean(axis=0)
            regional_raw[q] = q_mean_raw
            regional_norm[q] = q_mean_raw / (np.linalg.norm(q_mean_raw) + 1e-12)

        # Aggregated regional mean (unnormalized average of the 4 quadrants)
        agg_regional_raw = np.mean([regional_raw[q] for q in QUADRANTS], axis=0)
        agg_regional_norm = agg_regional_raw / (np.linalg.norm(agg_regional_raw) + 1e-12)

        # 4. Token Feature Norms on 14x14 Grid
        token_norms = np.linalg.norm(grid_tokens, axis=2)  # (14, 14)
        flat_norms = token_norms.flatten()

        norm_mean = float(np.mean(flat_norms))
        norm_std = float(np.std(flat_norms))
        norm_min = float(np.min(flat_norms))
        norm_max = float(np.max(flat_norms))
        norm_dispersion = float(norm_std / max(norm_mean, 1e-12))

        # 5. Pairwise Cosine Similarities Among 4 Regional Embeddings
        pair_cos = {}
        pair_keys = [("TL", "TR"), ("TL", "BL"), ("TL", "BR"), ("TR", "BL"), ("TR", "BR"), ("BL", "BR")]
        cos_values = []
        l2_dists = []

        for q1, q2 in pair_keys:
            cos_sim = float(np.dot(regional_norm[q1], regional_norm[q2]))
            l2_dist = float(np.linalg.norm(regional_norm[q1] - regional_norm[q2]))
            pair_cos[f"cos_sim_{q1}_{q2}"] = round(cos_sim, 5)
            pair_cos[f"l2_dist_{q1}_{q2}"] = round(l2_dist, 5)
            cos_values.append(cos_sim)
            l2_dists.append(l2_dist)

        regional_cos_mean = float(np.mean(cos_values))
        regional_cos_min = float(np.min(cos_values))
        regional_cos_max = float(np.max(cos_values))
        regional_cos_range = float(regional_cos_max - regional_cos_min)
        regional_l2_mean = float(np.mean(l2_dists))

        # 6. Regional Dispersion (mean squared distance to quadrant center)
        quad_center = np.mean([regional_norm[q] for q in QUADRANTS], axis=0)
        regional_dispersion = float(np.mean([np.sum((regional_norm[q] - quad_center) ** 2) for q in QUADRANTS]))

        # 7. Cross-Representation Similarities
        cos_cls_vs_global_mean = float(np.dot(cls_norm, global_mean_norm))
        cos_global_mean_vs_agg = float(np.dot(global_mean_norm, agg_regional_norm))
        cos_cls_vs_regional = {
            f"cos_sim_cls_{q}": round(float(np.dot(cls_norm, regional_norm[q])), 5) for q in QUADRANTS
        }
        cos_cls_vs_regional_mean = float(np.mean(list(cos_cls_vs_regional.values())))

        # 8. Classification & Attribution
        probs = self.clf.predict_proba(global_mean_norm.reshape(1, -1))[0]
        pred_idx = int(np.argmax(probs))
        pred_conf = float(probs[pred_idx])

        # LOO attribution on the predicted class
        loo_grid = self.compute_loo_attribution_grid(patch_tokens, target_class=pred_idx)

        # Regional attribution mass
        reg_att_mass = {}
        total_att = float(np.sum(np.abs(loo_grid))) + 1e-12
        for q, (r_s, c_s) in QUADRANT_SLICES.items():
            q_att = float(np.sum(loo_grid[r_s, c_s]))
            reg_att_mass[f"att_sum_{q}"] = round(q_att, 6)
            reg_att_mass[f"att_frac_abs_{q}"] = round(float(np.sum(np.abs(loo_grid[r_s, c_s]))) / total_att, 4)

        descriptors = {
            "token_norm_mean": round(norm_mean, 4),
            "token_norm_std": round(norm_std, 4),
            "token_norm_min": round(norm_min, 4),
            "token_norm_max": round(norm_max, 4),
            "token_norm_dispersion": round(norm_dispersion, 5),
            "regional_cos_sim_mean": round(regional_cos_mean, 5),
            "regional_cos_sim_min": round(regional_cos_min, 5),
            "regional_cos_sim_max": round(regional_cos_max, 5),
            "regional_cos_sim_range": round(regional_cos_range, 5),
            "regional_l2_dist_mean": round(regional_l2_mean, 5),
            "regional_dispersion": round(regional_dispersion, 6),
            "cos_sim_cls_vs_global_mean": round(cos_cls_vs_global_mean, 5),
            "cos_sim_global_mean_vs_agg_regional": round(cos_global_mean_vs_agg, 5),
            "cos_sim_cls_vs_regional_mean": round(cos_cls_vs_regional_mean, 5),
            "predicted_class": CANONICAL_CLASSES[pred_idx],
            "predicted_index": pred_idx,
            "confidence": round(pred_conf, 4),
            "is_correct": bool(pred_idx == true_class_idx),
        }
        descriptors.update(pair_cos)
        descriptors.update(cos_cls_vs_regional)
        descriptors.update(reg_att_mass)

        return descriptors, token_norms, loo_grid, grid_tokens

    def run_spatial_analysis(self) -> Dict[str, Any]:
        """Executes full Phase 8 intra-patch spatial analysis on the 96-sample cohort."""
        start_time = time.time()
        arch_info = self.verify_architecture()
        df_96 = self.select_deterministic_cohort(seed=42, n_per_class=12)
        model = self._load_model()

        all_descriptors = []
        exemplars_data = {}  # Store 1 exemplar per class for contact sheet and detailed figures

        logger.info("Extracting spatial token representations for N=96 images...")
        t_start = time.time()

        for i, row in df_96.iterrows():
            img_path = self.dataset_root / row["relative_path"]
            desc, norm_grid, loo_grid, _ = self.analyze_sample(img_path, int(row["class_index"]), model)

            rec = {
                "spatial_sample_id": row["spatial_sample_id"],
                "master_row_index": int(row["row_index"]),
                "relative_path": row["relative_path"],
                "class_name": row["class_name"],
                "class_index": int(row["class_index"]),
            }
            rec.update(desc)
            all_descriptors.append(rec)

            # Store the first exemplar of each class for visual rendering
            c_name = row["class_name"]
            if c_name not in exemplars_data:
                exemplars_data[c_name] = {
                    "row": row,
                    "desc": desc,
                    "norm_grid": norm_grid,
                    "loo_grid": loo_grid,
                    "img_path": img_path,
                }

            if (i + 1) % 24 == 0 or (i + 1) == len(df_96):
                elapsed = time.time() - t_start
                logger.info("  Processed %d / %d samples (%.2f img/s)", i + 1, len(df_96), (i + 1) / max(elapsed, 1e-3))

        total_runtime = time.time() - start_time
        df_desc = pd.DataFrame(all_descriptors)

        # Save descriptors
        csv_path = self.output_dir / "spatial_descriptors.csv"
        df_desc.to_csv(csv_path, index=False)
        logger.info("Saved spatial descriptors CSV: %s (%d rows)", csv_path.name, len(df_desc))

        json_path = self.output_dir / "spatial_descriptors.json"
        with open(json_path, "w") as f:
            json.dump(all_descriptors, f, indent=2)
        logger.info("Saved spatial descriptors JSON: %s", json_path.name)

        # Class-level summary statistics
        class_summary = {}
        key_metrics = [
            "token_norm_mean", "token_norm_std", "token_norm_dispersion",
            "regional_cos_sim_mean", "regional_l2_dist_mean", "regional_dispersion",
            "cos_sim_cls_vs_global_mean", "confidence"
        ]

        for c_idx, c_name in enumerate(CANONICAL_CLASSES):
            df_c = df_desc[df_desc["class_name"] == c_name]
            class_summary[c_name] = {
                "support": len(df_c),
                "accuracy": round(float(df_c["is_correct"].mean()), 4),
            }
            for m in key_metrics:
                class_summary[c_name][f"{m}_mean"] = round(float(df_c[m].mean()), 5)
                class_summary[c_name][f"{m}_std"] = round(float(df_c[m].std()), 5)

        overall_summary = {
            "total_samples": len(df_desc),
            "classes": CANONICAL_CLASSES,
            "overall_accuracy_on_spatial_cohort": round(float(df_desc["is_correct"].mean()), 4),
        }
        for m in key_metrics:
            overall_summary[f"{m}_mean"] = round(float(df_desc[m].mean()), 5)
            overall_summary[f"{m}_std"] = round(float(df_desc[m].std()), 5)

        summary_payload = {
            "metadata": {
                "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
                "n_samples": len(df_desc),
                "samples_per_class": 12,
                "seed": 42,
                "total_runtime_sec": round(total_runtime, 2),
                "model_architecture": arch_info,
            },
            "overall": overall_summary,
            "class_summary": class_summary,
        }

        summary_json_path = self.output_dir / "spatial_summary.json"
        with open(summary_json_path, "w") as f:
            json.dump(summary_payload, f, indent=2)
        logger.info("Saved spatial summary JSON: %s", summary_json_path.name)

        # Save configuration
        config = {
            "phase": "Phase 8: Intra-Patch Spatial Representation Analysis",
            "model_name": "owkin/phikon",
            "architecture": "ViT-B/16 (12 layers, 12 heads, hidden_dim=768, patch_size=16x16)",
            "spatial_grid": "14x14 tokens (196 spatial tokens)",
            "regional_partition": "2x2 coarse quadrants (TL, TR, BL, BR), each 7x7 tokens (49 tokens)",
            "sample_selection": {
                "source_split": "Held-out test split from 10% stratified subset (manifest_subset.csv)",
                "seed": 42,
                "samples_per_class": 12,
                "total_samples": 96,
                "classes": CANONICAL_CLASSES,
            },
            "descriptors_calculated": [
                "token_norm_mean", "token_norm_std", "token_norm_dispersion",
                "regional_pairwise_cosine_similarity", "regional_pairwise_l2_distance",
                "within_image_regional_dispersion", "cos_sim_cls_vs_global_mean",
                "cos_sim_global_mean_vs_agg_regional", "regional_attribution_mass"
            ],
            "executed_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_runtime_sec": round(total_runtime, 2),
        }
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Generate Visualizations
        self.generate_visualizations(df_desc, exemplars_data)

        logger.info("Phase 8 Intra-Patch Spatial Analysis finished in %.1f seconds.", total_runtime)
        return summary_payload

    def generate_visualizations(
        self,
        df_desc: pd.DataFrame,
        exemplars_data: Dict[str, Any],
    ) -> None:
        """Generates contact sheet, regional comparison plot, and representation similarity plot."""
        logger.info("Generating publication-ready spatial visualizations...")

        # -------------------------------------------------------------
        # Plot 1: Contact Sheet (8 Classes x 4 Panels)
        # Columns: Original H&E, 14x14 Token Norm, 14x14 LOO Attribution, 2x2 Regional Grid
        # -------------------------------------------------------------
        fig, axes = plt.subplots(nrows=8, ncols=4, figsize=(16, 26))
        col_titles = [
            "Original H&E Tile\n(224x224 RGB)",
            "Token Feature Norm\n(14x14 Grid, L2 Norm)",
            "LOO Token Attribution\n(14x14 Grid, Delta Prob)",
            "2x2 Regional Pooling\n(4 Quadrants: 7x7 Tokens Each)",
        ]

        for col, title in enumerate(col_titles):
            axes[0, col].set_title(title, fontsize=12, fontweight="bold", pad=10)

        for row_i, c_name in enumerate(CANONICAL_CLASSES):
            ex = exemplars_data[c_name]
            img = Image.open(ex["img_path"]).convert("RGB")
            norm_grid = ex["norm_grid"]
            loo_grid = ex["loo_grid"]
            desc = ex["desc"]

            # Panel 1: Original Image
            ax1 = axes[row_i, 0]
            ax1.imshow(img)
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_ylabel(f"{c_name}\n({ex['row']['relative_path'].split('/')[-1]})", fontsize=10, fontweight="bold")
            status_color = "#2ca02c" if desc["is_correct"] else "#d62728"
            ax1.set_xlabel(f"Pred: {desc['predicted_class']} ({desc['confidence']:.2f})", fontsize=10, color=status_color, fontweight="semibold")

            # Panel 2: 14x14 Token Norm Heatmap
            ax2 = axes[row_i, 1]
            im2 = ax2.imshow(norm_grid, cmap="viridis", interpolation="nearest")
            ax2.set_xticks(np.arange(0, 14, 2))
            ax2.set_yticks(np.arange(0, 14, 2))
            ax2.set_xticklabels(np.arange(0, 14, 2), fontsize=7)
            ax2.set_yticklabels(np.arange(0, 14, 2), fontsize=7)
            ax2.set_xlabel(f"Norm: {desc['token_norm_mean']:.2f} ± {desc['token_norm_std']:.2f}", fontsize=9)
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

            # Panel 3: 14x14 LOO Attribution Heatmap
            ax3 = axes[row_i, 2]
            vmax = max(abs(float(np.min(loo_grid))), abs(float(np.max(loo_grid))), 1e-4)
            im3 = ax3.imshow(loo_grid, cmap="seismic", vmin=-vmax, vmax=vmax, interpolation="nearest")
            ax3.set_xticks(np.arange(0, 14, 2))
            ax3.set_yticks(np.arange(0, 14, 2))
            ax3.set_xticklabels(np.arange(0, 14, 2), fontsize=7)
            ax3.set_yticklabels(np.arange(0, 14, 2), fontsize=7)
            ax3.set_xlabel("Saliency Mass (Red=High)", fontsize=9)
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

            # Panel 4: 2x2 Regional Pooling Overlaid on Image
            ax4 = axes[row_i, 3]
            ax4.imshow(img)
            # Draw quadrant cross lines at pixel coordinate 112 (7 * 16)
            ax4.axvline(112, color="yellow", linestyle="--", linewidth=1.5, alpha=0.9)
            ax4.axhline(112, color="yellow", linestyle="--", linewidth=1.5, alpha=0.9)
            ax4.set_xticks([])
            ax4.set_yticks([])

            # Annotate quadrant labels and mean cosine similarity
            ax4.text(28, 28, f"TL\n{desc['att_frac_abs_TL']*100:.0f}%", color="white", fontsize=8.5, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))
            ax4.text(140, 28, f"TR\n{desc['att_frac_abs_TR']*100:.0f}%", color="white", fontsize=8.5, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))
            ax4.text(28, 140, f"BL\n{desc['att_frac_abs_BL']*100:.0f}%", color="white", fontsize=8.5, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))
            ax4.text(140, 140, f"BR\n{desc['att_frac_abs_BR']*100:.0f}%", color="white", fontsize=8.5, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))

            ax4.set_xlabel(f"Quad CosSim: {desc['regional_cos_sim_mean']:.3f}", fontsize=9, fontweight="semibold")

        plt.suptitle("GastricPath AI — Phase 8: Intra-Patch Spatial Token & Regional Representation Gallery (N=96)",
                     fontsize=15, fontweight="bold", y=0.995)
        plt.tight_layout()
        contact_path = self.output_dir / "spatial_analysis_contact_sheet.png"
        plt.savefig(contact_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info("Saved spatial contact sheet: %s", contact_path.name)

        # -------------------------------------------------------------
        # Plot 2: Regional Cosine Similarity and Token Dispersion by Class
        # -------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Subplot A: Regional Cosine Similarity (Boxplot/Mean)
        ax = axes[0]
        data_cos = [df_desc[df_desc["class_name"] == c]["regional_cos_sim_mean"] for c in CANONICAL_CLASSES]
        bplot1 = ax.boxplot(data_cos, labels=CANONICAL_CLASSES, patch_artist=True, medianprops=dict(color="black", linewidth=1.5))
        colors = plt.cm.tab10(np.linspace(0, 1, 8))
        for patch, color in zip(bplot1["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel("Pairwise Regional Cosine Similarity", fontsize=11, fontweight="bold")
        ax.set_title("Panel A: 2x2 Regional Embedding Homogeneity (Pairwise CosSim)", fontsize=12, fontweight="bold")
        ax.set_ylim(0.70, 1.0)
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        # Subplot B: Token Norm Dispersion (std/mean)
        ax = axes[1]
        data_disp = [df_desc[df_desc["class_name"] == c]["token_norm_dispersion"] for c in CANONICAL_CLASSES]
        bplot2 = ax.boxplot(data_disp, labels=CANONICAL_CLASSES, patch_artist=True, medianprops=dict(color="black", linewidth=1.5))
        for patch, color in zip(bplot2["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel("Token Feature Norm Dispersion (std / mean)", fontsize=11, fontweight="bold")
        ax.set_title("Panel B: Intra-Patch Feature Norm Heterogeneity", fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        plt.suptitle("GastricPath AI — Phase 8: Within-Patch Spatial Descriptors Across 8 Tissue Classes",
                     fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plot2_path = self.output_dir / "regional_similarity_by_class.png"
        plt.savefig(plot2_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info("Saved regional similarity plot: %s", plot2_path.name)

        # -------------------------------------------------------------
        # Plot 3: Representation Similarity (CLS vs. Global Mean Patch)
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 6))
        data_cls_mean = [df_desc[df_desc["class_name"] == c]["cos_sim_cls_vs_global_mean"] for c in CANONICAL_CLASSES]
        bplot3 = ax.boxplot(data_cls_mean, labels=CANONICAL_CLASSES, patch_artist=True, medianprops=dict(color="red", linewidth=2))
        for patch, color in zip(bplot3["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)

        ax.axhline(df_desc["cos_sim_cls_vs_global_mean"].mean(), color="navy", linestyle="--", linewidth=1.5,
                   label=f"Overall Mean CosSim: {df_desc['cos_sim_cls_vs_global_mean'].mean():.3f}")
        ax.set_ylabel("Cosine Similarity (CLS vs. Global Mean Patch)", fontsize=11, fontweight="bold")
        ax.set_title("GastricPath AI — Phase 8: Alignment Between CLS Token and Mean Patch Embeddings", fontsize=13, fontweight="bold")
        ax.set_ylim(0.70, 0.98)
        ax.legend(loc="lower left", frameon=True)
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        plt.tight_layout()
        plot3_path = self.output_dir / "representation_similarity_plot.png"
        plt.savefig(plot3_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info("Saved representation similarity plot: %s", plot3_path.name)


if __name__ == "__main__":
    pipeline = SpatialAnalysisPipeline()
    pipeline.run_spatial_analysis()

