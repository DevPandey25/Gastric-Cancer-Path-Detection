"""Original Phase 5: CNN-RNN Explainability and Uncertainty Attribution Engine.

Implements model-supported interpretability and uncertainty for the historical baseline
GastricCancerRNN model (CNN-BiLSTM-Attention, models/checkpoints/gastric_rnn_classifier.pt):
1. Grad-CAM on the final convolutional layer (patch_encoder.encoder[8]).
2. Integrated Gradients across the patch sequence (m=20 Riemann steps).
3. Occlusion Sensitivity (28x28 sliding window, stride 28).
4. Monte Carlo (MC) Dropout uncertainty (T=20 stochastic forward passes).
5. Visualizations and JSON summary across 8 representative test tiles (1 per class).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

from app.core.config import settings
from app.ml.model import GastricCancerRNN
from app.ml.preprocessing import image_to_patch_sequence, load_rgb_image

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("original_phase5_cnn_attribution")

CANONICAL_CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]


class CNNAttributionEngine:
    """Explainability, gradient attribution, and MC Dropout for GastricCancerRNN."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self.workspace_root = Path(workspace_root or WORKSPACE_ROOT).resolve()
        self.output_dir = self.workspace_root / "research" / "explainability" / "original_phase5"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cpu")
        self.ckpt_path = self.workspace_root / "models" / "checkpoints" / "gastric_rnn_classifier.pt"
        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"Missing baseline checkpoint: {self.ckpt_path}")

        self.model = GastricCancerRNN(num_classes=len(CANONICAL_CLASSES)).to(self.device)
        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.val_accuracy = float(ckpt.get("val_accuracy", 0.0))
        logger.info("Loaded GastricCancerRNN (val_acc=%.4f) on CPU", self.val_accuracy)

        # Target layer for Grad-CAM: patch_encoder.encoder[8] (Conv2d 64->128)
        self.target_conv_layer = self.model.patch_encoder.encoder[8]

    def compute_gradcam(
        self, sequence: torch.Tensor, target_class: int
    ) -> Tuple[np.ndarray, np.ndarray, int, float]:
        """Computes Grad-CAM heatmap on patch_encoder.encoder[8] tiled across 16 patches."""
        activations: List[torch.Tensor] = []
        gradients: List[torch.Tensor] = []

        def forward_hook(module, inp, out):
            activations.append(out)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        h_f = self.target_conv_layer.register_forward_hook(forward_hook)
        h_b = self.target_conv_layer.register_full_backward_hook(backward_hook)

        self.model.zero_grad()
        logits, attention = self.model(sequence)
        pred_class = int(logits.argmax(dim=1).item())
        probs = F.softmax(logits, dim=1).squeeze(0).detach().numpy()
        pred_conf = float(probs[pred_class])

        target = target_class if target_class is not None else pred_class
        logits[0, target].backward()

        h_f.remove()
        h_b.remove()

        # Acts and grads shape: (16, 128, 14, 14)
        acts = activations[0].detach()
        grads = gradients[0].detach()

        # Global average pool gradients across spatial dims (14, 14) -> (16, 128)
        weights = grads.mean(dim=(2, 3), keepdim=True)
        # Weighted linear combination
        cam_patches = (weights * acts).sum(dim=1)  # (16, 14, 14)
        cam_patches = F.relu(cam_patches)

        # Assemble the 16 patches (4x4 grid) into 224x224
        # Each patch is 56x56
        assembled_map = np.zeros((224, 224), dtype=np.float32)
        idx = 0
        for r in range(4):
            for c in range(4):
                patch_cam = cam_patches[idx].numpy()
                idx += 1
                # Resize 14x14 patch cam to 56x56
                patch_cam_resized = cv2.resize(patch_cam, (56, 56), interpolation=cv2.INTER_LINEAR)
                assembled_map[r * 56 : (r + 1) * 56, c * 56 : (c + 1) * 56] = patch_cam_resized

        # Normalize to [0, 1]
        max_v = float(np.max(assembled_map))
        if max_v > 1e-12:
            assembled_map /= max_v

        attention_weights = attention.squeeze(0).detach().numpy()  # (16,)
        return assembled_map, attention_weights, pred_class, pred_conf

    def compute_integrated_gradients(
        self, sequence: torch.Tensor, target_class: int, m_steps: int = 20
    ) -> np.ndarray:
        """Computes Integrated Gradients for the input patch sequence."""
        baseline = torch.zeros_like(sequence)
        grads: List[torch.Tensor] = []

        for k in range(1, m_steps + 1):
            alpha = k / m_steps
            interpolated = baseline + alpha * (sequence - baseline)
            interpolated.requires_grad_(True)
            self.model.zero_grad()
            logits, _ = self.model(interpolated)
            logits[0, target_class].backward()
            grads.append(interpolated.grad.detach())

        avg_grads = torch.stack(grads, dim=0).mean(dim=0)  # (1, 16, 3, 56, 56)
        ig = (sequence - baseline) * avg_grads
        # Magnitude across color channels -> (16, 56, 56)
        ig_mag = torch.norm(ig.squeeze(0), dim=1).numpy()

        # Assemble 16 patches into 224x224
        assembled_ig = np.zeros((224, 224), dtype=np.float32)
        idx = 0
        for r in range(4):
            for c in range(4):
                assembled_ig[r * 56 : (r + 1) * 56, c * 56 : (c + 1) * 56] = ig_mag[idx]
                idx += 1

        max_v = float(np.max(assembled_ig))
        if max_v > 1e-12:
            assembled_ig /= max_v
        return assembled_ig

    def compute_occlusion_sensitivity(
        self, image: Image.Image, target_class: int, occ_size: int = 28, stride: int = 28
    ) -> np.ndarray:
        """Computes Occlusion Sensitivity map with a sliding gray occluder."""
        orig_arr = np.array(image, dtype=np.float32)
        base_seq = (
            image_to_patch_sequence(image, settings.patch_size, settings.sequence_length)
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            base_logits, _ = self.model(base_seq)
            base_prob = float(F.softmax(base_logits, dim=1)[0, target_class].item())

        h, w = 224, 224
        n_rows = len(range(0, h, stride))
        n_cols = len(range(0, w, stride))
        occ_grid = np.zeros((n_rows, n_cols), dtype=np.float32)

        for i, y in enumerate(range(0, h, stride)):
            for j, x in enumerate(range(0, w, stride)):
                occ_arr = orig_arr.copy()
                occ_arr[y : y + occ_size, x : x + occ_size] = 128.0  # Neutral gray
                occ_img = Image.fromarray(occ_arr.astype(np.uint8))
                seq = (
                    image_to_patch_sequence(occ_img, settings.patch_size, settings.sequence_length)
                    .unsqueeze(0)
                    .to(self.device)
                )
                with torch.no_grad():
                    logits, _ = self.model(seq)
                    prob = float(F.softmax(logits, dim=1)[0, target_class].item())
                # Drop in probability
                occ_grid[i, j] = max(0.0, base_prob - prob)

        # Upsample 8x8 drop map to 224x224
        occ_map = cv2.resize(occ_grid, (224, 224), interpolation=cv2.INTER_CUBIC)
        occ_map = np.clip(occ_map, 0.0, None)
        max_v = float(np.max(occ_map))
        if max_v > 1e-12:
            occ_map /= max_v
        return occ_map

    def compute_mc_dropout(
        self, sequence: torch.Tensor, t_passes: int = 20
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Computes Monte Carlo Dropout predictive mean and epistemic variance."""
        # Enable dropout during inference
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

        probs_list: List[np.ndarray] = []
        att_list: List[np.ndarray] = []

        with torch.no_grad():
            for _ in range(t_passes):
                logits, attention = self.model(sequence)
                probs = F.softmax(logits, dim=1).squeeze(0).numpy()
                probs_list.append(probs)
                att_list.append(attention.squeeze(0).numpy())

        # Reset model to eval
        self.model.eval()

        probs_arr = np.stack(probs_list, axis=0)  # (T, 8)
        mean_probs = np.mean(probs_arr, axis=0)  # (8,)
        var_probs = np.var(probs_arr, axis=0)  # (8,)
        pred_class = int(np.argmax(mean_probs))
        epistemic_variance = float(var_probs[pred_class])
        total_epistemic_entropy = float(-np.sum(mean_probs * np.log(mean_probs + 1e-12)))

        att_arr = np.stack(att_list, axis=0)  # (T, 16)
        mean_att = np.mean(att_arr, axis=0)  # (16,)
        return mean_probs, mean_att, epistemic_variance, total_epistemic_entropy

    def run_attribution_cohort(self) -> Dict[str, Any]:
        """Runs Grad-CAM, Integrated Gradients, Occlusion, and MC Dropout on 8 class exemplars."""
        logger.info("Executing CNN attribution engine across 8 canonical class exemplars...")
        manifest_path = (
            self.workspace_root
            / "research"
            / "foundation_models"
            / "cache"
            / "subset10pct"
            / "manifest_subset.csv"
        )
        df_manifest = pd.read_csv(manifest_path)
        test_df = df_manifest[df_manifest["split"] == "test"].copy()

        # Select 1 exemplar per class deterministically (seed 42)
        exemplars: List[Dict[str, Any]] = []
        for c in CANONICAL_CLASSES:
            sub = test_df[test_df["class_name"] == c].sort_values("relative_path")
            ex_row = sub.iloc[0]  # First alphabetical sample per class
            exemplars.append(ex_row.to_dict())

        results = []
        fig, axes = plt.subplots(nrows=8, ncols=6, figsize=(20, 26))
        col_headers = [
            "Original H&E Tile\n(224x224 RGB)",
            "Sequence Attention Grid\n(16 Patches, 4x4)",
            "Grad-CAM Heatmap\n(Conv Layer 8)",
            "Integrated Gradients\n(m=20 Steps)",
            "Occlusion Sensitivity\n(28x28 Sliding Box)",
            "MC Dropout Uncertainty\n(T=20 Passes)",
        ]
        for col_idx, h in enumerate(col_headers):
            axes[0, col_idx].set_title(h, fontsize=11, fontweight="bold", pad=8)

        for row_idx, ex in enumerate(exemplars):
            c_name = ex["class_name"]
            rel_path = ex["relative_path"]
            img_path = self.workspace_root / "data" / "raw" / "HMU-GC-HE-30K" / "all_image" / rel_path

            img = load_rgb_image(img_path, settings.image_size)
            seq = (
                image_to_patch_sequence(img, settings.patch_size, settings.sequence_length)
                .unsqueeze(0)
                .to(self.device)
            )

            # 1. Grad-CAM
            cam_map, att_weights, pred_c, pred_conf = self.compute_gradcam(seq, target_class=int(ex["class_index"]))

            # 2. Integrated Gradients
            ig_map = self.compute_integrated_gradients(seq, target_class=int(ex["class_index"]))

            # 3. Occlusion Sensitivity
            occ_map = self.compute_occlusion_sensitivity(img, target_class=int(ex["class_index"]))

            # 4. MC Dropout
            mean_probs, mean_att, epistemic_var, epistemic_entropy = self.compute_mc_dropout(seq, t_passes=20)

            rec = {
                "class_name": c_name,
                "relative_path": rel_path,
                "predicted_class": CANONICAL_CLASSES[pred_c],
                "confidence": round(pred_conf, 4),
                "is_correct": bool(pred_c == int(ex["class_index"])),
                "mc_epistemic_variance": round(epistemic_var, 6),
                "mc_epistemic_entropy": round(epistemic_entropy, 4),
                "max_gradcam_activation": round(float(np.max(cam_map)), 4),
                "mean_gradcam_activation": round(float(np.mean(cam_map)), 4),
            }
            results.append(rec)

            # Render row panels
            # Panel 1: Original Image
            ax1 = axes[row_idx, 0]
            ax1.imshow(img)
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_ylabel(f"{c_name}\n({rel_path.split('/')[-1]})", fontsize=10, fontweight="bold")
            color = "#2ca02c" if rec["is_correct"] else "#d62728"
            ax1.set_xlabel(f"Pred: {rec['predicted_class']} ({pred_conf:.2f})", fontsize=9, color=color, fontweight="semibold")

            # Panel 2: Attention Grid
            ax2 = axes[row_idx, 1]
            att_grid = att_weights.reshape(4, 4)
            im2 = ax2.imshow(att_grid, cmap="YlOrRd", interpolation="nearest")
            ax2.set_xticks([])
            ax2.set_yticks([])
            for r in range(4):
                for c in range(4):
                    ax2.text(c, r, f"{att_grid[r, c]:.2f}", ha="center", va="center", fontsize=8, color="black")
            ax2.set_xlabel(f"Max Att: {att_weights.max():.2f}", fontsize=8.5)

            # Panel 3: Grad-CAM
            ax3 = axes[row_idx, 2]
            ax3.imshow(img)
            im3 = ax3.imshow(cam_map, cmap="jet", alpha=0.45)
            ax3.set_xticks([])
            ax3.set_yticks([])
            ax3.set_xlabel("Focal Saliency", fontsize=8.5)

            # Panel 4: Integrated Gradients
            ax4 = axes[row_idx, 3]
            im4 = ax4.imshow(ig_map, cmap="magma", interpolation="nearest")
            ax4.set_xticks([])
            ax4.set_yticks([])
            ax4.set_xlabel("Input Attribution", fontsize=8.5)

            # Panel 5: Occlusion Sensitivity
            ax5 = axes[row_idx, 4]
            ax5.imshow(img)
            im5 = ax5.imshow(occ_map, cmap="hot", alpha=0.5)
            ax5.set_xticks([])
            ax5.set_yticks([])
            ax5.set_xlabel("Drop Magnitude", fontsize=8.5)

            # Panel 6: MC Dropout Predictive Uncertainty
            ax6 = axes[row_idx, 5]
            ax6.bar(np.arange(8), mean_probs, color="#3182bd", alpha=0.7, edgecolor="black")
            ax6.set_xticks(np.arange(8))
            ax6.set_xticklabels(CANONICAL_CLASSES, fontsize=6.5, rotation=45)
            ax6.set_ylim(0, 1.05)
            ax6.set_xlabel(f"Var: {epistemic_var:.4f} | H: {epistemic_entropy:.2f}", fontsize=8)

        plt.suptitle(
            "GastricPath AI — Original Phase 5: Baseline CNN-RNN Explainability & Uncertainty Gallery",
            fontsize=14,
            fontweight="bold",
            y=0.995,
        )
        plt.tight_layout()
        viz_path = self.output_dir / "cnn_explainability_comparison.png"
        plt.savefig(viz_path, dpi=180, bbox_inches="tight")
        plt.close()
        logger.info("Saved %s", viz_path.name)

        summary = {
            "model_name": "GastricCancerRNN (CNN-BiLSTM-Attention)",
            "checkpoint_path": "models/checkpoints/gastric_rnn_classifier.pt",
            "val_accuracy": self.val_accuracy,
            "target_conv_layer": "patch_encoder.encoder[8] (Conv2d 64->128)",
            "gradcam_method": "Channel-pooled gradient weighting across 16 sequential patches",
            "integrated_gradients_steps": 20,
            "occlusion_window_size": 28,
            "occlusion_stride": 28,
            "mc_dropout_passes": 20,
            "exemplar_results": results,
        }
        with open(self.output_dir / "cnn_attribution_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved cnn_attribution_summary.json")
        return summary


if __name__ == "__main__":
    engine = CNNAttributionEngine()
    engine.run_attribution_cohort()

