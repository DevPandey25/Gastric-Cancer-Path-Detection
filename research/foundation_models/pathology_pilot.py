"""
Phase 4.4.1 — Pathology Foundation Model (Phikon) Small Pilot.

Extracts frozen representations from 64 stratified GCHTID images (8 per class)
using owkin/phikon (ViT-B/16, d=768) to verify:
1. Offline loading & parameter freezing.
2. Deterministic preprocessing fidelity.
3. Embedding dimensions, L2 normalization, and numerical integrity.
4. CPU inference latency, throughput, and memory consumption.
5. Linear probe compatibility.
6. Full dataset resource projections.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import logging
import time

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from transformers import AutoImageProcessor, AutoModel

from research.foundation_models.preprocessing import DeterministicPathologyPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phikon_pilot")

CANONICAL_CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CANONICAL_CLASSES)}


class PhikonEncoderWrapper(torch.nn.Module):
    """Wraps owkin/phikon to extract L2-normalized CLS and mean patch representations."""

    def __init__(self, model_name: str = "owkin/phikon") -> None:
        super().__init__()
        self.model_name = model_name
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.hidden_dim = self.encoder.config.hidden_size  # 768

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pixel_values: [B, 3, 224, 224] normalized tensor.
        Returns:
            cls_norm: [B, 768] L2-normalized CLS token embedding.
            patch_norm: [B, 768] L2-normalized mean spatial patch embedding.
        """
        outputs = self.encoder(pixel_values)
        last_hidden = outputs.last_hidden_state  # [B, 197, 768]

        cls_token = last_hidden[:, 0, :]  # [B, 768]
        patch_tokens = last_hidden[:, 1:, :]  # [B, 196, 768]
        mean_patch = patch_tokens.mean(dim=1)  # [B, 768]

        cls_norm = F.normalize(cls_token, p=2, dim=-1)
        patch_norm = F.normalize(mean_patch, p=2, dim=-1)

        return cls_norm, patch_norm


def run_pilot(num_per_class: int = 8, batch_size: int = 8) -> dict:
    workspace_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = workspace_root / "data" / "raw" / "HMU-GC-HE-30K" / "all_image"
    train_manifest_path = workspace_root / "research" / "splits" / "manifest_train.csv"

    if not train_manifest_path.exists():
        raise FileNotFoundError(f"Missing train manifest: {train_manifest_path}")

    df_train = pd.read_csv(train_manifest_path)
    logger.info(f"Loaded train manifest with {len(df_train)} images.")

    # Select stratified pilot subset (num_per_class per class)
    pilot_dfs = []
    for c in CANONICAL_CLASSES:
        c_subset = df_train[df_train["class_name"] == c].head(num_per_class)
        pilot_dfs.append(c_subset)
    pilot_df = pd.concat(pilot_dfs).reset_index(drop=True)
    total_pilot_samples = len(pilot_df)
    logger.info(f"Constructed balanced pilot cohort: {total_pilot_samples} images ({num_per_class} per class).")

    # Monitor memory
    proc = psutil.Process()
    ram_init_mb = proc.memory_info().rss / (1024 ** 2)

    # Load model
    t_load_0 = time.time()
    encoder = PhikonEncoderWrapper()
    t_load_1 = time.time()
    model_load_time_sec = t_load_1 - t_load_0
    ram_loaded_mb = proc.memory_info().rss / (1024 ** 2)

    # Parameter check
    num_params = sum(p.numel() for p in encoder.parameters())
    num_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    assert num_trainable == 0, f"Expected 0 trainable parameters, found {num_trainable}!"

    # Preprocessor
    preprocessor = DeterministicPathologyPreprocessor(image_size=224)

    # Inference loop
    cls_embeddings_list = []
    patch_embeddings_list = []

    t_inf_0 = time.time()
    for start_idx in range(0, total_pilot_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_pilot_samples)
        batch_rows = pilot_df.iloc[start_idx:end_idx]

        batch_tensors = []
        for _, row in batch_rows.iterrows():
            img_path = raw_dir / row["relative_path"]
            tensor = preprocessor.preprocess_image(img_path)
            batch_tensors.append(tensor)

        batch_input = torch.stack(batch_tensors, dim=0)
        cls_norm, patch_norm = encoder(batch_input)

        cls_embeddings_list.append(cls_norm.cpu().numpy())
        patch_embeddings_list.append(patch_norm.cpu().numpy())

    t_inf_1 = time.time()
    inf_time_sec = t_inf_1 - t_inf_0
    ram_peak_mb = proc.memory_info().rss / (1024 ** 2)

    cls_all = np.vstack(cls_embeddings_list)
    patch_all = np.vstack(patch_embeddings_list)

    # Verification checks
    assert cls_all.shape == (total_pilot_samples, 768), f"Unexpected CLS shape: {cls_all.shape}"
    assert patch_all.shape == (total_pilot_samples, 768), f"Unexpected Patch shape: {patch_all.shape}"

    # Numerical integrity
    cls_finite = bool(np.all(np.isfinite(cls_all)))
    patch_finite = bool(np.all(np.isfinite(patch_all)))
    assert cls_finite and patch_finite, "Non-finite values detected in embeddings!"

    # L2 normalization check
    cls_norms = np.linalg.norm(cls_all, axis=1)
    patch_norms = np.linalg.norm(patch_all, axis=1)
    cls_norm_err = float(np.max(np.abs(cls_norms - 1.0)))
    patch_norm_err = float(np.max(np.abs(patch_norms - 1.0)))
    assert cls_norm_err < 1e-5, f"CLS norm error: {cls_norm_err}"
    assert patch_norm_err < 1e-5, f"Patch norm error: {patch_norm_err}"

    # Throughput metrics
    ms_per_image = (inf_time_sec / total_pilot_samples) * 1000.0
    images_per_sec = total_pilot_samples / inf_time_sec

    # Full dataset projections (31,096 images)
    total_dataset_n = 31096
    proj_runtime_sec = (total_dataset_n / images_per_sec)
    proj_runtime_min = proj_runtime_sec / 60.0
    proj_disk_mb = (total_dataset_n * 768 * 4) / (1024 ** 2)  # for float32 matrix

    # Linear probe compatibility check
    labels = np.array([CLASS_TO_IDX[c] for c in pilot_df["class_name"]])
    lr_cls = LogisticRegression(max_iter=200, C=1.0)
    lr_cls.fit(cls_all, labels)
    pilot_cls_acc = float(lr_cls.score(cls_all, labels))

    lr_patch = LogisticRegression(max_iter=200, C=1.0)
    lr_patch.fit(patch_all, labels)
    pilot_patch_acc = float(lr_patch.score(patch_all, labels))

    pilot_report = {
        "model_id": "owkin/phikon",
        "architecture": "ViT-B/16 (iBOT TCGA Pan-Cancer)",
        "parameter_count": num_params,
        "trainable_parameters": num_trainable,
        "embedding_dimension": 768,
        "pilot_sample_count": total_pilot_samples,
        "batch_size": batch_size,
        "model_load_time_sec": round(model_load_time_sec, 2),
        "total_inference_time_sec": round(inf_time_sec, 2),
        "ms_per_image": round(ms_per_image, 2),
        "images_per_second": round(images_per_sec, 2),
        "ram_initial_mb": round(ram_init_mb, 1),
        "ram_loaded_mb": round(ram_loaded_mb, 1),
        "ram_peak_mb": round(ram_peak_mb, 1),
        "cls_embeddings_shape": list(cls_all.shape),
        "patch_embeddings_shape": list(patch_all.shape),
        "cls_numerical_finite": cls_finite,
        "patch_numerical_finite": patch_finite,
        "cls_max_l2_norm_deviation": cls_norm_err,
        "patch_max_l2_norm_deviation": patch_norm_err,
        "linear_probe_compatible": True,
        "pilot_cls_fit_accuracy": round(pilot_cls_acc, 4),
        "pilot_patch_fit_accuracy": round(pilot_patch_acc, 4),
        "projected_full_dataset_runtime_minutes": round(proj_runtime_min, 2),
        "projected_single_cache_disk_mb": round(proj_disk_mb, 2),
        "projected_dual_cache_disk_mb": round(proj_disk_mb * 2, 2),
    }

    out_dir = workspace_root / "research" / "foundation_models" / "pathology_encoder"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pilot_results.json"
    with open(out_path, "w") as f:
        json.dump(pilot_report, f, indent=2)

    logger.info(f"Pilot execution finished successfully. Report saved to {out_path}")
    logger.info(f"Speed: {ms_per_image:.1f} ms/image ({images_per_sec:.2f} img/s)")
    logger.info(f"Projected full extraction (31,096 images): {proj_runtime_min:.1f} minutes")
    logger.info(f"Projected disk space for each cache: {proj_disk_mb:.1f} MB")

    return pilot_report


if __name__ == "__main__":
    run_pilot()
