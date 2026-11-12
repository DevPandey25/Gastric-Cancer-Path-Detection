"""Reproducible 10% Stratified Subset Extractor & Cache Builder (Phase 4.4 Reduced).

Constructs a reproducible 10% stratified subset (3,110 images) across all 8 classes
and canonical splits (Train: 2,177, Val: 466, Test: 467) using random_state=42.

Features:
1. Manifest generation: research/foundation_models/cache/subset10pct/manifest_subset.csv.
2. DINOv2 subset cache extraction: Retrieves exact matching rows from full DINOv2 cache.
3. Phikon subset cache extraction: Reuses the 1,045 already-extracted full-cache rows,
   and extracts only the remaining 2,065 images using frozen owkin/phikon on CPU (10 threads).
4. Numerical validation: Finite, non-zero, L2-normalized == 1.0.
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

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel

from research.foundation_models.preprocessing import DeterministicPathologyPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("subset_extractor")

CANONICAL_CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]
PHIKON_WEIGHT_SHA256 = "d01642bfeb683c6b039622eca273e0a8e4208065a4595b63edba178c1c4cfa68"
TARGET_COUNTS = {"train": 2177, "val": 466, "test": 467}  # Total: 3,110 images (10% of 31,096)


def create_stratified_subset_manifest(workspace_root: Path, seed: int = 42) -> pd.DataFrame:
    """Selects 3,110 stratified images across 8 classes and 3 splits."""
    full_manifest_path = workspace_root / "research" / "foundation_models" / "cache" / "full" / "dinov2_vits14" / "manifest.csv"
    if not full_manifest_path.exists():
        full_manifest_path = workspace_root / "research" / "manifests" / "manifest_all.csv"

    df_full = pd.read_csv(full_manifest_path)
    logger.info("Loaded master manifest: %d images.", len(df_full))

    rng = np.random.RandomState(seed)
    selected_indices = []

    for split_name, split_target in TARGET_COUNTS.items():
        df_split = df_full[df_full["split"] == split_name]
        base_n = split_target // len(CANONICAL_CLASSES)
        rem = split_target % len(CANONICAL_CLASSES)
        class_targets = {c: base_n + (1 if i < rem else 0) for i, c in enumerate(CANONICAL_CLASSES)}

        for c in CANONICAL_CLASSES:
            df_c = df_split[df_split["class_name"] == c]
            n_sample = class_targets[c]
            sampled = df_c.sample(n=n_sample, random_state=rng)
            selected_indices.extend(sampled.index.tolist())

    selected_indices.sort()
    df_subset = df_full.loc[selected_indices].copy().reset_index(drop=True)
    df_subset["subset_index"] = np.arange(len(df_subset))
    logger.info("Constructed 10%% stratified subset: %d images (%s)", len(df_subset), df_subset["split"].value_counts().to_dict())
    return df_subset


class PhikonEncoderWrapper(torch.nn.Module):
    """Wraps owkin/phikon to extract L2-normalized CLS and mean patch representations."""

    def __init__(self, model_name: str = "owkin/phikon") -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.hidden_dim = self.encoder.config.hidden_size  # 768

    @torch.inference_mode()
    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(pixel_values)
        last_hidden = outputs.last_hidden_state  # [B, 197, 768]
        cls_token = last_hidden[:, 0, :]  # [B, 768]
        patch_tokens = last_hidden[:, 1:, :]  # [B, 196, 768]
        mean_patch = patch_tokens.mean(dim=1)  # [B, 768]

        cls_norm = F.normalize(cls_token, p=2, dim=-1)
        patch_norm = F.normalize(mean_patch, p=2, dim=-1)
        return cls_norm, patch_norm


def build_subset_caches(workspace_root: Path | None = None) -> dict[str, Any]:
    start_time = time.time()
    if workspace_root is None:
        workspace_root = Path(__file__).resolve().parent.parent.parent.parent
    else:
        workspace_root = Path(workspace_root).resolve()

    dataset_root = workspace_root / "data" / "raw" / "HMU-GC-HE-30K" / "all_image"
    subset_dir = workspace_root / "research" / "foundation_models" / "cache" / "subset10pct"
    subset_dir.mkdir(parents=True, exist_ok=True)

    phikon_subset_dir = subset_dir / "phikon"
    dinov2_subset_dir = subset_dir / "dinov2_vits14"
    phikon_subset_dir.mkdir(parents=True, exist_ok=True)
    dinov2_subset_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create subset manifest
    df_subset = create_stratified_subset_manifest(workspace_root, seed=42)
    manifest_subset_path = subset_dir / "manifest_subset.csv"
    df_subset.to_csv(manifest_subset_path, index=False)
    logger.info("Saved subset manifest to %s", manifest_subset_path.name)

    n_subset = len(df_subset)
    assert n_subset == 3110

    # 2. Extract DINOv2 subset cache from full cache
    logger.info("Populating DINOv2 subset cache from full DINOv2 cache...")
    dino_full_dir = workspace_root / "research" / "foundation_models" / "cache" / "full" / "dinov2_vits14"
    dino_full_cls = np.memmap(dino_full_dir / "cls_embeddings.npy", dtype=np.float32, mode="r", shape=(31096, 384))
    dino_full_patch = np.memmap(dino_full_dir / "mean_patch_embeddings.npy", dtype=np.float32, mode="r", shape=(31096, 384))

    dino_sub_cls = np.memmap(dinov2_subset_dir / "cls_embeddings.npy", dtype=np.float32, mode="w+", shape=(n_subset, 384))
    dino_sub_patch = np.memmap(dinov2_subset_dir / "mean_patch_embeddings.npy", dtype=np.float32, mode="w+", shape=(n_subset, 384))

    row_indices = df_subset["row_index"].to_numpy()
    dino_sub_cls[:] = dino_full_cls[row_indices]
    dino_sub_patch[:] = dino_full_patch[row_indices]
    dino_sub_cls.flush()
    dino_sub_patch.flush()

    df_subset.to_csv(dinov2_subset_dir / "manifest.csv", index=False)
    logger.info("DINOv2 subset cache populated (%d samples, shape [%d, 384]).", n_subset, n_subset)

    # 3. Populate Phikon subset cache
    logger.info("Populating Phikon subset cache (reusing partial full cache + extracting missing)...")
    phikon_full_dir = workspace_root / "research" / "foundation_models" / "cache" / "full" / "phikon"
    phikon_full_cls = np.memmap(phikon_full_dir / "cls_embeddings.npy", dtype=np.float32, mode="r", shape=(31096, 768))
    phikon_full_patch = np.memmap(phikon_full_dir / "mean_patch_embeddings.npy", dtype=np.float32, mode="r", shape=(31096, 768))

    phi_sub_cls = np.memmap(phikon_subset_dir / "cls_embeddings.npy", dtype=np.float32, mode="w+", shape=(n_subset, 768))
    phi_sub_patch = np.memmap(phikon_subset_dir / "mean_patch_embeddings.npy", dtype=np.float32, mode="w+", shape=(n_subset, 768))

    # Read checkpoint to see how many were extracted in full cache
    ckpt_path = phikon_full_dir / "checkpoint.json"
    completed_in_full = 0
    if ckpt_path.exists():
        with open(ckpt_path, "r") as f:
            completed_in_full = json.load(f).get("completed_images", 0)

    logger.info("Verified %d images available in partial full Phikon cache.", completed_in_full)

    missing_subset_indices = []
    reused_count = 0

    for i, r_idx in enumerate(row_indices):
        if r_idx < completed_in_full:
            phi_sub_cls[i] = phikon_full_cls[r_idx]
            phi_sub_patch[i] = phikon_full_patch[r_idx]
            reused_count += 1
        else:
            missing_subset_indices.append(i)

    phi_sub_cls.flush()
    phi_sub_patch.flush()
    logger.info("Reused %d / %d embeddings from existing Phikon cache. Missing to extract: %d.", reused_count, n_subset, len(missing_subset_indices))

    # 4. Extract missing Phikon samples
    if missing_subset_indices:
        torch.set_num_threads(10)
        logger.info("Instantiating frozen Phikon model to extract %d missing images...", len(missing_subset_indices))
        model = PhikonEncoderWrapper()
        preprocessor = DeterministicPathologyPreprocessor(image_size=224)

        batch_size = 16
        n_missing = len(missing_subset_indices)
        t_ext_0 = time.time()

        for b_start in range(0, n_missing, batch_size):
            b_end = min(b_start + batch_size, n_missing)
            batch_sub_idxs = missing_subset_indices[b_start:b_end]

            tensors = []
            for s_idx in batch_sub_idxs:
                rel_path = df_subset.iloc[s_idx]["relative_path"]
                img_path = dataset_root / rel_path
                tensor = preprocessor.preprocess_image(img_path)
                tensors.append(tensor)

            batch_tensor = torch.stack(tensors)
            cls_norm, patch_norm = model(batch_tensor)

            for local_i, s_idx in enumerate(batch_sub_idxs):
                phi_sub_cls[s_idx] = cls_norm[local_i].cpu().numpy().astype(np.float32)
                phi_sub_patch[s_idx] = patch_norm[local_i].cpu().numpy().astype(np.float32)

            if (b_end % 160 == 0) or (b_end == n_missing):
                phi_sub_cls.flush()
                phi_sub_patch.flush()
                dur = time.time() - t_ext_0
                speed = b_end / max(dur, 1e-3)
                eta_min = (n_missing - b_end) / max(speed, 1e-3) / 60.0
                logger.info("Phikon Subset Progress: %d / %d (%.1f%%) | %.2f img/s | ETA: %.1f min", b_end, n_missing, (b_end / n_missing) * 100, speed, eta_min)

        phi_sub_cls.flush()
        phi_sub_patch.flush()
        logger.info("Finished extracting all missing Phikon samples.")

    df_subset.to_csv(phikon_subset_dir / "manifest.csv", index=False)

    # 5. Numerical Validation
    logger.info("Validating 10% subset caches...")
    for name, c_arr, p_arr, dim in [
        ("DINOv2", dino_sub_cls, dino_sub_patch, 384),
        ("Phikon", phi_sub_cls, phi_sub_patch, 768),
    ]:
        assert c_arr.shape == (n_subset, dim), f"{name} CLS shape mismatch: {c_arr.shape}"
        assert p_arr.shape == (n_subset, dim), f"{name} Patch shape mismatch: {p_arr.shape}"
        assert bool(np.all(np.isfinite(c_arr))), f"{name} non-finite in CLS!"
        assert bool(np.all(np.isfinite(p_arr))), f"{name} non-finite in Patch!"
        assert bool(np.all(np.abs(c_arr).sum(axis=1) > 0)), f"{name} zero vectors in CLS!"
        assert bool(np.all(np.abs(p_arr).sum(axis=1) > 0)), f"{name} zero vectors in Patch!"

        c_norms = np.linalg.norm(c_arr, axis=1)
        p_norms = np.linalg.norm(p_arr, axis=1)
        assert abs(np.mean(c_norms) - 1.0) < 1e-4
        assert abs(np.mean(p_norms) - 1.0) < 1e-4
        logger.info("%s 10%% subset cache validated successfully: All %d samples finite and L2-normalized (mean=1.0000).", name, n_subset)

    duration = time.time() - start_time
    summary = {
        "status": "success",
        "subset_samples": n_subset,
        "split_distribution": df_subset["split"].value_counts().to_dict(),
        "class_distribution": df_subset["class_name"].value_counts().to_dict(),
        "reused_phikon_embeddings": reused_count,
        "newly_extracted_phikon_embeddings": len(missing_subset_indices),
        "total_duration_seconds": round(duration, 2),
        "total_duration_minutes": round(duration / 60.0, 2),
    }

    with open(subset_dir / "subset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Subset preparation completed in %.1f minutes.", duration / 60.0)
    return summary


if __name__ == "__main__":
    build_subset_caches()

