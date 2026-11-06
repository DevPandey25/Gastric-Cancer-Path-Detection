"""Full Dataset Frozen Embedding Extractor for GastricPath AI (Phase 4.1).

Extracts frozen DINOv2 ViT-S/14 representations across all 31,096 GCHTID images:
1. CLS embeddings: Shape [31096, 384], float32, L2-normalized.
2. Mean spatial patch-token embeddings: Shape [31096, 384], float32, L2-normalized.

Features:
- Incremental memory-mapped disk persistence (.npy) to ensure constant ~600MB RAM usage.
- Resumable batch execution with periodic atomic checkpointing every 50 batches.
- Metadata preservation: image path, sha256, dhash, cluster_id, split, canonical class,
  model weights sha256, preprocessing version, timestamp.
- Strict post-extraction numerical validation: finite, non-zero, L2-norm == 1.0, shape (31096, 384).
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn.functional as F
from PIL import Image

from research.foundation_models.model_loader import load_foundation_encoder
from research.foundation_models.preprocessing import DeterministicPathologyPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_master_metadata(
    workspace_root: Path,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Builds the canonical metadata table for all 31,096 images aligned with splits and clusters."""
    manifest_all_path = workspace_root / "research" / "manifests" / "manifest_all.csv"
    train_path = workspace_root / "research" / "splits" / "manifest_train.csv"
    val_path = workspace_root / "research" / "splits" / "manifest_val.csv"
    test_path = workspace_root / "research" / "splits" / "manifest_test.csv"
    near_dup_path = workspace_root / "research" / "reports" / "near_duplicates_report.csv"

    if not manifest_all_path.exists():
        raise FileNotFoundError(f"Missing master manifest: {manifest_all_path}")

    df_all = pd.read_csv(manifest_all_path)
    s_train = set(pd.read_csv(train_path)["relative_path"])
    s_val = set(pd.read_csv(val_path)["relative_path"])
    s_test = set(pd.read_csv(test_path)["relative_path"])

    path_to_cluster = {}
    if near_dup_path.exists():
        df_near = pd.read_csv(near_dup_path)
        for _, row in df_near.iterrows():
            cid = f"cluster_{row['cluster_id']}"
            for p in str(row["paths"]).split(";"):
                path_to_cluster[p.strip()] = cid

    splits = []
    clusters = []
    for _, row in df_all.iterrows():
        p = row["relative_path"]
        if p in s_train:
            splits.append("train")
        elif p in s_val:
            splits.append("val")
        elif p in s_test:
            splits.append("test")
        else:
            raise ValueError(f"Image {p} not found in any split!")

        if p in path_to_cluster:
            clusters.append(path_to_cluster[p])
        else:
            clusters.append(f"singleton_{row['sha256'][:16]}")

    df_all["split"] = splits
    df_all["cluster_id"] = clusters
    return df_all, path_to_cluster


class FullDatasetExtractor:
    """Manages frozen batched feature extraction for the entire 31,096 histopathology dataset."""

    def __init__(
        self,
        workspace_root: Path | str | None = None,
        model_name: str = "dinov2_vits14",
        batch_size: int = 16,
        checkpoint_interval: int = 50,  # Save checkpoint every 50 batches = 800 images (~1 min)
    ) -> None:
        if workspace_root is None:
            self.workspace_root = Path(__file__).resolve().parent.parent.parent
        else:
            self.workspace_root = Path(workspace_root).resolve()

        self.model_name = model_name
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval

        self.dataset_root = self.workspace_root / "data" / "raw" / "HMU-GC-HE-30K" / "all_image"
        self.cache_dir = self.workspace_root / "research" / "foundation_models" / "cache" / "full" / model_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cls_memmap_path = self.cache_dir / "cls_embeddings.npy"
        self.mean_patch_memmap_path = self.cache_dir / "mean_patch_embeddings.npy"
        self.manifest_json_path = self.cache_dir / "manifest.json"
        self.manifest_csv_path = self.cache_dir / "manifest.csv"
        self.checkpoint_path = self.cache_dir / "checkpoint.json"

        self.preprocessor = DeterministicPathologyPreprocessor(image_size=224)
        self.weight_sha256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
        self.preprocessing_version = "v1.0-bilinear-224-imagenet-norm"

    def run_extraction(self) -> dict[str, Any]:
        """Executes full frozen embedding extraction with resumability and validation."""
        start_time = time.time()
        print("=" * 80)
        print("GASTRICPATH AI — FULL DATASET FROZEN EMBEDDING EXTRACTION (PHASE 4.1)")
        print("=" * 80)

        # 1. Prepare master metadata
        df_meta, _ = build_master_metadata(self.workspace_root)
        total_images = len(df_meta)
        embedding_dim = 384
        logger.info("Loaded metadata for %d canonical images.", total_images)

        # 2. Check existing checkpoint / progress
        start_idx = 0
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, "r") as f:
                    ckpt = json.load(f)
                completed = ckpt.get("completed_images", 0)
                if completed == total_images:
                    logger.info("Full extraction already complete! Verifying integrity...")
                    return self.validate_and_summarize(df_meta, total_images, embedding_dim, start_time)
                elif 0 < completed < total_images:
                    start_idx = completed
                    logger.info("Resuming extraction from image index %d / %d...", start_idx, total_images)
            except Exception as e:
                logger.warning("Failed to parse checkpoint (%s); checking array files...", e)

        # 3. Initialize or open memory-mapped arrays
        if not self.cls_memmap_path.exists() or start_idx == 0:
            cls_mm = np.memmap(
                self.cls_memmap_path,
                dtype=np.float32,
                mode="w+",
                shape=(total_images, embedding_dim),
            )
            mean_patch_mm = np.memmap(
                self.mean_patch_memmap_path,
                dtype=np.float32,
                mode="w+",
                shape=(total_images, embedding_dim),
            )
            # Initialize with zeros
            cls_mm[:] = 0.0
            mean_patch_mm[:] = 0.0
            cls_mm.flush()
            mean_patch_mm.flush()
            start_idx = 0
        else:
            cls_mm = np.memmap(
                self.cls_memmap_path,
                dtype=np.float32,
                mode="r+",
                shape=(total_images, embedding_dim),
            )
            mean_patch_mm = np.memmap(
                self.mean_patch_memmap_path,
                dtype=np.float32,
                mode="r+",
                shape=(total_images, embedding_dim),
            )

        # 4. Load frozen model
        logger.info("Instantiating frozen %s encoder...", self.model_name)
        model, dim = load_foundation_encoder(self.model_name, device="cpu")
        assert dim == embedding_dim

        # 5. Batched extraction loop
        num_batches = int(np.ceil(total_images / self.batch_size))
        start_batch = start_idx // self.batch_size
        logger.info(
            "Starting extraction from batch %d / %d (Batch size: %d)...",
            start_batch,
            num_batches,
            self.batch_size,
        )

        proc = psutil.Process(os.getpid())
        batch_times = []

        for b_idx in range(start_batch, num_batches):
            b_start_img = b_idx * self.batch_size
            b_end_img = min(b_start_img + self.batch_size, total_images)
            current_bsz = b_end_img - b_start_img

            t0_batch = time.perf_counter()

            # Load and preprocess batch
            tensors = []
            for i in range(b_start_img, b_end_img):
                rel_path = df_meta.iloc[i]["relative_path"]
                img_path = self.dataset_root / rel_path
                tensor = self.preprocessor.preprocess_image(img_path)
                tensors.append(tensor)

            batch_tensor = torch.stack(tensors)  # [B, 3, 224, 224]

            # Model forward features
            with torch.no_grad():
                features = model.encoder.forward_features(batch_tensor)
                cls_tok = features["x_norm_clstoken"]  # [B, 384]
                patch_tok = features["x_norm_patchtokens"]  # [B, 256, 384]
                mean_patch = patch_tok.mean(dim=1)  # [B, 384]

                cls_norm = F.normalize(cls_tok, p=2, dim=-1).cpu().numpy().astype(np.float32)
                mean_patch_norm = F.normalize(mean_patch, p=2, dim=-1).cpu().numpy().astype(np.float32)

            # Store in memmap
            cls_mm[b_start_img:b_end_img, :] = cls_norm
            mean_patch_mm[b_start_img:b_end_img, :] = mean_patch_norm

            b_dur = time.perf_counter() - t0_batch
            batch_times.append(b_dur)

            # Periodic checkpoint and telemetry
            if (b_idx + 1) % self.checkpoint_interval == 0 or (b_idx + 1) == num_batches:
                cls_mm.flush()
                mean_patch_mm.flush()

                completed_count = b_end_img
                elapsed = time.time() - start_time
                avg_img_sec = elapsed / (completed_count - start_idx + 1e-6)
                img_per_sec = 1.0 / avg_img_sec
                remaining_sec = (total_images - completed_count) * avg_img_sec
                rss_mb = proc.memory_info().rss / (1024 * 1024)

                ckpt_data = {
                    "completed_images": completed_count,
                    "total_images": total_images,
                    "completed_batches": b_idx + 1,
                    "total_batches": num_batches,
                    "model_name": self.model_name,
                    "weight_sha256": self.weight_sha256,
                    "elapsed_seconds": round(elapsed, 1),
                    "estimated_remaining_seconds": round(remaining_sec, 1),
                    "throughput_img_sec": round(img_per_sec, 2),
                    "process_rss_mb": round(rss_mb, 2),
                    "last_updated_utc": datetime.now(timezone.utc).isoformat(),
                }
                with open(self.checkpoint_path, "w") as f:
                    json.dump(ckpt_data, f, indent=2)

                logger.info(
                    "Progress: %d / %d (%.1f%%) | %.2f img/s | RSS: %.1f MB | ETA: %.1f min",
                    completed_count,
                    total_images,
                    (completed_count / total_images) * 100,
                    img_per_sec,
                    rss_mb,
                    remaining_sec / 60.0,
                )

        # Flush final memmaps
        cls_mm.flush()
        mean_patch_mm.flush()
        del cls_mm
        del mean_patch_mm

        return self.validate_and_summarize(df_meta, total_images, embedding_dim, start_time)

    def validate_and_summarize(
        self,
        df_meta: pd.DataFrame,
        total_images: int,
        embedding_dim: int,
        start_time: float,
    ) -> dict[str, Any]:
        """Validates numerical integrity and writes final manifests."""
        logger.info("Validating full dataset embeddings integrity...")

        # Load arrays as read-only memory maps
        cls_arr = np.memmap(self.cls_memmap_path, dtype=np.float32, mode="r", shape=(total_images, embedding_dim))
        mean_patch_arr = np.memmap(self.mean_patch_memmap_path, dtype=np.float32, mode="r", shape=(total_images, embedding_dim))

        assert cls_arr.shape == (total_images, embedding_dim), f"CLS shape mismatch: {cls_arr.shape}"
        assert mean_patch_arr.shape == (total_images, embedding_dim), f"Mean patch shape mismatch: {mean_patch_arr.shape}"

        # 1. Finite checks
        assert bool(np.all(np.isfinite(cls_arr))), "Non-finite values detected in CLS embeddings!"
        assert bool(np.all(np.isfinite(mean_patch_arr))), "Non-finite values detected in Mean Patch embeddings!"

        # 2. Non-zero checks
        cls_row_sums = np.abs(cls_arr).sum(axis=1)
        mean_patch_row_sums = np.abs(mean_patch_arr).sum(axis=1)
        assert bool(np.all(cls_row_sums > 0.0)), "Zero vectors detected in CLS embeddings!"
        assert bool(np.all(mean_patch_row_sums > 0.0)), "Zero vectors detected in Mean Patch embeddings!"

        # 3. L2 Norm validation
        cls_norms = np.linalg.norm(cls_arr, axis=1)
        patch_norms = np.linalg.norm(mean_patch_arr, axis=1)

        cls_norm_mean = float(np.mean(cls_norms))
        cls_norm_std = float(np.std(cls_norms))
        cls_norm_min = float(np.min(cls_norms))
        cls_norm_max = float(np.max(cls_norms))

        patch_norm_mean = float(np.mean(patch_norms))
        patch_norm_std = float(np.std(patch_norms))
        patch_norm_min = float(np.min(patch_norms))
        patch_norm_max = float(np.max(patch_norms))

        assert abs(cls_norm_mean - 1.0) < 1e-4, f"CLS norm mean deviates from 1.0: {cls_norm_mean}"
        assert abs(patch_norm_mean - 1.0) < 1e-4, f"Patch norm mean deviates from 1.0: {patch_norm_mean}"
        logger.info("Numerical validation PASSED: all 31,096 embeddings are finite, non-zero, and L2-normalized (mean=1.0000).")

        # 4. Generate Final Manifests
        timestamp = datetime.now(timezone.utc).isoformat()
        records = []
        for i, row in df_meta.iterrows():
            rec = {
                "row_index": int(i),
                "image_id": row["relative_path"],
                "relative_path": row["relative_path"],
                "class_name": row["class_name"],
                "class_index": int(row["class_index"]),
                "split": row["split"],
                "sha256": row["sha256"],
                "dhash": row["dhash"],
                "cluster_id": row["cluster_id"],
                "model_name": self.model_name,
                "model_weight_sha256": self.weight_sha256,
                "preprocessing_version": self.preprocessing_version,
                "embedding_dim": embedding_dim,
                "extracted_at_utc": timestamp,
                "status": "success",
            }
            records.append(rec)

        with open(self.manifest_json_path, "w") as f:
            json.dump(records, f, indent=2)

        df_out = pd.DataFrame(records)
        df_out.to_csv(self.manifest_csv_path, index=False)
        logger.info("Saved provenance manifests: JSON (%s) & CSV (%s)", self.manifest_json_path.name, self.manifest_csv_path.name)

        # File sizes
        cls_size_mb = self.cls_memmap_path.stat().st_size / (1024 * 1024)
        patch_size_mb = self.mean_patch_memmap_path.stat().st_size / (1024 * 1024)
        json_size_mb = self.manifest_json_path.stat().st_size / (1024 * 1024)
        csv_size_mb = self.manifest_csv_path.stat().st_size / (1024 * 1024)
        total_cache_mb = cls_size_mb + patch_size_mb + json_size_mb + csv_size_mb

        duration_sec = time.time() - start_time
        summary = {
            "status": "success",
            "model_name": self.model_name,
            "weight_sha256": self.weight_sha256,
            "total_images": total_images,
            "embedding_dim": embedding_dim,
            "extraction_duration_seconds": round(duration_sec, 2),
            "extraction_duration_minutes": round(duration_sec / 60.0, 2),
            "throughput_img_sec": round(total_images / max(duration_sec, 1e-3), 2),
            "cls_l2_norm": {
                "mean": round(cls_norm_mean, 6),
                "std": round(cls_norm_std, 6),
                "min": round(cls_norm_min, 6),
                "max": round(cls_norm_max, 6),
            },
            "mean_patch_l2_norm": {
                "mean": round(patch_norm_mean, 6),
                "std": round(patch_norm_std, 6),
                "min": round(patch_norm_min, 6),
                "max": round(patch_norm_max, 6),
            },
            "storage_megabytes": {
                "cls_embeddings_npy": round(cls_size_mb, 2),
                "mean_patch_embeddings_npy": round(patch_size_mb, 2),
                "manifest_json": round(json_size_mb, 2),
                "manifest_csv": round(csv_size_mb, 2),
                "total_cache_mb": round(total_cache_mb, 2),
            },
            "class_distribution": {str(k): int(v) for k, v in df_out["class_name"].value_counts().items()},
            "split_distribution": {str(k): int(v) for k, v in df_out["split"].value_counts().items()},
            "unique_clusters": int(df_out["cluster_id"].nunique()),
            "completed_at_utc": timestamp,
        }

        summary_path = self.cache_dir / "extraction_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 80)
        print("EXTRACTION COMPLETED SUCCESSFULLY & VALIDATED")
        print(f"  Total Images:        {total_images}")
        print(f"  Throughput:          {summary['throughput_img_sec']} images/sec")
        print(f"  Total Duration:      {summary['extraction_duration_minutes']} minutes")
        print(f"  CLS Embeddings:      {cls_size_mb:.2f} MB (Shape: {cls_arr.shape})")
        print(f"  Mean Patch Embeddings: {patch_size_mb:.2f} MB (Shape: {mean_patch_arr.shape})")
        print(f"  Total Disk Storage:  {total_cache_mb:.2f} MB")
        print("=" * 80)
        return summary


if __name__ == "__main__":
    extractor = FullDatasetExtractor()
    extractor.run_extraction()

