"""Frozen Foundation Model Feature Extraction Pipeline.

Extracts dense representation vectors from histopathology biopsy patches, applies L2 normalization,
validates numerical integrity, and associates each embedding with verifiable dataset provenance.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from research.foundation_models.cache_manager import EmbeddingCacheManager
from research.foundation_models.config import FoundationModelConfig
from research.foundation_models.metadata import ImageEmbeddingRecord
from research.foundation_models.model_loader import load_foundation_encoder
from research.foundation_models.preprocessing import DeterministicPathologyPreprocessor

logger = logging.getLogger(__name__)


class PathologyEmbeddingExtractor:
    """Extracts representation embeddings from histopathology images using frozen encoders."""

    def __init__(
        self,
        config: FoundationModelConfig | None = None,
        model: nn.Module | None = None,
        dataset_root: Path | str | None = None,
    ) -> None:
        self.config = config or FoundationModelConfig()
        self.device = torch.device(self.config.device)
        self.dataset_root = Path(dataset_root).resolve() if dataset_root else None

        if model is not None:
            self.model = model
            self.embedding_dim = self.config.embedding_dim
        else:
            self.model, self.embedding_dim = load_foundation_encoder(
                model_name=self.config.model_name,
                device=self.config.device,
                pretrained=True,
            )

        self.preprocessor = DeterministicPathologyPreprocessor(
            image_size=self.config.image_size,
            mean=self.config.mean,
            std=self.config.std,
        )

        self.cache_manager = EmbeddingCacheManager(
            cache_dir=self.config.cache_dir,
            model_name=self.config.model_name,
        )

    def extract_patch_embedding(self, image_input: Any) -> np.ndarray:
        """Extracts a single L2-normalized embedding vector of shape [embedding_dim]."""
        tensor = self.preprocessor.preprocess_image(image_input).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor)
            if self.config.normalize_l2:
                feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
        arr = feat.cpu().numpy().squeeze(0).astype(np.float32)

        if np.isnan(arr).any() or np.isinf(arr).any():
            raise ValueError("Extracted embedding contains NaN or Inf values!")
        return arr

    def extract_dataset_subset(
        self,
        image_records: list[dict[str, Any]],
        dataset_root: Path | str | None = None,
        save_to_cache: bool = True,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[np.ndarray, list[ImageEmbeddingRecord], dict[str, Any]]:
        """Extracts embeddings for a defined cohort of images in batches.

        Args:
            image_records: List of dicts with keys 'relative_path', 'sha256', 'class_name', 'dataset_split'
            dataset_root: Path to images directory
            save_to_cache: Whether to persist results to EmbeddingCacheManager
            progress_callback: Optional callback(current, total)
        """
        root = Path(dataset_root or self.dataset_root).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Dataset root directory does not exist: {root}")

        total_images = len(image_records)
        batch_size = self.config.batch_size
        all_embeddings: list[np.ndarray] = []
        all_metadata: list[ImageEmbeddingRecord] = []

        start_time = time.perf_counter()
        logger.info(
            "Starting embedding extraction for %d images with %s (batch_size=%d)...",
            total_images,
            self.config.model_name,
            batch_size,
        )

        for batch_start in range(0, total_images, batch_size):
            batch_slice = image_records[batch_start : batch_start + batch_size]
            batch_tensors: list[torch.Tensor] = []
            valid_slice_records: list[dict[str, Any]] = []

            for item in batch_slice:
                rel_path = item["relative_path"]
                full_path = root / rel_path
                if not full_path.exists():
                    alt_path = root / "all_image" / rel_path
                    if alt_path.exists():
                        full_path = alt_path
                    else:
                        raise FileNotFoundError(f"Image patch missing: {full_path}")

                t = self.preprocessor.preprocess_image(full_path)
                batch_tensors.append(t)
                valid_slice_records.append(item)

            if not batch_tensors:
                continue

            input_batch = torch.stack(batch_tensors, dim=0).to(self.device)

            with torch.no_grad():
                feats = self.model(input_batch)
                if self.config.normalize_l2:
                    feats = torch.nn.functional.normalize(feats, p=2, dim=-1)

            feats_arr = feats.cpu().numpy().astype(np.float32)

            # Validate numerical properties
            if np.isnan(feats_arr).any():
                raise ValueError(f"NaN detected in extracted features for batch starting at index {batch_start}")
            if np.isinf(feats_arr).any():
                raise ValueError(f"Inf detected in extracted features for batch starting at index {batch_start}")

            all_embeddings.append(feats_arr)

            now_str = datetime.now(timezone.utc).isoformat()
            for i, rec in enumerate(valid_slice_records):
                meta = ImageEmbeddingRecord(
                    image_id=rec.get("image_id", rec["relative_path"]),
                    relative_path=rec["relative_path"],
                    sha256=rec["sha256"],
                    class_name=rec["class_name"],
                    dataset_split=rec.get("dataset_split", rec.get("split", "unknown")),
                    model_name=self.config.model_name,
                    model_revision=self.config.model_revision,
                    preprocessing_version=self.config.preprocessing_version,
                    embedding_dim=self.embedding_dim,
                    extracted_at_utc=now_str,
                    l2_normalized=self.config.normalize_l2,
                )
                all_metadata.append(meta)

            if progress_callback:
                progress_callback(min(batch_start + batch_size, total_images), total_images)

        total_duration = time.perf_counter() - start_time
        final_matrix = np.vstack(all_embeddings) if all_embeddings else np.empty((0, self.embedding_dim), dtype=np.float32)

        metrics = {
            "total_images_extracted": len(all_metadata),
            "embedding_dimension": self.embedding_dim,
            "total_duration_seconds": round(total_duration, 4),
            "seconds_per_image": round(total_duration / max(1, len(all_metadata)), 4),
            "images_per_second": round(len(all_metadata) / max(0.001, total_duration), 2),
            "model_name": self.config.model_name,
            "device": str(self.device),
        }

        if save_to_cache and len(all_metadata) > 0:
            self.cache_manager.save_cache(final_matrix, all_metadata, append=True)

        return final_matrix, all_metadata, metrics

