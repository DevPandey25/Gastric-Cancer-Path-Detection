"""Embedding Cache Manager for GastricPath AI Foundation Models.

Persists embedding feature matrices as memory-mapped binary NumPy arrays (.npy)
and tracks patch provenance metadata in structured JSON/CSV manifests outside
the primary application database. Supports incremental appending, extraction resumption,
and cache integrity validation.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any
import numpy as np

from research.foundation_models.metadata import ImageEmbeddingRecord

logger = logging.getLogger(__name__)


class EmbeddingCacheManager:
    """Manages disk persistence and loading of embedding matrices and provenance manifests."""

    def __init__(self, cache_dir: Path | str, model_name: str = "dinov2_vits14") -> None:
        self.cache_dir = Path(cache_dir).resolve()
        self.model_name = model_name
        self.model_cache_dir = self.cache_dir / model_name
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

        self.embeddings_file = self.model_cache_dir / "embeddings.npy"
        self.manifest_json_file = self.model_cache_dir / "manifest.json"
        self.manifest_csv_file = self.model_cache_dir / "manifest.csv"

    def exists(self) -> bool:
        """Returns True if both embedding matrix and manifest exist on disk."""
        return self.embeddings_file.exists() and (
            self.manifest_json_file.exists() or self.manifest_csv_file.exists()
        )

    def load_cached_ids(self) -> set[str]:
        """Returns set of relative paths or image IDs already present in the cache."""
        if not self.manifest_json_file.exists():
            return set()
        with open(self.manifest_json_file, "r", encoding="utf-8") as f:
            records = json.load(f)
        return {r["relative_path"] for r in records}

    def save_cache(
        self,
        embeddings: np.ndarray,
        records: list[ImageEmbeddingRecord],
        append: bool = False,
    ) -> None:
        """Persists embedding matrix and metadata records atomically.

        Args:
            embeddings: Float32 array of shape [N, embedding_dim]
            records: List of N ImageEmbeddingRecord objects
            append: If True and cache exists, appends new rows
        """
        if len(records) != len(embeddings):
            raise ValueError(
                f"Mismatch between records count ({len(records)}) and embeddings rows ({len(embeddings)})"
            )

        if append and self.exists():
            existing_embeddings, existing_records = self.load_cache()
            # Filter out duplicates by relative_path
            existing_paths = {r.relative_path for r in existing_records}
            new_indices = [i for i, r in enumerate(records) if r.relative_path not in existing_paths]

            if new_indices:
                combined_embeddings = np.vstack([existing_embeddings, embeddings[new_indices]])
                combined_records = existing_records + [records[i] for i in new_indices]
            else:
                logger.info("All records already present in cache; no new embeddings appended.")
                return
            final_embeddings = combined_embeddings
            final_records = combined_records
        else:
            final_embeddings = embeddings.astype(np.float32)
            final_records = records

        # Atomic write for numpy array
        temp_npy = self.model_cache_dir / "embeddings.tmp.npy"
        np.save(temp_npy, final_embeddings)
        temp_npy.replace(self.embeddings_file)

        # Write manifest JSON
        temp_json = self.model_cache_dir / "manifest.tmp.json"
        with open(temp_json, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in final_records], f, indent=2)
        temp_json.replace(self.manifest_json_file)

        # Write manifest CSV for external transparency
        temp_csv = self.model_cache_dir / "manifest.tmp.csv"
        with open(temp_csv, "w", newline="", encoding="utf-8") as f:
            if final_records:
                fieldnames = list(final_records[0].to_dict().keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in final_records:
                    writer.writerow(r.to_dict())
        temp_csv.replace(self.manifest_csv_file)

        logger.info(
            "Successfully persisted %d embeddings (dim=%d) to %s",
            len(final_records),
            final_embeddings.shape[1] if len(final_records) > 0 else 0,
            self.model_cache_dir,
        )

    def load_cache(self) -> tuple[np.ndarray, list[ImageEmbeddingRecord]]:
        """Loads embeddings matrix and metadata records with integrity validation."""
        if not self.embeddings_file.exists():
            raise FileNotFoundError(f"Embedding file not found: {self.embeddings_file}")
        if not self.manifest_json_file.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_json_file}")

        embeddings = np.load(self.embeddings_file)

        with open(self.manifest_json_file, "r", encoding="utf-8") as f:
            raw_records = json.load(f)
        records = [ImageEmbeddingRecord.from_dict(r) for r in raw_records]

        # Integrity Validation
        if len(embeddings) != len(records):
            raise ValueError(
                f"Cache corruption detected! Matrix has {len(embeddings)} rows, "
                f"but manifest contains {len(records)} records."
            )

        return embeddings, records

    def verify_cache_integrity(self) -> dict[str, Any]:
        """Runs comprehensive integrity checks on cached embedding artifacts."""
        if not self.exists():
            return {"status": "empty", "valid": True, "total_records": 0}

        embeddings, records = self.load_cache()
        n_samples, dim = embeddings.shape

        # Check for NaN / Inf
        has_nan = bool(np.isnan(embeddings).any())
        has_inf = bool(np.isinf(embeddings).any())

        # Check L2 normalization (norms should be ~1.0)
        norms = np.linalg.norm(embeddings, axis=1)
        is_normalized = bool(np.allclose(norms, 1.0, atol=1e-3))

        # Check path uniqueness
        paths = [r.relative_path for r in records]
        unique_paths_count = len(set(paths))
        has_duplicate_paths = bool(unique_paths_count != len(paths))

        splits_distribution = {}
        classes_distribution = {}
        for r in records:
            splits_distribution[r.dataset_split] = splits_distribution.get(r.dataset_split, 0) + 1
            classes_distribution[r.class_name] = classes_distribution.get(r.class_name, 0) + 1

        is_valid = (not has_nan) and (not has_inf) and (not has_duplicate_paths)

        return {
            "status": "verified" if is_valid else "corrupt",
            "valid": is_valid,
            "total_records": n_samples,
            "embedding_dimension": dim,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "is_l2_normalized": is_normalized,
            "unique_paths_count": unique_paths_count,
            "has_duplicate_paths": has_duplicate_paths,
            "splits_distribution": splits_distribution,
            "classes_distribution": classes_distribution,
            "embeddings_file_bytes": self.embeddings_file.stat().st_size,
        }

