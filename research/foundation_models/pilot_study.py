"""Controlled Pilot Study for Pathology Foundation Model Feasibility.

Executes a controlled feasibility benchmark on exactly 32 images from the training split:
- Measures model loading latency and per-image inference speed.
- Verifies embedding dimension (384), finite values, and L2 normalization.
- Verifies deterministic reproducibility across multiple runs.
- Verifies cache save, reload, and round-trip integrity.
- Verifies retrieval self-exclusion and duplicate exclusion.
- Extrapolates storage footprint and full dataset extraction duration for 31,096 images.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import psutil
import torch

from research.foundation_models.cache_manager import EmbeddingCacheManager
from research.foundation_models.config import FoundationModelConfig
from research.foundation_models.embedding_extractor import PathologyEmbeddingExtractor
from research.foundation_models.metadata import ImageEmbeddingRecord
from research.foundation_models.model_loader import load_foundation_encoder
from research.foundation_models.retrieval import HistopathologySimilarityRetriever

logger = logging.getLogger(__name__)


def run_controlled_pilot_study(
    workspace_root: Path | str | None = None,
    pilot_size: int = 32,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Runs the controlled feasibility benchmark on at most 32 training images."""
    if workspace_root is None:
        workspace_root = Path(__file__).resolve().parent.parent.parent
    else:
        workspace_root = Path(workspace_root).resolve()

    process = psutil.Process(os.getpid())
    ram_initial_mb = process.memory_info().rss / (1024 * 1024)

    print("=" * 80)
    print("GASTRICPATH AI — PHASE 4.0 FOUNDATION MODEL CONTROLLED PILOT")
    print("=" * 80)
    print(f"Hardware: CPU-Only | Total RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print(f"Initial Process RSS: {ram_initial_mb:.2f} MB")

    # 1. Select at most 32 images from training split (balanced across 8 classes)
    train_manifest_path = workspace_root / "research" / "splits" / "manifest_train.csv"
    dataset_root = workspace_root / "data" / "raw" / "HMU-GC-HE-30K" / "all_image"

    if not train_manifest_path.exists():
        raise FileNotFoundError(f"Train manifest not found: {train_manifest_path}")

    df_train = pd.read_csv(train_manifest_path)
    classes = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]
    per_class_count = pilot_size // len(classes)

    selected_dfs = []
    for c in classes:
        c_subset = df_train[df_train["class_name"] == c].head(per_class_count)
        selected_dfs.append(c_subset)

    pilot_df = pd.concat(selected_dfs).reset_index(drop=True)
    actual_pilot_count = len(pilot_df)
    assert actual_pilot_count <= 32, f"Pilot cohort exceeded 32 images: {actual_pilot_count}"
    print(f"Selected {actual_pilot_count} pilot images from training split (4 per class across 8 classes).")

    pilot_records = []
    for _, row in pilot_df.iterrows():
        pilot_records.append({
            "image_id": row["relative_path"],
            "relative_path": row["relative_path"],
            "sha256": row["sha256"],
            "class_name": row["class_name"],
            "dataset_split": "train",
        })

    # 2. Measure Model Loading Time
    print("\n[Step 1/6] Loading Foundation Model (DINOv2 ViT-S/14)...")
    config = FoundationModelConfig(
        model_name="dinov2_vits14",
        embedding_dim=384,
        batch_size=batch_size,
        device="cpu",
        normalize_l2=True,
        cache_dir="research/foundation_models/cache/pilot",
    )

    t0 = time.perf_counter()
    model, embedding_dim = load_foundation_encoder(config.model_name, device=config.device)
    model_load_time_sec = time.perf_counter() - t0
    ram_after_model_mb = process.memory_info().rss / (1024 * 1024)
    model_ram_delta_mb = ram_after_model_mb - ram_initial_mb

    print(f"  Model Load Time: {model_load_time_sec:.4f} seconds")
    print(f"  RAM after Model Load: {ram_after_model_mb:.2f} MB (Delta: +{model_ram_delta_mb:.2f} MB)")
    print(f"  Reported Embedding Dim: {embedding_dim}")

    # 3. Perform Single Encoder Feasibility Run & Measure Speed
    print(f"\n[Step 2/6] Extracting Embeddings for {actual_pilot_count} Images (Batch Size = {batch_size})...")
    extractor = PathologyEmbeddingExtractor(config=config, model=model, dataset_root=dataset_root)

    t_inf_start = time.perf_counter()
    embeddings_run1, metadata_run1, metrics_run1 = extractor.extract_dataset_subset(
        image_records=pilot_records,
        dataset_root=dataset_root,
        save_to_cache=False,
    )
    inf_duration_sec = time.perf_counter() - t_inf_start
    per_image_inf_sec = inf_duration_sec / actual_pilot_count
    images_per_sec = actual_pilot_count / inf_duration_sec

    ram_after_inf_mb = process.memory_info().rss / (1024 * 1024)

    print(f"  Extraction Duration: {inf_duration_sec:.4f} seconds ({images_per_sec:.2f} img/s)")
    print(f"  Latency Per Image: {per_image_inf_sec * 1000:.2f} ms")
    print(f"  Output Tensor Shape: {embeddings_run1.shape}")

    # 4. Numerical Property Checks
    print("\n[Step 3/6] Verifying Numerical Properties & Normalization...")
    assert embeddings_run1.shape == (actual_pilot_count, 384), f"Shape mismatch: {embeddings_run1.shape}"
    assert not np.isnan(embeddings_run1).any(), "NaN values found in embeddings!"
    assert not np.isinf(embeddings_run1).any(), "Infinite values found in embeddings!"

    # Check L2 normalization
    norms = np.linalg.norm(embeddings_run1, axis=1)
    min_norm, max_norm = float(np.min(norms)), float(np.max(norms))
    assert np.allclose(norms, 1.0, atol=1e-3), f"Embeddings not L2-normalized: min={min_norm}, max={max_norm}"
    print(f"  L2 Norm Verification: All norms = 1.0000 (Min: {min_norm:.4f}, Max: {max_norm:.4f})")

    # 5. Deterministic Reproducibility Run
    print("\n[Step 4/6] Verifying Deterministic Reproducibility...")
    embeddings_run2, _, _ = extractor.extract_dataset_subset(
        image_records=pilot_records,
        dataset_root=dataset_root,
        save_to_cache=False,
    )
    max_abs_diff = float(np.max(np.abs(embeddings_run1 - embeddings_run2)))
    assert max_abs_diff < 1e-5, f"Non-deterministic outputs detected! Max abs diff: {max_abs_diff}"
    print(f"  Deterministic Check: PASS (Max absolute difference between runs: {max_abs_diff:.8e})")

    # 6. Cache Round-Trip Integrity
    print("\n[Step 5/6] Verifying Cache Round-Trip & Storage Integrity...")
    cache_mgr = EmbeddingCacheManager(cache_dir=workspace_root / config.cache_dir, model_name=config.model_name)
    cache_mgr.save_cache(embeddings_run1, metadata_run1, append=False)

    loaded_emb, loaded_meta = cache_mgr.load_cache()
    assert np.array_equal(embeddings_run1, loaded_emb), "Cache round-trip embedding mismatch!"
    assert len(metadata_run1) == len(loaded_meta), "Cache round-trip metadata length mismatch!"

    integrity = cache_mgr.verify_cache_integrity()
    assert integrity["valid"], f"Cache integrity verification failed: {integrity}"
    cache_bytes = integrity["embeddings_file_bytes"]
    print(f"  Cache File Size on Disk: {cache_bytes / 1024:.2f} KB ({cache_bytes} bytes for {actual_pilot_count} samples)")
    print(f"  Average Storage per Image Embedding: {cache_bytes / actual_pilot_count:.1f} bytes (exactly 384 * 4 = 1,536 bytes + header)")

    # 7. Retrieval Verification: Self-Exclusion & Duplicate Exclusion
    print("\n[Step 6/6] Verifying Retrieval Engine (Self-Exclusion & Duplicate Handling)...")
    retriever = HistopathologySimilarityRetriever(
        embeddings=loaded_emb,
        records=loaded_meta,
        reference_split="train",
        use_faiss=False,  # Test NumPy fallback
    )

    query_idx = 0
    query_rec = loaded_meta[query_idx]
    query_vec = loaded_emb[query_idx]

    # Query with self-exclusion enabled
    matches_self_excluded = retriever.query(
        query_vector=query_vec,
        top_k=5,
        query_image_id=query_rec.image_id,
        query_relative_path=query_rec.relative_path,
        query_sha256=query_rec.sha256,
        exclude_self=True,
        exclude_duplicates=True,
    )
    returned_ids = [m.neighbor_image_id for m in matches_self_excluded]
    assert query_rec.image_id not in returned_ids, "Self-match was NOT excluded!"
    print(f"  Self-Exclusion Test: PASS (Query '{query_rec.image_id}' not present in top-{len(returned_ids)})")

    # Verify duplicate exclusion
    # Create duplicate record with different ID but same SHA-256
    dup_meta = ImageEmbeddingRecord(
        image_id="DUPLICATE_OF_QUERY",
        relative_path="DUPLICATE_PATH",
        sha256=query_rec.sha256,  # Identical hash
        class_name=query_rec.class_name,
        dataset_split="train",
        model_name=config.model_name,
        model_revision=config.model_revision,
        preprocessing_version=config.preprocessing_version,
        embedding_dim=384,
        extracted_at_utc="2026-11-06T00:00:00Z",
        l2_normalized=True,
    )
    retriever_with_dup = HistopathologySimilarityRetriever(
        embeddings=np.vstack([loaded_emb, query_vec]),
        records=loaded_meta + [dup_meta],
        reference_split="train",
        use_faiss=False,
    )
    matches_dup_excluded = retriever_with_dup.query(
        query_vector=query_vec,
        top_k=5,
        query_image_id=query_rec.image_id,
        query_relative_path=query_rec.relative_path,
        query_sha256=query_rec.sha256,
        exclude_self=True,
        exclude_duplicates=True,
    )
    returned_dup_ids = [m.neighbor_image_id for m in matches_dup_excluded]
    assert "DUPLICATE_OF_QUERY" not in returned_dup_ids, "Duplicate hash was NOT excluded!"
    print(f"  Duplicate-Exclusion Test: PASS (Duplicate hash successfully filtered)")

    # 8. Full Dataset Extrapolations (31,096 images)
    total_dataset_images = 31096
    est_total_duration_sec = per_image_inf_sec * total_dataset_images
    est_total_duration_hours = est_total_duration_sec / 3600.0

    # Storage: 384 dimensions * 4 bytes per float32 = 1,536 bytes per patch
    est_storage_bytes = total_dataset_images * 384 * 4
    est_storage_mb = est_storage_bytes / (1024 * 1024)

    pilot_report = {
        "model_name": config.model_name,
        "embedding_dim": embedding_dim,
        "pilot_images_evaluated": actual_pilot_count,
        "model_loading_latency_seconds": round(model_load_time_sec, 4),
        "inference_latency_per_image_seconds": round(per_image_inf_sec, 4),
        "inference_latency_per_image_ms": round(per_image_inf_sec * 1000, 2),
        "throughput_images_per_second": round(images_per_sec, 2),
        "peak_process_rss_mb": round(ram_after_inf_mb, 2),
        "model_ram_footprint_mb": round(model_ram_delta_mb, 2),
        "single_embedding_size_bytes": 384 * 4,
        "cache_size_for_pilot_bytes": cache_bytes,
        "full_dataset_extrapolations": {
            "total_images": total_dataset_images,
            "estimated_extraction_duration_seconds": round(est_total_duration_sec, 2),
            "estimated_extraction_duration_minutes": round(est_total_duration_sec / 60.0, 2),
            "estimated_extraction_duration_hours": round(est_total_duration_hours, 2),
            "estimated_raw_matrix_storage_mb": round(est_storage_mb, 2),
            "estimated_manifest_storage_mb": round((total_dataset_images * 250) / (1024 * 1024), 2),  # ~250 bytes per JSON record
            "total_estimated_storage_mb": round(est_storage_mb + (total_dataset_images * 250) / (1024 * 1024), 2),
        },
        "retrieval_verification": {
            "self_match_excluded": True,
            "duplicate_hash_excluded": True,
            "backend": "NumPy Vectorized Cosine Search",
        },
    }

    report_output_path = workspace_root / "research" / "foundation_models" / "pilot_report.json"
    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_output_path, "w", encoding="utf-8") as f:
        json.dump(pilot_report, f, indent=2)
    print(f"\nSaved Controlled Pilot Report to {report_output_path}")

    print("\n" + "=" * 80)
    print("CONTROLLED PILOT BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Model: {config.model_name} (ViT-S/14, 22.05M params, dim=384)")
    print(f"Model Load Latency: {model_load_time_sec:.3f} s")
    print(f"CPU Latency Per Image: {per_image_inf_sec * 1000:.2f} ms ({images_per_sec:.2f} images/s)")
    print(f"Full Dataset (31,096 images) Projected Duration: {est_total_duration_hours:.2f} hours ({est_total_duration_sec / 60:.1f} minutes)")
    print(f"Full Dataset Projected Embedding Storage: {est_storage_mb:.2f} MB")
    print("=" * 80)

    return pilot_report


if __name__ == "__main__":
    run_controlled_pilot_study()

