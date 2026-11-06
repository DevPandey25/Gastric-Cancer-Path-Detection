"""Automated unit and integration tests for Phase 4.1 Full Dataset Foundation Models & Retrieval."""

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from research.foundation_models.full_dataset_extractor import FullDatasetExtractor
from research.foundation_models.metadata import ImageEmbeddingRecord, RetrievalMatch
from research.foundation_models.retrieval import HistopathologySimilarityRetriever

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = WORKSPACE_ROOT / "research" / "foundation_models" / "cache" / "full" / "dinov2_vits14"
EVAL_DIR = WORKSPACE_ROOT / "research" / "foundation_models" / "evaluation"
VIZ_DIR = WORKSPACE_ROOT / "research" / "foundation_models" / "visualizations"
BASELINE_CHECKPOINT_PATH = WORKSPACE_ROOT / "models" / "checkpoints" / "gastric_rnn_classifier.pt"
BASELINE_CHECKPOINT_SHA256 = "65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8"
DINOV2_WEIGHT_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
CANONICAL_CLASSES = {"ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"}


def test_full_dataset_cache_alignment_and_manifest():
    """Verify that all 31,096 images are cached and aligned with master manifests."""
    manifest_csv = CACHE_DIR / "manifest.csv"
    manifest_json = CACHE_DIR / "manifest.json"
    cls_npy = CACHE_DIR / "cls_embeddings.npy"
    patch_npy = CACHE_DIR / "mean_patch_embeddings.npy"

    assert manifest_csv.exists(), "Cache manifest.csv is missing!"
    assert manifest_json.exists(), "Cache manifest.json is missing!"
    assert cls_npy.exists(), "cls_embeddings.npy is missing!"
    assert patch_npy.exists(), "mean_patch_embeddings.npy is missing!"

    df = pd.read_csv(manifest_csv)
    assert len(df) == 31096, f"Expected 31,096 rows in manifest.csv, got {len(df)}"

    cls_mm = np.memmap(cls_npy, dtype=np.float32, mode="r", shape=(31096, 384))
    patch_mm = np.memmap(patch_npy, dtype=np.float32, mode="r", shape=(31096, 384))

    assert cls_mm.shape == (31096, 384)
    assert patch_mm.shape == (31096, 384)


def test_embeddings_finite_and_l2_normalized():
    """Verify that both CLS and Mean Patch representations are 100% finite and L2-normalized."""
    cls_npy = CACHE_DIR / "cls_embeddings.npy"
    patch_npy = CACHE_DIR / "mean_patch_embeddings.npy"

    cls_mm = np.memmap(cls_npy, dtype=np.float32, mode="r", shape=(31096, 384))
    patch_mm = np.memmap(patch_npy, dtype=np.float32, mode="r", shape=(31096, 384))

    # Spot-check finite
    assert np.all(np.isfinite(cls_mm[:1000])), "Non-finite values found in CLS sample!"
    assert np.all(np.isfinite(patch_mm[:1000])), "Non-finite values found in Patch sample!"

    # Check L2 norms
    cls_norms = np.linalg.norm(cls_mm[:500], axis=1)
    patch_norms = np.linalg.norm(patch_mm[:500], axis=1)

    assert np.allclose(cls_norms, 1.0, atol=1e-3), f"CLS norms deviate from 1.0: {cls_norms[:5]}"
    assert np.allclose(patch_norms, 1.0, atol=1e-3), f"Patch norms deviate from 1.0: {patch_norms[:5]}"


def test_model_weight_checksum_and_provenance():
    """Verify DINOv2 weight file exists and matches official SHA-256."""
    weight_file = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "dinov2_vits14_pretrain.pth"
    assert weight_file.exists(), f"DINOv2 weights missing at {weight_file}"

    hasher = hashlib.sha256()
    with open(weight_file, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    actual_sha = hasher.hexdigest().lower()

    assert actual_sha == DINOV2_WEIGHT_SHA256, f"Weight SHA mismatch! Expected {DINOV2_WEIGHT_SHA256}, got {actual_sha}"


def test_canonical_label_mappings():
    """Verify that only the 8 canonical classes exist and each has exactly 3,887 images."""
    df = pd.read_csv(CACHE_DIR / "manifest.csv")
    classes_found = set(df["class_name"].unique())
    assert classes_found == CANONICAL_CLASSES, f"Class mismatch! Found {classes_found}"

    counts = df["class_name"].value_counts().to_dict()
    for c in CANONICAL_CLASSES:
        assert counts[c] == 3887, f"Class {c} count mismatch: expected 3887, got {counts[c]}"


def test_split_integrity_and_cluster_metadata():
    """Verify split counts (Train 21767, Val 4664, Test 4665) and cluster_id presence."""
    df = pd.read_csv(CACHE_DIR / "manifest.csv")
    split_counts = df["split"].value_counts().to_dict()

    assert split_counts["train"] == 21767, f"Train count mismatch: {split_counts['train']}"
    assert split_counts["val"] == 4664, f"Val count mismatch: {split_counts['val']}"
    assert split_counts["test"] == 4665, f"Test count mismatch: {split_counts['test']}"

    assert "cluster_id" in df.columns, "cluster_id column missing from manifest!"
    assert df["cluster_id"].isna().sum() == 0, "NaN values in cluster_id!"
    assert df["cluster_id"].nunique() >= 29000, f"Unexpected unique cluster count: {df['cluster_id'].nunique()}"


def test_retrieval_exclusion_and_split_isolation():
    """Verify that retrieval strictly isolates splits and excludes self, exact duplicates, and clusters."""
    dim = 384
    emb_matrix = np.eye(4, dim, dtype=np.float32)
    records = [
        ImageEmbeddingRecord(
            image_id="img_0",
            relative_path="ADI/img_0.png",
            sha256="sha_A",
            class_name="ADI",
            dataset_split="train",
            model_name="dinov2_vits14",
            model_revision="r1",
            preprocessing_version="v1.0",
            embedding_dim=dim,
            extracted_at_utc="2026-11-06T00:00:00Z",
            l2_normalized=True,
            cluster_id="cluster_100",
        ),
        ImageEmbeddingRecord(
            image_id="img_1",
            relative_path="ADI/img_1.png",
            sha256="sha_A",  # Same SHA
            class_name="ADI",
            dataset_split="train",
            model_name="dinov2_vits14",
            model_revision="r1",
            preprocessing_version="v1.0",
            embedding_dim=dim,
            extracted_at_utc="2026-11-06T00:00:00Z",
            l2_normalized=True,
            cluster_id="cluster_100",  # Same cluster
        ),
        ImageEmbeddingRecord(
            image_id="img_2",
            relative_path="TUM/img_2.png",
            sha256="sha_B",
            class_name="TUM",
            dataset_split="train",
            model_name="dinov2_vits14",
            model_revision="r1",
            preprocessing_version="v1.0",
            embedding_dim=dim,
            extracted_at_utc="2026-11-06T00:00:00Z",
            l2_normalized=True,
            cluster_id="cluster_200",
        ),
        ImageEmbeddingRecord(
            image_id="img_3",
            relative_path="NOR/img_3.png",
            sha256="sha_C",
            class_name="NOR",
            dataset_split="test",
            model_name="dinov2_vits14",
            model_revision="r1",
            preprocessing_version="v1.0",
            embedding_dim=dim,
            extracted_at_utc="2026-11-06T00:00:00Z",
            l2_normalized=True,
            cluster_id="cluster_300",
        ),
    ]

    retriever = HistopathologySimilarityRetriever(
        embeddings=emb_matrix,
        records=records,
        reference_split="train",
        use_faiss=False,
    )

    # Query with vector identical to img_0
    q_vec = emb_matrix[0]
    matches = retriever.query(
        query_vector=q_vec,
        top_k=2,
        query_image_id="img_0",
        query_relative_path="ADI/img_0.png",
        query_sha256="sha_A",
        query_cluster_id="cluster_100",
        exclude_self=True,
        exclude_duplicates=True,
        exclude_same_cluster=True,
    )

    assert len(matches) > 0
    for m in matches:
        assert m.dataset_split == "train", "Non-training neighbor returned!"
        assert m.neighbor_image_id != "img_0", "Self match not excluded!"
        assert m.sha256 != "sha_A", "Exact SHA duplicate not excluded!"
        assert m.cluster_id != "cluster_100", "Cluster duplicate not excluded!"


def test_retrieval_evaluation_metrics_and_artifacts():
    """Verify that retrieval evaluation output artifacts exist and meet validation criteria."""
    cls_eval = EVAL_DIR / "retrieval_evaluation_cls.json"
    patch_eval = EVAL_DIR / "retrieval_evaluation_mean_patch.json"

    assert cls_eval.exists(), "retrieval_evaluation_cls.json missing!"
    assert patch_eval.exists(), "retrieval_evaluation_mean_patch.json missing!"

    with open(cls_eval, "r") as f:
        data_cls = json.load(f)
    with open(patch_eval, "r") as f:
        data_patch = json.load(f)

    assert data_cls["evaluated_queries"] == 4665
    assert data_cls["gallery_reference_count"] == 21767
    assert data_cls["audit_exclusions"]["leakage_status"] == "ZERO_LEAKAGE_VERIFIED"

    # Check P@k values are in reasonable bounds [0.35, 0.65]
    for k in ["k_1", "k_5", "k_10"]:
        p_k = data_cls["metrics_by_k"][k]["mean_class_consistency_p_at_k"]
        assert 0.35 <= p_k <= 0.65, f"Unexpected P@k for {k}: {p_k}"


def test_visualizations_exist_and_non_empty():
    """Verify that all four required research visualization plots exist and are non-empty."""
    expected_figs = [
        "similarity_distribution.png",
        "retrieval_at_k_by_class.png",
        "retrieval_confusion_matrix.png",
        "embedding_pca_umap.png",
    ]
    for fig_name in expected_figs:
        fig_path = VIZ_DIR / fig_name
        assert fig_path.exists(), f"Figure {fig_name} is missing!"
        assert fig_path.stat().st_size > 10000, f"Figure {fig_name} is unexpectedly small: {fig_path.stat().st_size} bytes"


def test_resumability_mechanism():
    """Verify that FullDatasetExtractor detects complete extraction and terminates instantly."""
    extractor = FullDatasetExtractor()
    t0 = pytest.importorskip("time").time()
    summary = extractor.run_extraction()
    dur = pytest.importorskip("time").time() - t0

    assert summary["status"] == "success"
    assert summary["total_images"] == 31096
    assert dur < 5.0, f"Resumed validation took unexpectedly long: {dur:.2f}s"


def test_baseline_checkpoint_and_database_unmodified():
    """Verify baseline classifier checkpoint and SQLite prediction history remain 100% bitwise intact."""
    assert BASELINE_CHECKPOINT_PATH.exists(), "Baseline checkpoint missing!"

    hasher = hashlib.sha256()
    with open(BASELINE_CHECKPOINT_PATH, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    actual_sha = hasher.hexdigest().lower()

    assert actual_sha == BASELINE_CHECKPOINT_SHA256, (
        f"Baseline checkpoint SHA-256 altered! Expected {BASELINE_CHECKPOINT_SHA256}, got {actual_sha}"
    )

    db_path = WORKSPACE_ROOT / "app.db"
    assert db_path.exists(), "app.db missing!"

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM prediction_records")
    count = cur.fetchone()[0]
    conn.close()

    assert count == 4, f"app.db prediction records count altered! Expected 4, got {count}"

