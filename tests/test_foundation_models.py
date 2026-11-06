"""Automated unit and integration tests for Phase 4.0 Foundation Model Infrastructure."""

import shutil
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

from research.foundation_models.cache_manager import EmbeddingCacheManager
from research.foundation_models.config import FoundationModelConfig
from research.foundation_models.embedding_extractor import PathologyEmbeddingExtractor
from research.foundation_models.metadata import ImageEmbeddingRecord
from research.foundation_models.model_loader import load_foundation_encoder
from research.foundation_models.preprocessing import DeterministicPathologyPreprocessor
from research.foundation_models.retrieval import HistopathologySimilarityRetriever

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


def _create_synthetic_patch(path: Path, color=(120, 150, 180)) -> Path:
    """Helper creating a temporary 224x224 RGB image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (224, 224), color=color)
    img.save(path)
    return path


def test_deterministic_preprocessing(tmp_path: Path):
    """Verify that preprocessing is strictly deterministic and output shapes match expectations."""
    preprocessor = DeterministicPathologyPreprocessor(image_size=224)
    img_path = _create_synthetic_patch(tmp_path / "patch.png")

    t1 = preprocessor.preprocess_image(img_path)
    t2 = preprocessor.preprocess_image(img_path)

    assert t1.shape == (3, 224, 224), f"Unexpected preprocessed shape: {t1.shape}"
    assert torch.equal(t1, t2), "Preprocessing produced non-deterministic outputs!"


def test_model_interface_and_frozen_weights():
    """Verify encoder instantiation, output shape, and frozen gradient status."""
    model, embedding_dim = load_foundation_encoder("dinov2_vits14", device="cpu")
    assert embedding_dim == 384, f"Expected 384 embedding dim, got {embedding_dim}"

    # Verify all parameters are frozen
    for name, param in model.named_parameters():
        assert not param.requires_grad, f"Parameter {name} is not frozen!"

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == (2, 384), f"Expected [2, 384], got {out.shape}"
    assert not torch.isnan(out).any(), "Model output contains NaN!"
    assert not torch.isinf(out).any(), "Model output contains Inf!"


def test_embedding_extractor_finite_and_normalized(tmp_path: Path):
    """Verify extraction produces finite, L2-normalized embeddings."""
    config = FoundationModelConfig(
        model_name="dinov2_vits14",
        embedding_dim=384,
        device="cpu",
        normalize_l2=True,
    )
    extractor = PathologyEmbeddingExtractor(config=config)
    img_path = _create_synthetic_patch(tmp_path / "test_extract.png")

    emb = extractor.extract_patch_embedding(img_path)
    assert emb.shape == (384,), f"Expected shape (384,), got {emb.shape}"
    assert not np.isnan(emb).any(), "Embedding contains NaN!"
    assert not np.isinf(emb).any(), "Embedding contains Inf!"

    norm = np.linalg.norm(emb)
    assert np.isclose(norm, 1.0, atol=1e-3), f"Embedding norm {norm} is not 1.0!"


def test_cache_manager_round_trip(tmp_path: Path):
    """Verify cache save, load, and integrity verification."""
    cache_dir = tmp_path / "cache_test"
    mgr = EmbeddingCacheManager(cache_dir=cache_dir, model_name="dinov2_vits14")

    # Generate synthetic embeddings and metadata
    n_samples = 5
    embeddings = np.random.randn(n_samples, 384).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    records = [
        ImageEmbeddingRecord(
            image_id=f"patch_{i}",
            relative_path=f"ADI/patch_{i}.png",
            sha256=f"hash_{i}",
            class_name="ADI",
            dataset_split="train",
            model_name="dinov2_vits14",
            model_revision="test_rev",
            preprocessing_version="v1.0",
            embedding_dim=384,
            extracted_at_utc="2026-11-06T00:00:00Z",
            l2_normalized=True,
        )
        for i in range(n_samples)
    ]

    mgr.save_cache(embeddings, records, append=False)
    assert mgr.exists()

    loaded_emb, loaded_rec = mgr.load_cache()
    assert np.allclose(embeddings, loaded_emb)
    assert len(loaded_rec) == n_samples
    assert loaded_rec[0].relative_path == "ADI/patch_0.png"

    integrity = mgr.verify_cache_integrity()
    assert integrity["valid"]
    assert integrity["total_records"] == n_samples
    assert integrity["embedding_dimension"] == 384


def test_retrieval_self_and_duplicate_exclusion():
    """Verify nearest neighbor search correctly excludes query self and duplicate hashes."""
    dim = 384
    # Create 3 orthogonal reference vectors + 1 identical to query
    v_query = np.zeros(dim, dtype=np.float32)
    v_query[0] = 1.0

    v_dup = v_query.copy()  # Duplicate vector and hash
    v_other1 = np.zeros(dim, dtype=np.float32)
    v_other1[1] = 1.0
    v_other2 = np.zeros(dim, dtype=np.float32)
    v_other2[2] = 1.0

    emb_matrix = np.vstack([v_query, v_dup, v_other1, v_other2])
    records = [
        ImageEmbeddingRecord(
            image_id="img_query",
            relative_path="ADI/query.png",
            sha256="sha_query",
            class_name="ADI",
            dataset_split="train",
            model_name="dinov2_vits14",
            model_revision="r1",
            preprocessing_version="v1.0",
            embedding_dim=dim,
            extracted_at_utc="2026-11-06T00:00:00Z",
            l2_normalized=True,
        ),
        ImageEmbeddingRecord(
            image_id="img_duplicate",
            relative_path="ADI/dup.png",
            sha256="sha_query",  # Identical hash
            class_name="ADI",
            dataset_split="train",
            model_name="dinov2_vits14",
            model_revision="r1",
            preprocessing_version="v1.0",
            embedding_dim=dim,
            extracted_at_utc="2026-11-06T00:00:00Z",
            l2_normalized=True,
        ),
        ImageEmbeddingRecord(
            image_id="img_other1",
            relative_path="TUM/other1.png",
            sha256="sha_other1",
            class_name="TUM",
            dataset_split="train",
            model_name="dinov2_vits14",
            model_revision="r1",
            preprocessing_version="v1.0",
            embedding_dim=dim,
            extracted_at_utc="2026-11-06T00:00:00Z",
            l2_normalized=True,
        ),
        ImageEmbeddingRecord(
            image_id="img_other2",
            relative_path="LYM/other2.png",
            sha256="sha_other2",
            class_name="LYM",
            dataset_split="train",
            model_name="dinov2_vits14",
            model_revision="r1",
            preprocessing_version="v1.0",
            embedding_dim=dim,
            extracted_at_utc="2026-11-06T00:00:00Z",
            l2_normalized=True,
        ),
    ]

    retriever = HistopathologySimilarityRetriever(
        embeddings=emb_matrix,
        records=records,
        use_faiss=False,
    )

    matches = retriever.query(
        query_vector=v_query,
        top_k=3,
        query_image_id="img_query",
        query_relative_path="ADI/query.png",
        query_sha256="sha_query",
        exclude_self=True,
        exclude_duplicates=True,
    )

    returned_ids = [m.neighbor_image_id for m in matches]
    assert "img_query" not in returned_ids, "Self was not excluded!"
    assert "img_duplicate" not in returned_ids, "Duplicate hash was not excluded!"
    assert returned_ids[0] in ("img_other1", "img_other2"), "Unexpected neighbor returned!"


def test_retrieval_split_isolation():
    """Verify that retrieval strictly isolates candidate pools by split when configured."""
    dim = 384
    emb_matrix = np.eye(4, dim, dtype=np.float32)
    records = [
        ImageEmbeddingRecord(
            image_id=f"img_{i}",
            relative_path=f"path_{i}",
            sha256=f"hash_{i}",
            class_name="ADI",
            dataset_split="train" if i < 2 else "test",
            model_name="dinov2_vits14",
            model_revision="r1",
            preprocessing_version="v1.0",
            embedding_dim=dim,
            extracted_at_utc="2026-11-06T00:00:00Z",
            l2_normalized=True,
        )
        for i in range(4)
    ]

    # Retriever configured with reference_split='train'
    train_retriever = HistopathologySimilarityRetriever(
        embeddings=emb_matrix,
        records=records,
        reference_split="train",
        use_faiss=False,
    )

    assert train_retriever.num_items == 2
    # Query with a test vector
    q_vec = emb_matrix[3]
    matches = train_retriever.query(q_vec, top_k=2)

    for m in matches:
        assert m.dataset_split == "train", f"Split leakage! Found neighbor with split '{m.dataset_split}'"


def test_graceful_optional_dependency_handling():
    """Verify that unsupported models or missing libraries raise clean descriptive errors."""
    with pytest.raises(ValueError, match="Unsupported foundation model"):
        load_foundation_encoder("unsupported_model_xyz")


def test_fastapi_regression_without_foundation_models():
    """Verify that FastAPI application continues to start and respond without foundation model overhead."""
    from app.main import app, health

    assert app is not None
    data = health()
    assert data["status"] == "ok"
    assert "Gastric" in data["app"]

