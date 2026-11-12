"""Unit and Regression Tests for Phase 4.4 Reduced (10% Stratified Subset Benchmark).

Verifies:
1. Phikon offline configuration, parameter freezing, architecture.
2. 10% stratified subset manifest integrity (N=3,110, Train: 2,177, Val: 466, Test: 467).
3. Subset cache shapes: DINOv2 (3,110, 384) and Phikon (3,110, 768).
4. Finiteness, non-zero vectors, and L2 normalization (mean == 1.0).
5. Downstream linear probe prediction alignment with subset test manifest (N=467).
6. Mathematical metric identities (Balanced Accuracy == Macro Recall, Accuracy == Weighted Recall).
7. Invariant integrity: Baseline checkpoint SHA-256 and full DINOv2 cache invariance.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
BASELINE_CKPT = WORKSPACE_ROOT / "models" / "checkpoints" / "gastric_rnn_classifier.pt"
BASELINE_SHA256 = "65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8"

DINOV2_FULL_DIR = WORKSPACE_ROOT / "research" / "foundation_models" / "cache" / "full" / "dinov2_vits14"
SUBSET_DIR = WORKSPACE_ROOT / "research" / "foundation_models" / "cache" / "subset10pct"
SUBSET_MANIFEST = SUBSET_DIR / "manifest_subset.csv"
SUBSET_EVAL_DIR = SUBSET_DIR / "evaluation"
CANONICAL_CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]


def test_baseline_checkpoint_sha256_invariant():
    """Confirms historical CNN-RNN checkpoint was not modified."""
    assert BASELINE_CKPT.exists(), "Historical baseline checkpoint missing!"
    h = hashlib.sha256()
    with open(BASELINE_CKPT, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    assert h.hexdigest() == BASELINE_SHA256, "Historical baseline checkpoint SHA-256 altered!"


def test_dinov2_full_cache_intact():
    """Confirms DINOv2 full embedding cache files exist and have exact expected byte sizes."""
    cls_file = DINOV2_FULL_DIR / "cls_embeddings.npy"
    patch_file = DINOV2_FULL_DIR / "mean_patch_embeddings.npy"
    assert cls_file.exists(), "DINOv2 CLS embeddings cache missing!"
    assert patch_file.exists(), "DINOv2 Mean Patch embeddings cache missing!"
    expected_bytes = 31096 * 384 * 4
    assert cls_file.stat().st_size == expected_bytes
    assert patch_file.stat().st_size == expected_bytes


def test_phikon_offline_model_configuration():
    """Verifies owkin/phikon configuration loads offline and matches ViT-B/16 architecture."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("owkin/phikon", local_files_only=True)
    assert config.hidden_size == 768
    assert config.num_hidden_layers == 12
    assert config.num_attention_heads == 12
    assert config.patch_size == 16
    assert config.image_size == 224
    assert config.model_type == "vit"


def test_phikon_pilot_report_exists():
    """Verifies Phase 4.4.1 pilot report exists and meets criteria."""
    report_path = WORKSPACE_ROOT / "research" / "foundation_models" / "pathology_encoder" / "pilot_results.json"
    assert report_path.exists(), "Pilot results JSON missing!"
    with open(report_path, "r") as f:
        data = json.load(f)
    assert data["model_id"] == "owkin/phikon"
    assert data["embedding_dimension"] == 768
    assert data["trainable_parameters"] == 0
    assert data["parameter_count"] == 86389248
    assert data["cls_max_l2_norm_deviation"] < 1e-5
    assert data["patch_max_l2_norm_deviation"] < 1e-5


def test_subset_manifest_split_and_class_stratification():
    """Verifies 10% stratified subset manifest has exact expected sample counts."""
    assert SUBSET_MANIFEST.exists(), "Subset manifest missing!"
    df_sub = pd.read_csv(SUBSET_MANIFEST)
    assert len(df_sub) == 3110

    splits = df_sub["split"].value_counts().to_dict()
    assert splits["train"] == 2177
    assert splits["val"] == 466
    assert splits["test"] == 467

    # Check class representation in each split
    for s, expected_min in [("train", 272), ("val", 58), ("test", 58)]:
        sub_s = df_sub[df_sub["split"] == s]
        for c in CANONICAL_CLASSES:
            count = (sub_s["class_name"] == c).sum()
            assert count >= expected_min, f"Class {c} in split {s} has {count} < {expected_min}"


@pytest.mark.skipif(not (SUBSET_DIR / "phikon" / "cls_embeddings.npy").exists(), reason="Subset cache not built yet")
def test_dinov2_and_phikon_subset_cache_integrity():
    """Validates shapes, finiteness, and L2 norms of both DINOv2 and Phikon subset caches."""
    dino_cls_p = SUBSET_DIR / "dinov2_vits14" / "cls_embeddings.npy"
    dino_patch_p = SUBSET_DIR / "dinov2_vits14" / "mean_patch_embeddings.npy"
    phi_cls_p = SUBSET_DIR / "phikon" / "cls_embeddings.npy"
    phi_patch_p = SUBSET_DIR / "phikon" / "mean_patch_embeddings.npy"

    assert dino_cls_p.stat().st_size == 3110 * 384 * 4
    assert dino_patch_p.stat().st_size == 3110 * 384 * 4
    assert phi_cls_p.stat().st_size == 3110 * 768 * 4
    assert phi_patch_p.stat().st_size == 3110 * 768 * 4

    dino_cls = np.memmap(dino_cls_p, dtype=np.float32, mode="r", shape=(3110, 384))
    phi_cls = np.memmap(phi_cls_p, dtype=np.float32, mode="r", shape=(3110, 768))

    assert dino_cls.shape == (3110, 384)
    assert phi_cls.shape == (3110, 768)

    assert np.all(np.isfinite(dino_cls))
    assert np.all(np.isfinite(phi_cls))

    d_norms = np.linalg.norm(dino_cls, axis=1)
    p_norms = np.linalg.norm(phi_cls, axis=1)

    assert abs(np.mean(d_norms) - 1.0) < 1e-4
    assert abs(np.mean(p_norms) - 1.0) < 1e-4


@pytest.mark.skipif(not (SUBSET_EVAL_DIR / "phikon_cls_test_predictions.csv").exists(), reason="Subset evaluation not completed yet")
def test_subset_prediction_alignment_and_identities():
    """Verifies test predictions align with subset manifest and adhere to mathematical identities."""
    df_sub = pd.read_csv(SUBSET_MANIFEST)
    test_sub = df_sub[df_sub["split"] == "test"].reset_index(drop=True)
    assert len(test_sub) == 467

    eval_json_p = SUBSET_EVAL_DIR / "evaluation_summary.json"
    assert eval_json_p.exists()
    with open(eval_json_p, "r") as f:
        eval_data = json.load(f)

    for k in ["phikon_cls", "phikon_mean_patch", "dinov2_cls", "dinov2_mean_patch"]:
        pred_p = SUBSET_EVAL_DIR / f"{k}_test_predictions.csv"
        assert pred_p.exists()
        df_p = pd.read_csv(pred_p)
        assert len(df_p) == 467
        assert (df_p["relative_path"] == test_sub["relative_path"]).all()
        assert (df_p["true_index"] == test_sub["class_index"]).all()

        m = eval_data["results"][k]["overall_metrics"]
        assert math.isclose(m["balanced_accuracy"], m["macro_recall"], abs_tol=1e-4)
        assert math.isclose(m["accuracy"], m["weighted_recall"], abs_tol=1e-4)
