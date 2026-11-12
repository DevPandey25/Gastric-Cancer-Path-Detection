"""Test suite for Phase 8: Intra-Patch Spatial Representation Analysis.

Validates:
1. ViT-B/16 architecture verification (224x224, patch 16x16, 14x14=196 tokens, d=768).
2. Deterministic spatial cohort selection (N=96, 12 per canonical class, test split).
3. 2x2 regional quadrant partitioning (TL, TR, BL, BR, 49 tokens each, partition sum=196).
4. Spatial descriptors validity (finite, bounded cosine similarities, non-negative norms).
5. Output artifacts integrity (manifest, CSV, JSON, summary, publication plots).
6. Invariant preservation (Phase 4.4, 5, 6, 7 integrity, historical checkpoint SHA-256).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
SPATIAL_DIR = WORKSPACE_ROOT / "research" / "spatial_analysis"
CANONICAL_CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]

BASELINE_CKPT = WORKSPACE_ROOT / "models" / "checkpoints" / "gastric_rnn_classifier.pt"
BASELINE_SHA256 = "65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8"
SUBSET_DIR = WORKSPACE_ROOT / "research" / "foundation_models" / "cache" / "subset10pct"
EXP_DIR = WORKSPACE_ROOT / "research" / "explainability"
ROBUSTNESS_DIR = WORKSPACE_ROOT / "research" / "robustness"
OOD_DIR = WORKSPACE_ROOT / "research" / "ood"


class TestPhase8Architecture:
    """Validates ViT-B/16 spatial token structure and grid partitioning."""

    def test_spatial_config_exists_and_valid(self):
        cfg_path = SPATIAL_DIR / "config.json"
        assert cfg_path.exists(), f"Missing config: {cfg_path}"
        with open(cfg_path) as f:
            cfg = json.load(f)

        assert "owkin/phikon" in cfg["model_name"]
        assert cfg["sample_selection"]["total_samples"] == 96
        assert cfg["sample_selection"]["samples_per_class"] == 12
        assert cfg["sample_selection"]["seed"] == 42
        assert set(cfg["sample_selection"]["classes"]) == set(CANONICAL_CLASSES)

    def test_summary_architecture_dimensions(self):
        summary_path = SPATIAL_DIR / "spatial_summary.json"
        assert summary_path.exists(), f"Missing summary: {summary_path}"
        with open(summary_path) as f:
            data = json.load(f)

        arch = data["metadata"]["model_architecture"]
        assert arch["image_size"] == 224
        assert arch["patch_size"] == 16
        assert arch["hidden_size"] == 768
        assert arch["grid_dimensions"] == [14, 14]
        assert arch["num_spatial_tokens"] == 196
        assert arch["total_tokens_with_cls"] == 197

        # Quadrant partition validation: 4 quadrants x 49 tokens = 196 tokens
        quads = arch["quadrant_partition"]
        assert len(quads) == 4
        assert set(quads.keys()) == {"TL", "TR", "BL", "BR"}
        total_tokens = sum(q["num_tokens"] for q in quads.values())
        assert total_tokens == 196


class TestPhase8CohortManifest:
    """Validates the deterministic 96-image held-out test cohort."""

    @pytest.fixture
    def df_manifest_96(self) -> pd.DataFrame:
        manifest_path = SPATIAL_DIR / "spatial_subset_manifest_96.csv"
        assert manifest_path.exists(), f"Missing manifest: {manifest_path}"
        return pd.read_csv(manifest_path)

    def test_cohort_size_and_balance(self, df_manifest_96: pd.DataFrame):
        assert len(df_manifest_96) == 96
        counts = df_manifest_96["class_name"].value_counts()
        for c in CANONICAL_CLASSES:
            assert counts[c] == 12, f"Class {c} has {counts.get(c, 0)} samples, expected 12"

    def test_cohort_split_assignment(self, df_manifest_96: pd.DataFrame):
        splits = df_manifest_96["split"].unique()
        assert list(splits) == ["test"], f"All samples must be from 'test' split, found {splits}"

    def test_sample_ids_and_paths_unique(self, df_manifest_96: pd.DataFrame):
        assert df_manifest_96["spatial_sample_id"].nunique() == 96
        assert df_manifest_96["relative_path"].nunique() == 96
        for path in df_manifest_96["relative_path"]:
            full_path = WORKSPACE_ROOT / "data" / "raw" / "HMU-GC-HE-30K" / "all_image" / path
            assert full_path.exists(), f"Image path does not exist: {full_path}"


class TestPhase8SpatialDescriptors:
    """Validates computed within-patch spatial descriptors and statistics."""

    @pytest.fixture
    def df_descriptors(self) -> pd.DataFrame:
        csv_path = SPATIAL_DIR / "spatial_descriptors.csv"
        assert csv_path.exists(), f"Missing descriptors: {csv_path}"
        return pd.read_csv(csv_path)

    def test_descriptors_completeness_and_finite(self, df_descriptors: pd.DataFrame):
        assert len(df_descriptors) == 96
        required_cols = [
            "token_norm_mean", "token_norm_std", "token_norm_dispersion",
            "regional_cos_sim_mean", "regional_l2_dist_mean", "regional_dispersion",
            "cos_sim_cls_vs_global_mean", "cos_sim_global_mean_vs_agg_regional",
            "predicted_class", "confidence", "is_correct",
            "att_frac_abs_TL", "att_frac_abs_TR", "att_frac_abs_BL", "att_frac_abs_BR"
        ]
        for col in required_cols:
            assert col in df_descriptors.columns, f"Missing column: {col}"
            assert not df_descriptors[col].isna().any(), f"NaN values in {col}"
            if pd.api.types.is_numeric_dtype(df_descriptors[col]):
                assert np.all(np.isfinite(df_descriptors[col])), f"Non-finite values in {col}"

    def test_cosine_similarity_bounds(self, df_descriptors: pd.DataFrame):
        cos_cols = [c for c in df_descriptors.columns if "cos_sim" in c]
        assert len(cos_cols) >= 8  # Pairwise regional + cross-representation
        for c in cos_cols:
            assert (df_descriptors[c] >= -1.0).all() and (df_descriptors[c] <= 1.0).all(), (
                f"Cosine similarity in {c} outside [-1.0, 1.0]"
            )

    def test_token_norms_strictly_positive(self, df_descriptors: pd.DataFrame):
        assert (df_descriptors["token_norm_mean"] > 0).all()
        assert (df_descriptors["token_norm_min"] > 0).all()
        assert (df_descriptors["token_norm_max"] > df_descriptors["token_norm_min"]).all()
        assert (df_descriptors["token_norm_dispersion"] > 0).all()

    def test_regional_attribution_mass_sum(self, df_descriptors: pd.DataFrame):
        att_sum = (
            df_descriptors["att_frac_abs_TL"]
            + df_descriptors["att_frac_abs_TR"]
            + df_descriptors["att_frac_abs_BL"]
            + df_descriptors["att_frac_abs_BR"]
        )
        # Quadrant absolute fractions sum to approximately 1.0
        assert np.allclose(att_sum, 1.0, atol=0.05), "Quadrant attribution fractions should sum to ~1.0"


class TestPhase8Artifacts:
    """Validates publication figures and JSON summary integrity."""

    def test_publication_plots_exist(self):
        plots = [
            SPATIAL_DIR / "spatial_analysis_contact_sheet.png",
            SPATIAL_DIR / "regional_similarity_by_class.png",
            SPATIAL_DIR / "representation_similarity_plot.png",
        ]
        for p in plots:
            assert p.exists(), f"Plot does not exist: {p}"
            assert p.stat().st_size > 15_000, f"Plot file unusually small: {p} ({p.stat().st_size} bytes)"

    def test_summary_json_contains_all_classes(self):
        summary_path = SPATIAL_DIR / "spatial_summary.json"
        with open(summary_path) as f:
            data = json.load(f)

        assert "overall" in data
        assert "class_summary" in data
        assert len(data["class_summary"]) == 8
        for c in CANONICAL_CLASSES:
            assert c in data["class_summary"]
            c_data = data["class_summary"][c]
            assert c_data["support"] == 12
            assert 0.0 <= c_data["accuracy"] <= 1.0
            assert c_data["token_norm_mean_mean"] > 0.0
            assert c_data["regional_cos_sim_mean_mean"] > 0.5


class TestPhase8Invariants:
    """Guarantees that historical models, checkpoints, and prior phases remain untouched."""

    def test_historical_checkpoint_sha256(self):
        assert BASELINE_CKPT.exists(), "Historical checkpoint missing"
        hasher = hashlib.sha256()
        with open(BASELINE_CKPT, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        assert hasher.hexdigest() == BASELINE_SHA256, "Historical checkpoint was modified!"

    def test_phase4_4_cache_intact(self):
        assert (SUBSET_DIR / "manifest_subset.csv").exists()
        assert (SUBSET_DIR / "phikon" / "mean_patch_embeddings.npy").stat().st_size == 3110 * 768 * 4
        assert (SUBSET_DIR / "evaluation" / "evaluation_summary.json").exists()

    def test_phase5_phase6_phase7_artifacts_intact(self):
        assert (EXP_DIR / "fitted_phikon_linear_probe.joblib").exists()
        assert (EXP_DIR / "attributions_grid_32x14x14.npy").exists()

        assert (ROBUSTNESS_DIR / "metrics_robustness.json").exists()
        assert (ROBUSTNESS_DIR / "predictions_robustness.csv").exists()
        assert len(pd.read_csv(ROBUSTNESS_DIR / "predictions_robustness.csv")) == 2335

        assert (OOD_DIR / "ood_evaluation_summary.json").exists()
        assert (OOD_DIR / "class_means.npy").exists()
