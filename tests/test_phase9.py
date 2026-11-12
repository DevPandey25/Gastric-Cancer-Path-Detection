"""Test suite for Phase 9: Research Error Analysis and Class-Level Findings.

Validates:
1. Complete representation of all N=467 clean test samples.
2. Exact confusion matrix totals (310 correct + 157 errors = 467 total).
3. Class-level metrics mathematical validity (Precision, Recall, F1, Specificity).
4. Deterministic 48-exemplar gallery integrity (24 correct, 24 incorrect, 6 per class).
5. Ground-truth vs predicted label consistency for gallery items.
6. Cross-phase metadata validity (Phases 5, 6, 7, and 8 linkage).
7. Artifacts existence and non-emptiness (CSVs, JSONs, PNGs).
8. Invariant preservation (Historical CNN-RNN checkpoint SHA-256, Phase 4.4-8 caches).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
ERROR_DIR = WORKSPACE_ROOT / "research" / "error_analysis"
CANONICAL_CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]

BASELINE_CKPT = WORKSPACE_ROOT / "models" / "checkpoints" / "gastric_rnn_classifier.pt"
BASELINE_SHA256 = "65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8"
SUBSET_DIR = WORKSPACE_ROOT / "research" / "foundation_models" / "cache" / "subset10pct"
EXP_DIR = WORKSPACE_ROOT / "research" / "explainability"
ROBUSTNESS_DIR = WORKSPACE_ROOT / "research" / "robustness"
OOD_DIR = WORKSPACE_ROOT / "research" / "ood"
SPATIAL_DIR = WORKSPACE_ROOT / "research" / "spatial_analysis"


class TestPhase9PredictionCoverage:
    """Validates full representation of the N=467 held-out test split."""

    def test_summary_test_population(self):
        summary_path = ERROR_DIR / "error_analysis_summary.json"
        assert summary_path.exists(), f"Missing summary: {summary_path}"
        with open(summary_path) as f:
            data = json.load(f)

        meta = data["metadata"]
        assert meta["evaluation_population"] == 467
        assert meta["total_correct"] == 310
        assert meta["total_errors"] == 157
        assert meta["overall_accuracy"] == round(310 / 467, 4)

    def test_class_metrics_coverage_and_sums(self):
        metrics_path = ERROR_DIR / "class_metrics.csv"
        assert metrics_path.exists(), f"Missing metrics: {metrics_path}"
        df_metrics = pd.read_csv(metrics_path)

        # 8 classes + Macro + Weighted = 10 rows
        assert len(df_metrics) == 10
        class_rows = df_metrics[df_metrics["class_name"].isin(CANONICAL_CLASSES)]
        assert len(class_rows) == 8

        assert class_rows["support"].sum() == 467
        assert class_rows["tp"].sum() == 310
        assert class_rows["fn"].sum() == 157
        assert class_rows["fp"].sum() == 157

        for _, row in class_rows.iterrows():
            assert 0.0 <= row["precision"] <= 1.0
            assert 0.0 <= row["recall"] <= 1.0
            assert 0.0 <= row["f1_score"] <= 1.0
            assert 0.0 <= row["specificity"] <= 1.0


class TestPhase9ConfusionPairs:
    """Validates off-diagonal error pairs and percentages."""

    def test_confusion_pairs_sum_to_total_errors(self):
        pairs_path = ERROR_DIR / "confusion_pairs.csv"
        assert pairs_path.exists(), f"Missing confusion pairs: {pairs_path}"
        df_pairs = pd.read_csv(pairs_path)

        assert len(df_pairs) > 0
        assert df_pairs["count"].sum() == 157
        assert np.isclose(df_pairs["pct_of_all_errors"].sum(), 100.0, atol=0.2)

        # Check top confusion pair
        top_pair = df_pairs.iloc[0]
        assert top_pair["count"] == 9
        assert (top_pair["true_class"], top_pair["predicted_class"]) in [("ADI", "MUC"), ("NOR", "TUM")]


class TestPhase9GalleryIntegrity:
    """Validates the deterministic 48-image gallery selection and cross-phase flags."""

    @pytest.fixture
    def df_gallery(self) -> pd.DataFrame:
        manifest_path = ERROR_DIR / "error_analysis_manifest.csv"
        assert manifest_path.exists(), f"Missing manifest: {manifest_path}"
        return pd.read_csv(manifest_path)

    def test_gallery_dimensions_and_balance(self, df_gallery: pd.DataFrame):
        assert len(df_gallery) == 48
        assert (df_gallery["gallery_type"] == "correct").sum() == 24
        assert (df_gallery["gallery_type"] == "incorrect").sum() == 24

        # Exactly 3 correct and 3 incorrect per class
        for c in CANONICAL_CLASSES:
            sub = df_gallery[df_gallery["true_class"] == c]
            assert len(sub) == 6
            assert (sub["gallery_type"] == "correct").sum() == 3
            assert (sub["gallery_type"] == "incorrect").sum() == 3

    def test_gallery_label_consistency(self, df_gallery: pd.DataFrame):
        for _, row in df_gallery.iterrows():
            if row["gallery_type"] == "correct":
                assert row["is_correct"] is True or row["is_correct"] == 1
                assert row["true_class"] == row["predicted_class"]
            else:
                assert row["is_correct"] is False or row["is_correct"] == 0
                assert row["true_class"] != row["predicted_class"]

            # Confidence bounded
            assert 0.0 < row["confidence"] <= 1.0

            # Real image exists
            img_path = WORKSPACE_ROOT / "data" / "raw" / "HMU-GC-HE-30K" / "all_image" / row["relative_path"]
            assert img_path.exists(), f"Image does not exist: {img_path}"

    def test_cross_phase_linkages(self, df_gallery: pd.DataFrame):
        # All gallery items are from test set and evaluated in Phase 6
        assert df_gallery["in_phase6_robustness"].all()

        # At least one item overlaps with Phase 5 and Phase 8
        assert df_gallery["in_phase5_explainability"].sum() > 0
        assert df_gallery["in_phase8_spatial"].sum() > 0

        # Phase 7 Mahalanobis score present and finite
        assert df_gallery["mahalanobis_score"].notna().all()
        assert (df_gallery["mahalanobis_score"] > 0).all()


class TestPhase9Visualizations:
    """Validates that all 6 required publication figures exist and are valid PNGs."""

    def test_publication_plots_exist(self):
        plots = [
            ERROR_DIR / "confusion_matrix_counts.png",
            ERROR_DIR / "confusion_matrix_normalized.png",
            ERROR_DIR / "top_confusion_pairs.png",
            ERROR_DIR / "correct_prediction_gallery.png",
            ERROR_DIR / "incorrect_prediction_gallery.png",
            ERROR_DIR / "confidence_error_analysis.png",
        ]
        for p in plots:
            assert p.exists(), f"Plot does not exist: {p}"
            assert p.stat().st_size > 15_000, f"Plot unusually small: {p} ({p.stat().st_size} bytes)"


class TestPhase9Invariants:
    """Guarantees historical models, checkpoints, and prior phases remain untouched."""

    def test_historical_checkpoint_sha256(self):
        assert BASELINE_CKPT.exists(), "Historical checkpoint missing"
        hasher = hashlib.sha256()
        with open(BASELINE_CKPT, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        assert hasher.hexdigest() == BASELINE_SHA256, "Historical checkpoint modified!"

    def test_prior_phases_artifacts_intact(self):
        # Phase 4.4
        assert (SUBSET_DIR / "manifest_subset.csv").exists()
        assert (SUBSET_DIR / "evaluation" / "phikon_mean_patch_test_predictions.csv").exists()

        # Phase 5
        assert (EXP_DIR / "fitted_phikon_linear_probe.joblib").exists()

        # Phase 6
        assert (ROBUSTNESS_DIR / "metrics_robustness.json").exists()

        # Phase 7
        assert (OOD_DIR / "ood_evaluation_summary.json").exists()

        # Phase 8
        assert (SPATIAL_DIR / "spatial_summary.json").exists()

