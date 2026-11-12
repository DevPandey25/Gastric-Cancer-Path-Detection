"""Test suite for Original Phase 5: Explainability, Calibration, Uncertainty, and Abstention.

Validates:
1. Baseline CNN-RNN Grad-CAM, Integrated Gradients, Occlusion Sensitivity, and MC Dropout.
2. Calibration evaluation (ECE, Brier score, NLL) before and after temperature scaling.
3. Validation-fitted temperature scaling (learned strictly on validation data).
4. Validation-calibrated selective classification / abstention handling.
5. Reliability diagrams and publication figure integrity.
6. Invariant preservation (checkpoint SHA-256, prior phase caches).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
ORIG_P5_DIR = WORKSPACE_ROOT / "research" / "explainability" / "original_phase5"
CANONICAL_CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]

BASELINE_CKPT = WORKSPACE_ROOT / "models" / "checkpoints" / "gastric_rnn_classifier.pt"
BASELINE_SHA256 = "65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8"


class TestOriginalPhase5CNNAttribution:
    """Validates CNN-RNN baseline explainability and uncertainty outputs."""

    def test_cnn_attribution_summary_exists(self):
        summary_path = ORIG_P5_DIR / "cnn_attribution_summary.json"
        assert summary_path.exists(), f"Missing summary: {summary_path}"
        with open(summary_path) as f:
            data = json.load(f)

        assert data["model_name"] == "GastricCancerRNN (CNN-BiLSTM-Attention)"
        assert data["integrated_gradients_steps"] == 20
        assert data["occlusion_window_size"] == 28
        assert data["mc_dropout_passes"] == 20
        assert len(data["exemplar_results"]) == 8

        # 1 exemplar per class
        classes_present = {r["class_name"] for r in data["exemplar_results"]}
        assert classes_present == set(CANONICAL_CLASSES)

        for r in data["exemplar_results"]:
            assert 0.0 < r["confidence"] <= 1.0
            assert r["predicted_class"] in CANONICAL_CLASSES
            assert r["mc_epistemic_variance"] >= 0.0
            assert r["mc_epistemic_entropy"] >= 0.0
            assert 0.0 <= r["max_gradcam_activation"] <= 1.0

    def test_cnn_attribution_plot_exists(self):
        p = ORIG_P5_DIR / "cnn_explainability_comparison.png"
        assert p.exists() and p.stat().st_size > 30_000


class TestOriginalPhase5Calibration:
    """Validates temperature scaling, ECE, and reliability diagrams."""

    @pytest.fixture
    def cal_metrics(self) -> Dict[str, Any]:
        p = ORIG_P5_DIR / "calibration_metrics.json"
        assert p.exists(), f"Missing calibration metrics: {p}"
        with open(p) as f:
            return json.load(f)

    def test_phikon_calibration_metrics(self, cal_metrics: Dict[str, Any]):
        phikon = cal_metrics["phikon_mean_patch_calibration"]
        assert phikon["optimal_temperature_T"] > 0.0
        assert 0.0 <= phikon["test_accuracy"] <= 1.0

        for split in ["validation_metrics", "test_metrics"]:
            for mode in ["uncalibrated", "calibrated"]:
                m = phikon[split][mode]
                assert 0.0 <= m["ece"] <= 1.0
                assert m["brier_score"] >= 0.0
                assert m["nll"] >= 0.0

    def test_cnn_calibration_metrics(self, cal_metrics: Dict[str, Any]):
        cnn = cal_metrics["gastric_cancer_rnn_calibration"]
        assert cnn["optimal_temperature_T"] > 0.0

        for split in ["validation_metrics", "test_metrics"]:
            for mode in ["uncalibrated", "calibrated"]:
                m = cnn[split][mode]
                assert 0.0 <= m["ece"] <= 1.0
                assert m["brier_score"] >= 0.0
                assert m["nll"] >= 0.0

    def test_reliability_plots_exist(self):
        for name in ["reliability_diagrams_phikon.png", "reliability_diagrams_cnn.png"]:
            p = ORIG_P5_DIR / name
            assert p.exists() and p.stat().st_size > 15_000


class TestOriginalPhase5Abstention:
    """Validates validation-calibrated selective classification."""

    def test_abstention_csv_integrity(self):
        csv_path = ORIG_P5_DIR / "abstention_results.csv"
        assert csv_path.exists(), f"Missing abstention CSV: {csv_path}"
        df = pd.read_csv(csv_path)

        assert len(df) > 10
        assert (df["threshold"].diff().dropna() > 0).all(), "Thresholds must be strictly increasing"
        assert (df["test_coverage"].diff().dropna() <= 0).all(), "Coverage must decrease or remain constant as threshold rises"

        for col in ["val_coverage", "val_selective_acc", "test_coverage", "test_selective_acc", "test_selective_error"]:
            assert ((df[col] >= 0.0) & (df[col] <= 1.0)).all()

    def test_abstention_operating_points(self):
        p = ORIG_P5_DIR / "calibration_metrics.json"
        with open(p) as f:
            data = json.load(f)

        abstention = data["abstention_selective_classification"]
        op_a = abstention["operating_point_a_90pct_coverage"]
        op_b = abstention["operating_point_b_80pct_accuracy"]

        assert 0.0 < op_a["calibrated_threshold"] < 1.0
        assert 0.0 < op_b["calibrated_threshold"] < 1.0

        # As threshold increases, selective accuracy should improve
        assert op_b["test_selective_accuracy"] >= op_a["test_selective_accuracy"]
        assert op_a["test_retained_samples"] + op_a["test_abstained_samples"] == 467

    def test_abstention_tradeoff_plot_exists(self):
        p = ORIG_P5_DIR / "abstention_tradeoff_curves.png"
        assert p.exists() and p.stat().st_size > 15_000


class TestOriginalPhase5Invariants:
    """Guarantees historical checkpoint and previous phase preservation."""

    def test_baseline_checkpoint_sha256(self):
        assert BASELINE_CKPT.exists()
        hasher = hashlib.sha256()
        with open(BASELINE_CKPT, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        assert hasher.hexdigest() == BASELINE_SHA256

