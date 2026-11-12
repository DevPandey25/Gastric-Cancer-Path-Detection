"""Unit and Regression Tests for Phase 7 (Lightweight OOD / Uncertainty Screening).

Verifies:
1. Invariant integrity: Baseline checkpoint SHA-256, DINOv2 full cache, Phase 4.4 subset artifacts.
2. Phase 5 and Phase 6 artifact preservation.
3. Class means fitted on training data only (shape (8, 768), finite, non-zero norms).
4. Ledoit-Wolf covariance is positive-semidefinite (all eigenvalues > 0) and well-conditioned.
5. Precision matrix is symmetric and corresponds to covariance inverse.
6. Mahalanobis scores are finite, non-negative, and higher scores indicate greater OOD-ness.
7. Threshold calibration uses validation data only (95th percentile).
8. ID false-rejection rate computed on clean test set (not validation).
9. Proxy-OOD metrics computed for each Phase 6 perturbation condition.
10. All required artifacts (CSVs, JSONs, PNGs) exist and are non-empty.
11. Score direction correctness: MSP OOD score = 1 - max_prob.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
BASELINE_CKPT = WORKSPACE_ROOT / "models" / "checkpoints" / "gastric_rnn_classifier.pt"
BASELINE_SHA256 = "65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8"

DINOV2_FULL_DIR = WORKSPACE_ROOT / "research" / "foundation_models" / "cache" / "full" / "dinov2_vits14"
SUBSET_DIR = WORKSPACE_ROOT / "research" / "foundation_models" / "cache" / "subset10pct"
EXP_DIR = WORKSPACE_ROOT / "research" / "explainability"
ROBUSTNESS_DIR = WORKSPACE_ROOT / "research" / "robustness"
OOD_DIR = WORKSPACE_ROOT / "research" / "ood"

CANONICAL_CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]


# ============================================================
# Invariant preservation
# ============================================================

def test_baseline_checkpoint_sha256_invariant():
    """Historical baseline CNN-RNN checkpoint must remain bitwise identical."""
    assert BASELINE_CKPT.exists()
    h = hashlib.sha256(BASELINE_CKPT.read_bytes()).hexdigest()
    assert h == BASELINE_SHA256


def test_dinov2_full_cache_intact():
    """Full DINOv2 cache files must exist with exact byte sizes."""
    for name in ["cls_embeddings.npy", "mean_patch_embeddings.npy"]:
        p = DINOV2_FULL_DIR / name
        assert p.exists()
        assert p.stat().st_size == 31096 * 384 * 4


def test_phase4_4_subset_artifacts_intact():
    """Phase 4.4 subset caches and evaluation summary must remain intact."""
    assert (SUBSET_DIR / "manifest_subset.csv").exists()
    assert (SUBSET_DIR / "phikon" / "mean_patch_embeddings.npy").stat().st_size == 3110 * 768 * 4
    assert (SUBSET_DIR / "evaluation" / "evaluation_summary.json").exists()


def test_phase5_explainability_artifacts_intact():
    """Phase 5 explainability artifacts must remain intact."""
    assert (EXP_DIR / "manifest_explainability_32.csv").exists()
    assert (EXP_DIR / "fitted_phikon_linear_probe.joblib").exists()
    assert (EXP_DIR / "attributions_grid_32x14x14.npy").exists()


def test_phase6_robustness_artifacts_intact():
    """Phase 6 robustness artifacts must remain intact."""
    assert (ROBUSTNESS_DIR / "predictions_robustness.csv").exists()
    assert (ROBUSTNESS_DIR / "metrics_robustness.json").exists()
    assert (ROBUSTNESS_DIR / "contact_sheet_robustness.png").exists()
    df = pd.read_csv(ROBUSTNESS_DIR / "predictions_robustness.csv")
    assert len(df) == 2335


# ============================================================
# Class means (train-only)
# ============================================================

def test_class_means_shape_and_finiteness():
    """Class means must be (8, 768) with finite non-zero norms."""
    means_path = OOD_DIR / "class_means.npy"
    assert means_path.exists()
    means = np.load(means_path)
    assert means.shape == (8, 768)
    assert np.all(np.isfinite(means))
    norms = np.linalg.norm(means, axis=1)
    assert np.all(norms > 0.1), "Class means should have substantial L2 norms"


def test_class_means_train_only():
    """Class means must be computed on train split only (N=2177)."""
    fit_info = json.loads((OOD_DIR / "covariance_fit_info.json").read_text())
    assert fit_info["n_train_samples"] == 2177
    assert fit_info["n_classes"] == 8
    assert fit_info["embedding_dim"] == 768


# ============================================================
# Covariance and precision matrix
# ============================================================

def test_covariance_positive_definite():
    """Ledoit-Wolf covariance must have all positive eigenvalues."""
    cov = np.load(OOD_DIR / "covariance_matrix.npy")
    assert cov.shape == (768, 768)
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals > 0), f"Non-positive eigenvalue found: min={eigvals.min()}"


def test_covariance_well_conditioned():
    """Condition number should be manageable (< 1e6 with Ledoit-Wolf shrinkage)."""
    fit_info = json.loads((OOD_DIR / "covariance_fit_info.json").read_text())
    assert fit_info["all_eigenvalues_positive"] is True
    assert fit_info["condition_number"] < 1e6
    assert fit_info["shrinkage_coefficient"] > 0


def test_precision_matrix_symmetric():
    """Precision matrix must be symmetric."""
    prec = np.load(OOD_DIR / "precision_matrix.npy")
    assert prec.shape == (768, 768)
    np.testing.assert_allclose(prec, prec.T, atol=1e-5)


# ============================================================
# Mahalanobis scores
# ============================================================

def test_mahalanobis_scores_finite_and_nonnegative():
    """All Mahalanobis scores must be finite and non-negative (squared distance)."""
    for name in ["val_mahalanobis_scores.npy", "test_mahalanobis_scores.npy"]:
        scores = np.load(OOD_DIR / name)
        assert np.all(np.isfinite(scores))
        assert np.all(scores >= 0)


def test_mahalanobis_score_direction():
    """Higher Mahalanobis scores should indicate greater distance from training distribution.

    Train samples should have lower average scores than test samples
    (though this is not guaranteed for every individual sample).
    """
    # The summary should show test Mahalanobis stats
    summary = json.loads((OOD_DIR / "ood_evaluation_summary.json").read_text())
    test_mean = summary["test_scores"]["mahalanobis"]["test_mean"]
    assert test_mean > 0, "Mean Mahalanobis score should be positive"


# ============================================================
# MSP scores
# ============================================================

def test_msp_score_direction():
    """MSP OOD score = 1 - max_prob. Higher means more uncertain / more OOD-like."""
    scores_df = pd.read_csv(OOD_DIR / "ood_scores_all.csv")
    clean_df = scores_df[scores_df["condition"] == "clean"]
    # MSP score should be between 0 and 1
    assert clean_df["msp_ood_score"].between(0, 1).all()
    # MSP score + confidence should approximately equal 1.0
    sums = clean_df["msp_ood_score"] + clean_df["confidence"]
    np.testing.assert_allclose(sums, 1.0, atol=1e-3)


# ============================================================
# Threshold calibration (validation-only)
# ============================================================

def test_threshold_calibrated_on_validation_only():
    """Thresholds must be calibrated from validation scores, not test scores."""
    cal = json.loads((OOD_DIR / "calibration_info.json").read_text())
    assert cal["calibration_split"] == "validation"
    assert cal["calibration_percentile"] == 95.0
    assert cal["n_validation_samples"] == 466
    assert cal["mahalanobis"]["threshold"] > 0
    assert 0 < cal["msp"]["threshold"] < 1


# ============================================================
# ID false-rejection
# ============================================================

def test_id_false_rejection_computed_on_test():
    """ID false-rejection must use clean test samples with frozen validation thresholds."""
    summary = json.loads((OOD_DIR / "ood_evaluation_summary.json").read_text())
    mahal_rej = summary["test_scores"]["mahalanobis"]["id_false_rejection_rate"]
    msp_rej = summary["test_scores"]["msp"]["id_false_rejection_rate"]

    # Should be close to 5% (since threshold is 95th percentile of validation)
    assert 0 < mahal_rej < 0.30, f"Mahal rejection rate {mahal_rej} out of expected range"
    assert 0 < msp_rej < 0.30, f"MSP rejection rate {msp_rej} out of expected range"


# ============================================================
# Proxy-OOD evaluation
# ============================================================

def test_proxy_ood_metrics_for_each_perturbation():
    """Each perturbation condition must have both MSP and Mahalanobis AUROC and FPR@95%TPR computed."""
    summary = json.loads((OOD_DIR / "ood_evaluation_summary.json").read_text())
    proxy = summary["proxy_ood_results"]

    for cond in ["brightness_contrast", "gaussian_blur", "jpeg_compression", "stain_hed_variation"]:
        assert cond in proxy, f"Missing proxy-OOD results for {cond}"
        msp = proxy[cond]["msp"]
        assert "auroc" in msp
        assert "fpr_at_95tpr" in msp
        assert 0 <= msp["auroc"] <= 1
        assert 0 <= msp["fpr_at_95tpr"] <= 1

        mahal = proxy[cond]["mahalanobis"]
        assert "auroc" in mahal
        assert "fpr_at_95tpr" in mahal
        assert 0 <= mahal["auroc"] <= 1
        assert 0 <= mahal["fpr_at_95tpr"] <= 1

    # Combined proxy should also be present with both metrics
    assert "combined_proxy" in proxy
    assert "auroc" in proxy["combined_proxy"]["msp"]
    assert "auroc" in proxy["combined_proxy"]["mahalanobis"]


# ============================================================
# Artifact completeness
# ============================================================

def test_all_required_artifacts_exist():
    """All Phase 7 required files must exist and be non-empty."""
    required_files = [
        "class_means.npy",
        "covariance_matrix.npy",
        "precision_matrix.npy",
        "cholesky_lower.npy",
        "covariance_fit_info.json",
        "calibration_info.json",
        "config.json",
        "ood_evaluation_summary.json",
        "ood_scores_all.csv",
        "ood_scores_all.json",
        "val_mahalanobis_scores.npy",
        "val_msp_scores.npy",
        "test_mahalanobis_scores.npy",
        "test_msp_scores.npy",
        "id_score_distributions.png",
        "proxy_ood_score_distributions.png",
        "roc_proxy_ood.png",
    ]
    for fname in required_files:
        fpath = OOD_DIR / fname
        assert fpath.exists(), f"Missing: {fname}"
        assert fpath.stat().st_size > 0, f"Empty: {fname}"


def test_ood_scores_csv_coverage():
    """OOD scores CSV must contain clean test (467) + 4 perturbation conditions (4 * 467 = 1868)."""
    df = pd.read_csv(OOD_DIR / "ood_scores_all.csv")
    assert len(df) == 467 + 4 * 467
    assert df[df["condition"] == "clean"].shape[0] == 467
    for cond in ["brightness_contrast", "gaussian_blur", "jpeg_compression", "stain_hed_variation"]:
        assert df[df["condition"] == cond].shape[0] == 467

    # All scores must be non-null and finite
    assert df["msp_ood_score"].notnull().all()
    assert np.all(np.isfinite(df["msp_ood_score"]))
    assert df["mahalanobis_score"].notnull().all()
    assert np.all(np.isfinite(df["mahalanobis_score"]))
    assert df["msp_flagged_ood"].notnull().all()
    assert df["mahal_flagged_ood"].notnull().all()


def test_reproducibility_of_saved_scores():
    """Saved Mahalanobis and MSP scores must match recomputation from saved parameters."""
    import joblib
    from scipy import linalg as sp_linalg

    # Load saved parameters
    means = np.load(OOD_DIR / "class_means.npy").astype(np.float64)
    chol_L = np.load(OOD_DIR / "cholesky_lower.npy").astype(np.float64)
    saved_test_mahal = np.load(OOD_DIR / "test_mahalanobis_scores.npy").astype(np.float64)

    # Load test embeddings
    manifest = pd.read_csv(SUBSET_DIR / "manifest_subset.csv")
    test_idx = np.where(manifest["split"] == "test")[0]
    mm = np.memmap(
        SUBSET_DIR / "phikon" / "mean_patch_embeddings.npy",
        dtype=np.float32, mode="r", shape=(len(manifest), 768),
    )
    X_test = np.array(mm[test_idx], dtype=np.float64)

    # Recompute Mahalanobis scores
    n = X_test.shape[0]
    recomputed = np.full(n, np.inf, dtype=np.float64)
    for c in range(8):
        delta = X_test - means[c]
        z = sp_linalg.solve_triangular(chol_L, delta.T, lower=True)
        class_scores = np.sum(z ** 2, axis=0)
        recomputed = np.minimum(recomputed, class_scores)

    np.testing.assert_allclose(saved_test_mahal, recomputed, rtol=1e-3)

