"""Comprehensive Verification Suite for Original Phase 10: Multimodal Clinical Research.

Tests:
1. Clinical schema validation, column standardization, and trailing empty column removal.
2. Missingness calculations across clinical fields.
3. Patient identifier uniqueness and duplicate detection.
4. Categorical and numerical preprocessing transformations.
5. Train-only preprocessing fit (preventing data leakage across partitions).
6. Patient-group aware splitting and leakage prevention.
7. Biomarker workbook validation (Table s2 schema, mature_mirna_id, target_symbol).
8. Molecular regulatory network construction and query engine.
9. Linkage audit validation and strict rejection of invalid / heuristic joins.
10. Multimodal fusion gatekeeper enforcement and architecture forward passes.
11. Survival analysis eligibility audit and strict blocker enforcement.
12. Graceful handling of unavailable fields and missing files.
13. Baseline checkpoint SHA-256 invariant preservation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from research.multimodal.biomarker_analysis import BiomarkerWorkbookAuditor
from research.multimodal.clinical_models import (
    ClinicalLogisticRegressionModel,
    ClinicalMLPEncoder,
)
from research.multimodal.clinical_preprocessing import (
    ClinicalTabularPreprocessor,
    split_clinical_cohort,
)
from research.multimodal.clinical_schema import (
    STANDARDIZED_COLUMN_MAP,
    ClinicalSchemaAuditor,
)
from research.multimodal.fusion import (
    EarlyFusionClassifier,
    LateFusionClassifier,
    MultimodalFusionGatekeeper,
    MultimodalLinkageBlockedError,
)
from research.multimodal.linkage_audit import PatientLinkageAuditor
from research.multimodal.molecular_network import MolecularNetworkEngine
from research.multimodal.survival_analysis import (
    ConditionalSurvivalModelRunner,
    SurvivalAnalysisBlockedError,
    SurvivalEligibilityAuditor,
)

EXPECTED_BASELINE_SHA256 = "65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8"


# ---------------------------------------------------------------------------
# 1. Clinical Schema and Data Quality Tests
# ---------------------------------------------------------------------------

def test_clinical_schema_loading_and_cleaning():
    auditor = ClinicalSchemaAuditor()
    df = auditor.load_and_clean_raw_dataframe()

    # Must contain 300 rows and exactly 15 standardized clinical columns
    assert len(df) == 300
    assert "Unnamed: 15" not in df.columns
    assert "Unnamed: 16" not in df.columns
    assert "patient_id" in df.columns
    assert "her2_status" in df.columns
    assert "comprehensive_stage" in df.columns


def test_clinical_patient_id_uniqueness():
    auditor = ClinicalSchemaAuditor()
    df = auditor.load_and_clean_raw_dataframe()
    assert df["patient_id"].nunique() == 300, "Duplicate patient IDs detected in clinical cohort!"


def test_clinical_missingness_quantification():
    auditor = ClinicalSchemaAuditor()
    df = auditor.load_and_clean_raw_dataframe()
    field_defs = auditor.generate_data_dictionary(df)

    def_map = {fd.standard_name: fd for fd in field_defs}
    assert def_map["patient_id"].missing_count == 0
    assert def_map["age"].missing_count == 0
    assert def_map["sex"].missing_count == 0
    assert def_map["her2_status"].missing_count == 0
    assert def_map["venous_invasion"].missing_pct > 25.0
    assert def_map["lauren_classification"].missing_pct > 15.0


def test_clinical_schema_summary_artifact():
    p = Path("research/multimodal/clinical_schema_summary.json")
    assert p.exists(), f"Missing schema artifact: {p}"
    with open(p) as f:
        data = json.load(f)
    assert data["total_patient_records"] == 300
    assert not data["survival_outcome_fields_present"]


# ---------------------------------------------------------------------------
# 2. Clinical Preprocessing and Leakage Prevention Tests
# ---------------------------------------------------------------------------

def test_clinical_preprocessing_train_only_fit():
    auditor = ClinicalSchemaAuditor()
    df = auditor.load_and_clean_raw_dataframe()

    train_df, test_df = split_clinical_cohort(df, test_size=0.2, random_state=42)

    preprocessor = ClinicalTabularPreprocessor(target_name="her2_overexpression")
    assert not preprocessor.is_fitted

    # Transform without fit must raise RuntimeError
    with pytest.raises(RuntimeError, match="must be fitted before calling transform"):
        _ = preprocessor.transform(test_df)

    x_train = preprocessor.fit_transform(train_df)
    assert preprocessor.is_fitted
    assert x_train.shape[0] == len(train_df)
    assert not np.any(np.isnan(x_train))

    # Transform test set using parameters learned strictly from train
    x_test = preprocessor.transform(test_df)
    assert x_test.shape[0] == len(test_df)
    assert x_test.shape[1] == x_train.shape[1]
    assert not np.any(np.isnan(x_test))


def test_split_clinical_cohort_patient_isolation():
    auditor = ClinicalSchemaAuditor()
    df = auditor.load_and_clean_raw_dataframe()

    train_df, test_df = split_clinical_cohort(df, test_size=0.25, random_state=42)
    train_pids = set(train_df["patient_id"])
    test_pids = set(test_df["patient_id"])

    # Zero overlap between patient IDs
    overlap = train_pids.intersection(test_pids)
    assert len(overlap) == 0, f"Patient leakage detected: {overlap}"


# ---------------------------------------------------------------------------
# 3. Clinical Models and Tabular Encoder Tests
# ---------------------------------------------------------------------------

def test_clinical_mlp_encoder_forward():
    in_dim = 25
    encoder = ClinicalMLPEncoder(in_dim=in_dim, embedding_dim=128, hidden_dim=64)
    encoder.eval()

    mock_x = torch.randn(8, in_dim)
    with torch.no_grad():
        emb = encoder(mock_x)

    assert emb.shape == (8, 128)
    assert not torch.isnan(emb).any()


def test_clinical_logistic_regression_fit():
    x_mock = np.random.randn(50, 10).astype(np.float32)
    y_mock = np.random.randint(0, 2, size=50)

    model = ClinicalLogisticRegressionModel(c=1.0)
    model.fit(x_mock, y_mock)

    preds = model.predict(x_mock[:5])
    probs = model.predict_proba(x_mock[:5])

    assert len(preds) == 5
    assert probs.shape == (5, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# 4. Biomarker Workbook and Regulatory Network Tests
# ---------------------------------------------------------------------------

def test_biomarker_workbook_audit():
    auditor = BiomarkerWorkbookAuditor()
    df = auditor.load_workbook_sheet()

    assert len(df) == 3711
    assert "mature_mirna_id" in df.columns
    assert "target_symbol" in df.columns
    assert "predicted.sum" in df.columns
    assert df["mature_mirna_id"].iloc[0] == "hsa-miR-497-5p"


def test_biomarker_audit_summary_artifact():
    p = Path("research/multimodal/biomarker_audit_summary.json")
    assert p.exists()
    with open(p) as f:
        data = json.load(f)
    assert data["total_interaction_rows"] == 3711
    assert data["mirna_profile"]["mature_mirna_id"] == "hsa-miR-497-5p"
    assert data["target_genes"]["unique_symbols"] > 3400


def test_molecular_network_construction_and_query():
    engine = MolecularNetworkEngine()
    g7 = engine.build_interaction_graph(min_consensus=7)

    assert "hsa-miR-497-5p" in g7
    # Exactly 11 targets have consensus >= 7 (1 with consensus=8 [SEMA6D], 10 with consensus=7) + 1 miRNA node = 12
    assert g7.number_of_nodes() == 12
    assert g7.number_of_edges() == 11

    # Unanimous 8/8 consensus yields exactly 1 gene: SEMA6D
    g8 = engine.build_interaction_graph(min_consensus=8)
    assert g8.number_of_nodes() == 2  # miRNA + SEMA6D
    assert g8.number_of_edges() == 1
    assert "SEMA6D" in g8

    # Consensus >= 5 yields > 200 nodes
    g5 = engine.build_interaction_graph(min_consensus=5)
    assert g5.number_of_nodes() > 200

    # Query interface
    results = engine.query_targets(min_consensus=7, top_k=10)
    assert len(results) == 10
    assert results[0]["consensus_databases_count"] >= 7


# ---------------------------------------------------------------------------
# 5. Linkage Audit and Rejection of Invalid Joins Tests
# ---------------------------------------------------------------------------

def test_linkage_audit_confirms_unavailability():
    auditor = PatientLinkageAuditor()
    img_clin = auditor.audit_gchtid_to_clinical()
    assert img_clin.linkage_status == "UNAVAILABLE"
    assert not img_clin.explicit_crosswalk_exists
    assert img_clin.linked_records_count == 0

    clin_bio = auditor.audit_clinical_to_biomarker()
    assert clin_bio.linkage_status == "UNAVAILABLE"


def test_linkage_gatekeeper_rejects_unlinked_merge():
    """
    CRITICAL PROVENANCE TEST: Merging datasets without overlapping keys
    must raise MultimodalLinkageBlockedError.
    """
    df_img = pd.DataFrame({"filename": ["ADI-001.png", "TUM-002.png"]})
    df_clin = pd.DataFrame({"Patient": ["1600097", "1606490"]})

    with pytest.raises(MultimodalLinkageBlockedError, match="Zero overlapping identifiers"):
        PatientLinkageAuditor.validate_linkage_attempt(
            source_df=df_img,
            target_df=df_clin,
            source_key="filename",
            target_key="Patient",
        )


def test_multimodal_fusion_gatekeeper_blocks_unlinked_fusion():
    """
    SAFETY INVARIANT: Image-clinical fusion must throw error when crosswalk is absent.
    """
    with pytest.raises(MultimodalLinkageBlockedError, match="BLOCKED: Image-clinical fusion cannot be performed"):
        MultimodalFusionGatekeeper.assert_fusion_prerequisites_met(
            has_verified_crosswalk=False,
            crosswalk_verified_count=0,
            cohort_sample_size=300,
            target_defined=True,
        )


def test_multimodal_fusion_forward_passes_synthetic_fixture():
    """
    Verifies fusion neural architectures (EarlyFusion, LateFusion) operate correctly
    when supplied with matched feature dimensions in software test mode.
    """
    b = 4
    img_emb = torch.randn(b, 768)
    clin_emb = torch.randn(b, 128)

    # Early Fusion
    early = EarlyFusionClassifier(image_dim=768, clinical_dim=128, num_classes=2)
    early.eval()
    with torch.no_grad():
        out_early = early(img_emb, clin_emb)
    assert out_early.shape == (b, 2)

    # Late Fusion
    late = LateFusionClassifier(image_dim=768, clinical_dim=128, num_classes=2)
    late.eval()
    with torch.no_grad():
        p_fused, p_img, p_clin = late(img_emb, clin_emb)
    assert p_fused.shape == (b, 2)
    assert torch.allclose(p_fused.sum(dim=1), torch.ones(b), atol=1e-5)


# ---------------------------------------------------------------------------
# 6. Survival Analysis Eligibility Tests
# ---------------------------------------------------------------------------

def test_survival_analysis_audit_confirms_blocker():
    auditor = ClinicalSchemaAuditor()
    df = auditor.load_and_clean_raw_dataframe()

    res = SurvivalEligibilityAuditor.audit_dataset_for_survival(df)
    assert not res.is_eligible
    assert len(res.missing_requirements) >= 2
    assert "follow-up duration" in res.blocker_description.lower()

    # Calling Kaplan-Meier on ineligible data must raise SurvivalAnalysisBlockedError
    with pytest.raises(SurvivalAnalysisBlockedError, match="strictly cross-sectional pathological variables"):
        ConditionalSurvivalModelRunner.run_kaplan_meier_analysis(df)


def test_survival_model_runs_on_synthetic_fixture():
    """Confirms Kaplan-Meier executes when supplied with synthetic test survival endpoints."""
    mock_survival_df = pd.DataFrame({
        "os_months": [12.0, 24.0, 6.0, 36.0, 18.0],
        "vital_status": [1, 0, 1, 0, 1],
        "surgery_date": ["2020-01-01"] * 5,
    })

    km_res = ConditionalSurvivalModelRunner.run_kaplan_meier_analysis(mock_survival_df)
    assert km_res["total_patients"] == 5
    assert km_res["total_events"] == 3
    assert len(km_res["curve"]) > 0


# ---------------------------------------------------------------------------
# 7. Baseline Invariant and Visualizations Tests
# ---------------------------------------------------------------------------

def test_phase10_visualizations_exist():
    vis_dir = Path("research/multimodal/visualizations")
    expected = [
        vis_dir / "clinical_missingness_heatmap.png",
        vis_dir / "clinical_cohort_demographics.png",
        vis_dir / "mirna_prediction_database_distribution.png",
        vis_dir / "mirna_target_network_graph.png",
    ]
    for p in expected:
        assert p.exists(), f"Missing visualization: {p}"
        assert p.stat().st_size > 10000, f"Visualization {p} appears empty or corrupted"


def test_baseline_checkpoint_sha256_unmodified():
    ckpt_path = Path("models/checkpoints/gastric_rnn_classifier.pt")
    assert ckpt_path.exists(), f"Baseline checkpoint not found: {ckpt_path}"
    sha256 = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    assert sha256 == EXPECTED_BASELINE_SHA256, (
        f"CRITICAL: Baseline checkpoint altered! Expected {EXPECTED_BASELINE_SHA256}, got {sha256}"
    )
