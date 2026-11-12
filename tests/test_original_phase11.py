"""
Automated Test Suite for Original Phase 11: Unified Research Workbench UI.

Tests:
1. Research API route availability and 200 OK responses.
2. Research overview metadata, sample counts, and phase completion matrix.
3. Dataset explorer GCHTID, Clinical aggregate, and in-silico biomarker summaries.
4. Patient privacy protection (zero admission IDs / individual patient rows exposed).
5. Phase 10 HER-2 terminology verification ("IHC 2+/3+ exploratory category — not confirmed HER2-positive status").
6. Biomarker in-silico target network disclosures.
7. Model lab inventory, checkpoint hashes, and evaluation protocol differentiation.
8. Foundation model comparative metrics (Phikon vs DINOv2 on 3,110 samples).
9. Explainability methods, faithfulness metrics, and biological causation caveats.
10. Segmentation 9-class ontology and data blockage guarantee (0 fake Dice/IoU).
11. Cellular morphology candidate detections and non-validation disclosures.
12. Spatial graphs CV performance (MLP baseline outperforming GNNs).
13. Whole-slide WSI zero-data audit status and synthetic fixture verification.
14. Similarity search fast cosine retrieval on cached Phikon embeddings.
15. Uncertainty ECE, Brier score, and synthetic perturbation decay curves.
16. Experiment tracker integration.
17. Validation consolidated matrix and clinical limitations.
18. Research reports catalog and markdown reader.
19. Baseline checkpoint SHA-256 invariant verification.
"""

from pathlib import Path
import pytest
from starlette.requests import Request

from app.core.config import settings
from app.main import app
from app.research_workbench import (
    get_research_overview,
    get_dataset_explorer_data,
    search_biomarker_targets,
    get_model_lab_data,
    get_foundation_models_data,
    get_explainability_data,
    get_segmentation_data,
    get_cellular_data,
    get_spatial_graphs_data,
    get_wsi_data,
    get_similarity_sample_queries,
    search_similar_patches,
    get_uncertainty_data,
    get_experiments_data,
    get_validation_data,
    get_reports_list,
    get_report_content,
)


def test_research_overview_endpoint():
    data = get_research_overview()
    assert "platform_title" in data
    assert "dataset_inventory" in data
    assert data["dataset_inventory"]["gchtid_tiles"]["count"] == 31096
    assert data["dataset_inventory"]["clinical_cohort"]["count"] == 300
    assert data["dataset_inventory"]["biomarker_workbook"]["interaction_records"] == 3711
    assert "phases_matrix" in data
    assert len(data["phases_matrix"]) >= 10
    assert "key_limitations" in data
    assert any("patch-level" in lim.lower() for lim in data["key_limitations"])


def test_dataset_explorer_gchtid():
    data = get_dataset_explorer_data()
    assert "gchtid" in data
    gchtid = data["gchtid"]
    assert gchtid["total_images"] == 31096
    assert len(gchtid["classes"]) == 8
    assert gchtid["splits"]["train"] == 21767
    assert gchtid["splits"]["val"] == 4664
    assert gchtid["splits"]["test"] == 4665
    assert gchtid["duplicate_audit"]["duplicate_groups"] == 932
    assert gchtid["duplicate_audit"]["cross_class_conflicts"] == 0


def test_dataset_explorer_clinical_privacy_and_her2():
    data = get_dataset_explorer_data()
    assert "clinical" in data
    clinical = data["clinical"]
    assert clinical["cohort_size"] == 300
    assert clinical["patient_ids_protected"] is True

    # Assert NO patient IDs in clinical summary
    keys = list(clinical.keys())
    assert "patient_id" not in keys
    assert "Patient" not in keys

    # Verify Phase 10 HER-2 label
    her2_info = clinical["her2_distribution"]
    assert "IHC 2+/3+ exploratory category — not confirmed HER2-positive status." in her2_info["scientific_label"]
    assert "ISH" in her2_info["scientific_caveat"]
    assert len(her2_info["categories"]) == 4

    # Verify missingness
    assert clinical["missingness"]["venous_invasion"] > 25.0
    assert clinical["linkage_status"] == "UNAVAILABLE"


def test_dataset_explorer_biomarker_and_search():
    data = get_dataset_explorer_data()
    assert "biomarker" in data
    biomarker = data["biomarker"]
    assert biomarker["mirna_id"] == "hsa-miR-497-5p"
    assert biomarker["total_records"] == 3711
    assert biomarker["unique_target_symbols"] == 3497
    assert "In-silico" in biomarker["scientific_disclosure"]

    # Test search query
    search_res = search_biomarker_targets(q="CCNE1", min_consensus=1)
    assert search_res["total_matches"] >= 1
    assert any(r["target_symbol"] == "CCNE1" for r in search_res["records"])


def test_model_lab_inventory_and_benchmarks():
    data = get_model_lab_data()
    assert "models_registered" in data
    assert len(data["models_registered"]) >= 1
    assert "benchmarks_by_protocol" in data
    assert len(data["benchmarks_by_protocol"]) >= 2

    # Assert distinct protocols are not conflated
    full_test = [b for b in data["benchmarks_by_protocol"] if b["sample_size"] == 4665]
    subset_test = [b for b in data["benchmarks_by_protocol"] if b["sample_size"] == 400]
    assert len(full_test) == 1
    assert len(subset_test) == 1
    assert full_test[0]["accuracy"] == 0.7104
    assert subset_test[0]["accuracy"] == 0.7150


def test_foundation_models_data():
    data = get_foundation_models_data()
    assert data["total_samples"] == 3110
    assert data["winning_model"] == "phikon_mean_patch"
    assert len(data["models"]) >= 4

    phikon_mp = [m for m in data["models"] if m["key"] == "phikon_mean_patch"][0]
    assert phikon_mp["accuracy"] > 0.64
    assert phikon_mp["auroc"] > 0.93
    assert "scientific_caveat" in data


def test_explainability_methods_and_caveat():
    data = get_explainability_data()
    assert len(data["methods_available"]) == 4
    method_names = [m["method"] for m in data["methods_available"]]
    assert "Grad-CAM" in method_names
    assert "Integrated Gradients" in method_names
    assert "Occlusion Sensitivity" in method_names
    assert "Phikon Token Attribution" in method_names
    assert "faithfulness" in data
    assert "biological causation" in data["scientific_caveat"].lower()


def test_segmentation_ontology_and_data_blocker():
    data = get_segmentation_data()
    assert data["num_classes"] == 9
    assert len(data["ontology"]) == 9
    codes = [c["code"] for c in data["ontology"]]
    assert "BG" in codes
    assert "TUM" in codes
    assert "ADI" in codes
    assert data["genuine_annotations_found"] == 0
    assert data["status_code"] == "blocked_by_data"
    assert "No fabricated Dice/IoU" in data["scientific_integrity_guarantee"]


def test_cellular_candidate_detection_and_caveat():
    data = get_cellular_data()
    assert data["pilot_cohort"]["processed_patches"] == 64
    assert data["pilot_cohort"]["total_candidates"] == 3895
    assert "candidate" in data["scientific_caveat"].lower()
    assert "not manually validated nuclei" in data["scientific_caveat"].lower()


def test_spatial_graphs_and_mlp_superiority():
    data = get_spatial_graphs_data()
    assert len(data["models"]) == 4
    mlp = [m for m in data["models"] if m["model_name"] == "Morphology MLP Baseline"][0]
    gcn = [m for m in data["models"] if "GCN" in m["model_name"]][0]

    # Morphology MLP baseline outperforming GNN
    assert mlp["balanced_accuracy"] > gcn["balanced_accuracy"]
    assert "outperformed all GNN" in data["key_finding"]


def test_whole_slide_analysis_zero_wsi_audit():
    data = get_wsi_data()
    assert data["repository_wsi_count"] == 0
    assert data["status_code"] == "blocked_by_data"
    assert len(data["modules_implemented"]) == 5
    assert "synthetic" in data["tested_fixture"].lower()


def test_similarity_search_retrieval():
    sample_queries = get_similarity_sample_queries()
    if sample_queries["available"]:
        assert len(sample_queries["samples"]) >= 1
        query_idx = sample_queries["samples"][0]["subset_index"]
        res = search_similar_patches(index=query_idx, top_k=3)
        assert res["top_k"] == 3
        assert len(res["neighbors"]) == 3
        assert res["neighbors"][0]["cosine_similarity"] <= 1.0
        assert "scientific_caveat" in res


def test_uncertainty_and_perturbation_decay():
    data = get_uncertainty_data()
    assert "in_distribution_metrics" in data
    assert data["in_distribution_metrics"]["expected_calibration_error"] < 0.05
    assert len(data["ood_detectors"]) == 2
    assert len(data["perturbation_robustness"]["tested_perturbations"]) == 4


def test_experiment_tracking():
    data = get_experiments_data()
    assert data["total_experiments"] >= 1
    assert "experiments" in data


def test_validation_consolidated_matrix():
    data = get_validation_data()
    assert "validation_matrix" in data
    assert len(data["validation_matrix"]) >= 10
    statuses = [v["status"] for v in data["validation_matrix"]]
    assert "Validated" in statuses
    assert "Blocked by Data" in statuses
    assert "Blocked by Linkage" in statuses
    assert "Fully Passing" in statuses


def test_reports_catalog_and_content_retrieval():
    catalog = get_reports_list()
    assert len(catalog["reports"]) >= 8
    filenames = [r["filename"] for r in catalog["reports"]]
    assert "PHASE10_MULTIMODAL_CLINICAL_RESEARCH_REPORT.md" in filenames

    # Test reading report content
    doc = get_report_content("PHASE10_MULTIMODAL_CLINICAL_RESEARCH_REPORT.md")
    assert doc["size_bytes"] > 1000
    assert "Multimodal Clinical Research" in doc["content"]


def test_baseline_checkpoint_sha256_unmodified():
    import hashlib
    checkpoint_path = settings.model_checkpoint
    assert checkpoint_path.exists(), f"Checkpoint {checkpoint_path} not found"

    hasher = hashlib.sha256()
    with open(checkpoint_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)

    expected_hash = "65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8"
    assert hasher.hexdigest().lower() == expected_hash.lower(), "Checkpoint SHA-256 hash mismatch!"


def test_dashboard_html_contains_workbench_nav():
    from app.main import dashboard
    req = Request({"type": "http"})
    res = dashboard(request=req)
    assert res.status_code == 200
    html = res.body.decode("utf-8")
    assert "Live Inference Dashboard" in html
    assert "Research Workbench (14 Modules)" in html
    assert 'data-module="overview"' in html
    assert 'data-module="dataset"' in html
    assert 'data-module="models"' in html
    assert 'data-module="foundation"' in html
    assert 'data-module="explainability"' in html
    assert 'data-module="segmentation"' in html
    assert 'data-module="cellular"' in html
    assert 'data-module="spatial-graphs"' in html
    assert 'data-module="wsi"' in html
    assert 'data-module="similarity"' in html
    assert 'data-module="uncertainty"' in html
    assert 'data-module="experiments"' in html
    assert 'data-module="validation"' in html
    assert 'data-module="reports"' in html


def test_dashboard_payload_her2_and_biomarker_corrections():
    from app.analytics import demo_summary
    data = demo_summary()
    assert "clinical" in data
    clinical = data["clinical"]
    assert "her2_exploratory_category_label" in clinical
    assert "IHC 2+/3+ exploratory category — not confirmed HER2-positive status." in clinical["her2_exploratory_category_label"]
    assert clinical["her2_unimodal_cv_benchmark"]["majority_class_baseline_accuracy"] > 0.75

    assert "biomarkers" in data
    biomarkers = data["biomarkers"]
    assert biomarkers["mirna_id"] == "hsa-miR-497-5p"
    assert "In-silico" in biomarkers["scientific_disclosure"]


def test_reports_catalog_has_all_phase_reports():
    catalog = get_reports_list()
    filenames = [r["filename"] for r in catalog["reports"]]
    assert "PHASE1_IMPLEMENTATION_REPORT.md" in filenames
    assert "PHASE2_IMPLEMENTATION_REPORT.md" in filenames
    assert "PHASE3_IMPLEMENTATION_REPORT.md" in filenames
    assert "PHASE4_4_REDUCED_COMPARISON_REPORT.md" in filenames
    assert "PHASE5_ORIGINAL_COMPLETION_REPORT.md" in filenames
    assert "PHASE6_TISSUE_SEGMENTATION_REPORT.md" in filenames
    assert "PHASE7_NUCLEI_CELLULAR_MORPHOLOGY_REPORT.md" in filenames
    assert "PHASE8_SPATIAL_GRAPH_GNN_REPORT.md" in filenames
    assert "PHASE9_WSI_ENGINE_REPORT.md" in filenames
    assert "PHASE10_MULTIMODAL_CLINICAL_RESEARCH_REPORT.md" in filenames
    assert "IMPLEMENTATION_STATUS.md" in filenames
