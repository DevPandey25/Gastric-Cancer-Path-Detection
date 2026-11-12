"""Master Execution and Evaluation Pipeline for Original Phase 10: Multimodal Clinical Research.

Orchestrates:
1. Patient-level cross-dataset provenance and linkage audit.
2. Clinical cohort schema standardization and missingness analysis.
3. Clinical demographic and missingness visualization generation.
4. Stratified 5-fold cross-validation of unimodal clinical tabular models.
5. In-silico biomarker workbook audit and consensus distribution analysis.
6. Molecular regulatory network construction and visualization.
7. Survival analysis eligibility audit.
8. Multimodal fusion gatekeeper validation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from research.multimodal.biomarker_analysis import BiomarkerWorkbookAuditor
from research.multimodal.clinical_analysis import (
    compute_cohort_descriptive_statistics,
    render_clinical_demographics_figure,
    render_clinical_missingness_heatmap,
    test_bivariate_clinical_associations,
)
from research.multimodal.clinical_models import run_clinical_cross_validation
from research.multimodal.clinical_schema import ClinicalSchemaAuditor
from research.multimodal.linkage_audit import PatientLinkageAuditor
from research.multimodal.molecular_network import MolecularNetworkEngine
from research.multimodal.survival_analysis import SurvivalEligibilityAuditor

logger = logging.getLogger("multimodal.evaluation")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = Path("research/multimodal")
VIS_DIR = BASE_DIR / "visualizations"
LINKAGE_JSON = BASE_DIR / "linkage_audit_summary.json"
SCHEMA_JSON = BASE_DIR / "clinical_schema_summary.json"
BIOMARKER_JSON = BASE_DIR / "biomarker_audit_summary.json"
MODELING_JSON = BASE_DIR / "clinical_modeling_summary.json"


def run_phase10_pipeline() -> Dict[str, Any]:
    """
    Executes end-to-end Phase 10 clinical and multimodal research pipeline.
    """
    logger.info("Executing Original Phase 10: Multimodal Clinical Research Pipeline...")
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Patient-Level Linkage Audit
    logger.info("1/7 Auditing cross-dataset patient-level provenance and linkage...")
    linkage_auditor = PatientLinkageAuditor()
    linkage_summary = linkage_auditor.execute_complete_audit(output_json_path=LINKAGE_JSON)

    # 2. Clinical Schema & Missingness Audit
    logger.info("2/7 Auditing HMU-GC-Clinical schema and field missingness...")
    schema_auditor = ClinicalSchemaAuditor()
    schema_summary = schema_auditor.execute_schema_audit(output_json_path=SCHEMA_JSON)
    clean_clinical_df = schema_auditor.load_and_clean_raw_dataframe()

    # 3. Clinical Demographics and Missingness Visualizations
    logger.info("3/7 Generating publication clinical visualizations...")
    missingness_path = VIS_DIR / "clinical_missingness_heatmap.png"
    render_clinical_missingness_heatmap(clean_clinical_df, missingness_path)

    demographics_path = VIS_DIR / "clinical_cohort_demographics.png"
    render_clinical_demographics_figure(clean_clinical_df, demographics_path)

    descriptive_stats = compute_cohort_descriptive_statistics(clean_clinical_df)
    assoc_results = test_bivariate_clinical_associations(clean_clinical_df)

    # 4. Supervised Unimodal Clinical Modeling (HER-2 Status Prediction)
    logger.info("4/7 Evaluating unimodal clinical models (HER-2 status prediction via 5-fold CV)...")
    cv_lr = run_clinical_cross_validation(
        clean_clinical_df,
        target_name="her2_overexpression",
        model_type="logistic_regression",
        n_splits=5,
    )
    cv_rf = run_clinical_cross_validation(
        clean_clinical_df,
        target_name="her2_overexpression",
        model_type="random_forest",
        n_splits=5,
    )
    modeling_summary = {
        "task": "Unimodal Clinical HER-2 Overexpression Prediction (2+/3+ vs 0/1+)",
        "cohort_size": len(clean_clinical_df),
        "prevalence": cv_lr["prevalence"],
        "logistic_regression": cv_lr,
        "random_forest": cv_rf,
    }
    with open(MODELING_JSON, "w") as f:
        json.dump(modeling_summary, f, indent=2)

    # 5. Biomarker Workbook Audit
    logger.info("5/7 Auditing Gastric.xlsx in-silico miRNA target predictions...")
    biomarker_auditor = BiomarkerWorkbookAuditor()
    biomarker_summary = biomarker_auditor.execute_biomarker_audit(output_json_path=BIOMARKER_JSON)

    # 6. Molecular Regulatory Network
    logger.info("6/7 Generating molecular interaction network and consensus distribution...")
    network_engine = MolecularNetworkEngine()
    consensus_dist_path = VIS_DIR / "mirna_prediction_database_distribution.png"
    network_engine.render_consensus_distribution_figure(consensus_dist_path)

    network_graph_path = VIS_DIR / "mirna_target_network_graph.png"
    network_engine.render_molecular_network_figure(network_graph_path, top_n_genes=30)

    # 7. Survival Eligibility Audit
    logger.info("7/7 Auditing survival analysis eligibility...")
    survival_audit = SurvivalEligibilityAuditor.audit_dataset_for_survival(clean_clinical_df)

    pipeline_summary = {
        "phase": "Original Phase 10: Multimodal Clinical Research",
        "linkage_status": linkage_summary["overall_linkage_status"],
        "clinical_cohort_size": len(clean_clinical_df),
        "clinical_modeling": {
            "target": cv_lr["target_description"],
            "lr_accuracy": f"{cv_lr['accuracy_mean']:.4f} +/- {cv_lr['accuracy_std']:.4f}",
            "lr_balanced_acc": f"{cv_lr['balanced_acc_mean']:.4f} +/- {cv_lr['balanced_acc_std']:.4f}",
            "lr_auroc": f"{cv_lr['auroc_mean']:.4f} +/- {cv_lr['auroc_std']:.4f}",
            "rf_balanced_acc": f"{cv_rf['balanced_acc_mean']:.4f} +/- {cv_rf['balanced_acc_std']:.4f}",
        },
        "biomarker_summary": {
            "mirna": biomarker_summary["mirna_profile"]["mature_mirna_id"],
            "unique_targets": biomarker_summary["target_genes"]["unique_symbols"],
            "high_consensus_targets_ge_7": biomarker_summary["top_consensus_targets_count_ge_7"],
        },
        "survival_eligibility": {
            "is_eligible": survival_audit.is_eligible,
            "blocker": survival_audit.blocker_description,
        },
        "visualizations": {
            "missingness_heatmap": str(missingness_path.resolve()).replace("\\", "/"),
            "cohort_demographics": str(demographics_path.resolve()).replace("\\", "/"),
            "consensus_distribution": str(consensus_dist_path.resolve()).replace("\\", "/"),
            "network_graph": str(network_graph_path.resolve()).replace("\\", "/"),
        },
    }

    logger.info("Original Phase 10 Pipeline completed successfully.")
    return pipeline_summary


if __name__ == "__main__":
    res = run_phase10_pipeline()
    print("Execution complete.")
    print("Linkage Status:", res["linkage_status"])
    print("Clinical Modeling:", res["clinical_modeling"])
    print("Survival Eligible:", res["survival_eligibility"]["is_eligible"])

