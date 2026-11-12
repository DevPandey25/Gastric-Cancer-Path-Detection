"""Multimodal Clinical Research and Provenance Package for GastricPath AI.

Original Phase 10 research modules:
- Patient-level cross-dataset provenance and linkage auditing.
- Structured clinical cohort schema, missingness quantification, and data dictionary.
- Reproducible, leakage-free clinical tabular preprocessing and splitting.
- Exploratory clinical statistics, demographic figures, and bivariate associations.
- Supervised clinical tabular modeling (HER-2 status prediction) and continuous neural encoders.
- In-silico miRNA biomarker workbook audit (Gastric.xlsx) and consensus analysis.
- Regulatory miRNA-target network graph construction and query engine.
- Multimodal image-clinical fusion architectures (Early, Late, Cross-Attention) with gatekeeper.
- Conditional survival analysis eligibility auditing.
"""

from research.multimodal.biomarker_analysis import (
    ALGORITHM_COLUMNS,
    BiomarkerWorkbookAuditor,
)
from research.multimodal.clinical_analysis import (
    compute_cohort_descriptive_statistics,
    render_clinical_demographics_figure,
    render_clinical_missingness_heatmap,
    test_bivariate_clinical_associations,
)
from research.multimodal.clinical_models import (
    ClinicalLogisticRegressionModel,
    ClinicalMLPEncoder,
    run_clinical_cross_validation,
)
from research.multimodal.clinical_preprocessing import (
    ClinicalTabularPreprocessor,
    split_clinical_cohort,
)
from research.multimodal.clinical_schema import (
    STANDARDIZED_COLUMN_MAP,
    ClinicalFieldDefinition,
    ClinicalSchemaAuditor,
)
from research.multimodal.fusion import (
    EarlyFusionClassifier,
    LateFusionClassifier,
    MultimodalFusionGatekeeper,
)
from research.multimodal.linkage_audit import (
    LinkageRelationshipAudit,
    MultimodalLinkageBlockedError,
    PatientLinkageAuditor,
)
from research.multimodal.molecular_network import MolecularNetworkEngine
from research.multimodal.survival_analysis import (
    ConditionalSurvivalModelRunner,
    SurvivalAnalysisBlockedError,
    SurvivalEligibilityAuditor,
    SurvivalEligibilityResult,
)

__all__ = [
    "PatientLinkageAuditor",
    "LinkageRelationshipAudit",
    "MultimodalLinkageBlockedError",
    "ClinicalSchemaAuditor",
    "ClinicalFieldDefinition",
    "STANDARDIZED_COLUMN_MAP",
    "ClinicalTabularPreprocessor",
    "split_clinical_cohort",
    "compute_cohort_descriptive_statistics",
    "render_clinical_missingness_heatmap",
    "render_clinical_demographics_figure",
    "test_bivariate_clinical_associations",
    "ClinicalMLPEncoder",
    "ClinicalLogisticRegressionModel",
    "run_clinical_cross_validation",
    "BiomarkerWorkbookAuditor",
    "ALGORITHM_COLUMNS",
    "MolecularNetworkEngine",
    "EarlyFusionClassifier",
    "LateFusionClassifier",
    "MultimodalFusionGatekeeper",
    "SurvivalEligibilityAuditor",
    "SurvivalEligibilityResult",
    "SurvivalAnalysisBlockedError",
    "ConditionalSurvivalModelRunner",
]

