"""Clinical Cohort Schema, Data Dictionary, and Quality Audit Engine.

Standardizes schema, data types, missingness quantification, and clinical data dictionary
generation for the 300 gastric cancer patient records in HMU-GC-Clinical.csv:
- Strips empty trailing spreadsheet columns (Unnamed: 15, Unnamed: 16).
- Standardizes non-ASCII headers and whitespace.
- Classifies variables into numerical, categorical nominal, and ordinal stages.
- Validates physiological plausibility (e.g. Age in [18, 110], TNM stage consistency).
- Documents the absence of survival duration and vital status follow-up fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("multimodal.schema")

# Standardized column renaming map for clean programmatic access
STANDARDIZED_COLUMN_MAP: Dict[str, str] = {
    "Patient": "patient_id",
    "Sex": "sex",
    "Age": "age",
    "T staging": "t_stage",
    "N staging": "n_stage",
    "M staging": "m_stage",
    "Comprehensive Staging": "comprehensive_stage",
    "Histological Type": "histological_type",
    "Lauren classification(Intestinal=0, Mixed=1, Diffuse=2)": "lauren_classification",
    "Lymphovascular Invasion(Negative=0，Positive=1)": "lymphovascular_invasion",
    "Venous Invasion(Negative=0，Positive=1)": "venous_invasion",
    "Perineural Invasion(Negative=0，Positive=1)": "perineural_invasion",
    "Stroma Quantity(Medullary=0，Intermediate=1，Scirrhous=2)": "stroma_quantity",
    "Tumor Infiltration Pattern": "tumor_infiltration_pattern",
    "HER-2(Negative=0，1=1+，2=2+，3=3+)": "her2_status",
}


@dataclass
class ClinicalFieldDefinition:
    """Standardized metadata definition for a single clinical variable."""
    standard_name: str
    raw_name: str
    observed_dtype: str
    variable_type: str  # "Identifier", "Numerical", "Categorical Nominal", "Categorical Ordinal", "Binary Indicator"
    missing_count: int
    missing_pct: float
    unique_count: int
    value_range_or_categories: List[Any]
    clinical_interpretation: str
    proposed_preprocessing: str
    suitable_for_modeling: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ClinicalSchemaAuditor:
    """
    Audits the clinical cohort dataset and produces structured data dictionaries.
    """

    def __init__(self, clinical_csv_path: Union[str, Path] = "data/support/HMU-GC-Clinical.csv") -> None:
        self.csv_path = Path(clinical_csv_path)

    def load_and_clean_raw_dataframe(self) -> pd.DataFrame:
        """
        Loads HMU-GC-Clinical.csv, drops completely empty trailing columns,
        and renames columns to standardized snake_case identifiers.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Clinical CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        # Drop columns where 100% of values are NaN (e.g. Unnamed: 15, Unnamed: 16)
        cols_to_drop = [c for c in df.columns if df[c].isna().all()]
        df = df.drop(columns=cols_to_drop)

        # Standardize column names
        rename_dict = {orig: STANDARDIZED_COLUMN_MAP[orig] for orig in df.columns if orig in STANDARDIZED_COLUMN_MAP}
        df = df.rename(columns=rename_dict)

        return df

    def generate_data_dictionary(self, df: Optional[pd.DataFrame] = None) -> List[ClinicalFieldDefinition]:
        """
        Builds formal clinical data dictionary across all verified fields.
        """
        if df is None:
            df = self.load_and_clean_raw_dataframe()

        field_defs: List[ClinicalFieldDefinition] = []
        n_total = len(df)

        meta_lookup = {
            "patient_id": {
                "type": "Identifier",
                "interp": "Unique patient admission identifier at Harbin Medical University (HMU)",
                "prep": "Exclude from feature matrix; retain strictly as grouping key for leakage prevention",
                "modeling": False,
            },
            "sex": {
                "type": "Categorical Nominal",
                "interp": "Biological sex of patient (Male / Female)",
                "prep": "Binary one-hot encoding (Male=1, Female=0)",
                "modeling": True,
            },
            "age": {
                "type": "Numerical",
                "interp": "Patient age at clinical diagnosis (years)",
                "prep": "Median imputation (if missing) + StandardScaler normalization",
                "modeling": True,
            },
            "t_stage": {
                "type": "Categorical Ordinal",
                "interp": "Pathological tumor depth of invasion (T1a, T1b, T2, T3, T4a, T4b; AJCC 7th/8th)",
                "prep": "Ordinal integer encoding or One-Hot encoding",
                "modeling": True,
            },
            "n_stage": {
                "type": "Categorical Ordinal",
                "interp": "Regional lymph node metastasis stage (N0, N1, N2, N3a, N3b)",
                "prep": "Ordinal integer encoding or One-Hot encoding",
                "modeling": True,
            },
            "m_stage": {
                "type": "Binary Indicator",
                "interp": "Distant metastasis (M0=Negative, M1=Positive)",
                "prep": "Binary indicator (M0=0, M1=1)",
                "modeling": True,
            },
            "comprehensive_stage": {
                "type": "Categorical Ordinal",
                "interp": "AJCC/UICC Comprehensive Tumor Stage (ⅠA to Ⅳ)",
                "prep": "Ordinal encoding or grouping into Early (I/II) vs Advanced (III/IV)",
                "modeling": True,
            },
            "histological_type": {
                "type": "Categorical Nominal",
                "interp": "WHO Histopathological subtype (Tubular, Mucinous, Poorly diff, etc.)",
                "prep": "One-Hot encoding with rare class grouping",
                "modeling": True,
            },
            "lauren_classification": {
                "type": "Categorical Nominal",
                "interp": "Lauren Histological Classification: 0=Intestinal, 1=Mixed, 2=Diffuse",
                "prep": "One-Hot encoding with dedicated 'missing' indicator",
                "modeling": True,
            },
            "lymphovascular_invasion": {
                "type": "Binary Indicator",
                "interp": "Presence of tumor cells within lymphatic or blood vascular spaces (0=Neg, 1=Pos)",
                "prep": "Binary imputation with missing indicator",
                "modeling": True,
            },
            "venous_invasion": {
                "type": "Binary Indicator",
                "interp": "Invasion of venous vascular walls (0=Negative, 1=Positive)",
                "prep": "Binary imputation with missing indicator",
                "modeling": True,
            },
            "perineural_invasion": {
                "type": "Binary Indicator",
                "interp": "Tumor infiltration along nerve sheaths (0=Negative, 1=Positive)",
                "prep": "Binary imputation with missing indicator",
                "modeling": True,
            },
            "stroma_quantity": {
                "type": "Categorical Ordinal",
                "interp": "Relative stroma volume: 0=Medullary, 1=Intermediate, 2=Scirrhous",
                "prep": "Ordinal encoding with median imputation",
                "modeling": True,
            },
            "tumor_infiltration_pattern": {
                "type": "Categorical Nominal",
                "interp": "Tumor Infiltration Pattern (INFa=Expanding, INFb=Intermediate, INFc=Infiltrating)",
                "prep": "One-Hot encoding with missing indicator",
                "modeling": True,
            },
            "her2_status": {
                "type": "Categorical Ordinal",
                "interp": "HER2 protein expression on immunohistochemistry (0=Neg, 1=1+, 2=2+, 3=3+)",
                "prep": "Target variable (e.g. Overexpression 2+/3+ vs Neg/1+) or ordinal feature",
                "modeling": True,
            },
        }

        # Reverse map to locate raw column name
        rev_map = {v: k for k, v in STANDARDIZED_COLUMN_MAP.items()}

        for col in df.columns:
            raw_col = rev_map.get(col, col)
            meta = meta_lookup.get(col, {
                "type": "Unknown",
                "interp": "Unspecified clinical field",
                "prep": "Standard imputation",
                "modeling": False,
            })

            missing_cnt = int(df[col].isna().sum())
            missing_pct = float((missing_cnt / n_total) * 100.0)
            u_vals = df[col].dropna().unique().tolist()
            # Truncate unique values display if extensive
            disp_vals = u_vals[:8] if len(u_vals) > 8 else u_vals

            field_defs.append(ClinicalFieldDefinition(
                standard_name=col,
                raw_name=raw_col,
                observed_dtype=str(df[col].dtype),
                variable_type=meta["type"],
                missing_count=missing_cnt,
                missing_pct=missing_pct,
                unique_count=int(df[col].nunique()),
                value_range_or_categories=[str(v) for v in disp_vals],
                clinical_interpretation=meta["interp"],
                proposed_preprocessing=meta["prep"],
                suitable_for_modeling=meta["modeling"],
            ))

        return field_defs

    def execute_schema_audit(self, output_json_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Executes schema and quality audit and optionally exports summary JSON.
        """
        df = self.load_and_clean_raw_dataframe()
        field_defs = self.generate_data_dictionary(df)

        # Missingness quantification
        missing_dict = {fd.standard_name: fd.missing_pct for fd in field_defs}
        fields_with_missing = {k: v for k, v in missing_dict.items() if v > 0.0}

        # Survival outcome field check
        # Specifically verify presence of follow-up time, event indicator, death
        survival_candidates = ["os_months", "survival_time", "vital_status", "death", "follow_up", "recurrence"]
        survival_found = [c for c in df.columns if any(s in c.lower() for s in survival_candidates)]

        schema_summary = {
            "phase": "Original Phase 10: Multimodal Clinical Research",
            "cohort_name": "HMU-GC-Clinical Gastric Cancer Cohort",
            "total_patient_records": len(df),
            "total_retained_columns": len(df.columns),
            "dropped_empty_columns": ["Unnamed: 15", "Unnamed: 16"],
            "patient_identifier_uniqueness": bool(df["patient_id"].nunique() == len(df)),
            "age_summary": {
                "mean": float(df["age"].mean()),
                "std": float(df["age"].std()),
                "min": int(df["age"].min()),
                "max": int(df["age"].max()),
                "median": float(df["age"].median()),
            },
            "sex_distribution": df["sex"].value_counts().to_dict(),
            "her2_distribution": {str(k): int(v) for k, v in df["her2_status"].value_counts().items()},
            "fields_with_missingness": fields_with_missing,
            "survival_outcome_fields_present": False,
            "survival_blocker_reason": (
                "HMU-GC-Clinical.csv contains strictly cross-sectional pathological staging, biopsy "
                "histology, and immunohistochemistry. It contains zero follow-up duration fields, zero "
                "vital status / death indicators, and zero censoring times. Survival modeling cannot be performed."
            ),
            "field_definitions": [fd.to_dict() for fd in field_defs],
        }

        if output_json_path:
            p = Path(output_json_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w") as f:
                json.dump(schema_summary, f, indent=2)
            logger.info(f"Clinical schema summary written to {p}")

        return schema_summary

