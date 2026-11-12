"""Reproducible Preprocessing Engine for Clinical Tabular Data.

Implements strict train-only preprocessing transformations:
- Numerical median imputation and standard scaling (preventing distribution shift leakage).
- Missingness indicators and categorical encoding for pathological staging variables.
- Harmonized handling of missing invasion metrics and Lauren classifications.
- Patient-group aware stratification (guaranteeing repeat observations stay isolated).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from research.multimodal.clinical_schema import ClinicalSchemaAuditor

logger = logging.getLogger("multimodal.preprocessing")

T_STAGE_ORDINAL_MAP = {"T1a": 1, "T1b": 2, "T2": 3, "T3": 4, "T4a": 5, "T4b": 6}
N_STAGE_ORDINAL_MAP = {"N0": 0, "N1": 1, "N2": 2, "N3a": 3, "N3b": 4}
M_STAGE_MAP = {"M0": 0, "M1": 1}
COMPREHENSIVE_STAGE_ORDINAL_MAP = {
    "ⅠA": 1, "ⅠB": 2, "ⅡA": 3, "ⅡB": 4, "ⅢA": 5, "ⅢB": 6, "ⅢC": 7, "Ⅳ": 8
}


class ClinicalTabularPreprocessor:
    """
    Stateful preprocessor for HMU-GC-Clinical records.
    Strictly follows scikit-learn fit/transform pattern:
    fitted ONLY on training data to prevent data leakage.
    """

    def __init__(
        self,
        target_name: str = "her2_overexpression",
        include_comprehensive_stage: bool = False,
    ) -> None:
        """
        Args:
            target_name: One of 'her2_overexpression' (2+/3+ vs 0/1+),
                         'her2_positive' (1+/2+/3+ vs 0), or 'advanced_stage' (Stage III/IV vs I/II).
            include_comprehensive_stage: Whether to include overall stage as a predictor
                                        (set to False if predicting staging itself).
        """
        self.target_name = target_name
        self.include_comprehensive_stage = include_comprehensive_stage
        self.is_fitted = False

        # Learned state from training data
        self.age_mean: float = 0.0
        self.age_std: float = 1.0
        self.nominal_encoder: Optional[OneHotEncoder] = None
        self.feature_names_: List[str] = []

        # Imputation defaults learned from train fold
        self.median_imputations_: Dict[str, float] = {}

    def extract_target_vector(self, df: pd.DataFrame) -> Tuple[np.ndarray, str]:
        """
        Extracts clinically meaningful binary target vector from standardized DataFrame.
        """
        if self.target_name == "her2_overexpression":
            # Clinical consensus (ToGA criteria): HER2 2+ and 3+ indicate potential overexpression
            y = (df["her2_status"] >= 2).astype(int).to_numpy()
            target_desc = "HER2 Overexpression (2+/3+ vs 0/1+)"
        elif self.target_name == "her2_positive":
            # Any detected HER2 expression (1+, 2+, 3+) vs complete negative (0)
            y = (df["her2_status"] >= 1).astype(int).to_numpy()
            target_desc = "HER2 Expression Present (1+/2+/3+ vs 0)"
        elif self.target_name == "advanced_stage":
            # Advanced Gastric Cancer: Stage III or IV vs Stage I or II
            adv_stages = {"ⅢA", "ⅢB", "ⅢC", "Ⅳ"}
            y = df["comprehensive_stage"].isin(adv_stages).astype(int).to_numpy()
            target_desc = "Advanced Stage (III/IV vs I/II)"
        else:
            raise ValueError(f"Unknown clinical target name: {self.target_name}")

        return y, target_desc

    def fit(self, df: pd.DataFrame) -> "ClinicalTabularPreprocessor":
        """
        Learns transformation parameters strictly from the training dataframe.
        """
        # 1. Numerical scaling parameter (Age)
        self.age_mean = float(df["age"].mean())
        self.age_std = float(df["age"].std()) if df["age"].std() > 0 else 1.0

        # 2. Imputation parameters for missing pathological indicators
        for col in ["lauren_classification", "lymphovascular_invasion", "venous_invasion", "perineural_invasion", "stroma_quantity"]:
            if col in df.columns:
                self.median_imputations_[col] = float(df[col].median(skipna=True))

        # 3. Categorical Nominal OneHotEncoder
        nominal_cols = ["sex", "histological_type", "tumor_infiltration_pattern"]
        sub_nom = df[nominal_cols].copy()
        # Fill missing with 'Missing' category
        for c in nominal_cols:
            sub_nom[c] = sub_nom[c].fillna("Missing").astype(str)

        self.nominal_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.nominal_encoder.fit(sub_nom)

        # 4. Construct complete feature name list
        feat_names = ["age_scaled"]
        feat_names.extend(["t_stage_ord", "n_stage_ord", "m_stage_bin"])
        if self.include_comprehensive_stage:
            feat_names.append("comprehensive_stage_ord")

        feat_names.extend([
            "lauren_classification",
            "lauren_missing_flag",
            "lymphovascular_invasion",
            "lymphovascular_missing_flag",
            "venous_invasion",
            "venous_missing_flag",
            "perineural_invasion",
            "perineural_missing_flag",
            "stroma_quantity",
            "stroma_missing_flag",
        ])

        # Nominal one-hot feature names
        enc_names = self.nominal_encoder.get_feature_names_out(nominal_cols).tolist()
        feat_names.extend(enc_names)

        self.feature_names_ = feat_names
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Applies learned transformations to a dataset (training, validation, or test).
        """
        if not self.is_fitted:
            raise RuntimeError("ClinicalTabularPreprocessor must be fitted before calling transform().")

        features_list: List[np.ndarray] = []

        # 1. Age scaled
        age_arr = ((df["age"].to_numpy(dtype=np.float32) - self.age_mean) / self.age_std).reshape(-1, 1)
        features_list.append(age_arr)

        # 2. Ordinal TNM staging
        t_ord = df["t_stage"].map(T_STAGE_ORDINAL_MAP).fillna(3).to_numpy(dtype=np.float32).reshape(-1, 1)
        n_ord = df["n_stage"].map(N_STAGE_ORDINAL_MAP).fillna(0).to_numpy(dtype=np.float32).reshape(-1, 1)
        m_bin = df["m_stage"].map(M_STAGE_MAP).fillna(0).to_numpy(dtype=np.float32).reshape(-1, 1)
        features_list.extend([t_ord, n_ord, m_bin])

        if self.include_comprehensive_stage:
            stage_ord = df["comprehensive_stage"].map(COMPREHENSIVE_STAGE_ORDINAL_MAP).fillna(4).to_numpy(dtype=np.float32).reshape(-1, 1)
            features_list.append(stage_ord)

        # 3. Pathological invasion indicators with explicit missingness flags
        for col in ["lauren_classification", "lymphovascular_invasion", "venous_invasion", "perineural_invasion", "stroma_quantity"]:
            s = df[col]
            missing_flag = s.isna().astype(np.float32).to_numpy().reshape(-1, 1)
            val = s.fillna(self.median_imputations_.get(col, 0.0)).to_numpy(dtype=np.float32).reshape(-1, 1)
            features_list.extend([val, missing_flag])

        # 4. Nominal One-Hot Encoded features
        nominal_cols = ["sex", "histological_type", "tumor_infiltration_pattern"]
        sub_nom = df[nominal_cols].copy()
        for c in nominal_cols:
            sub_nom[c] = sub_nom[c].fillna("Missing").astype(str)

        nom_encoded = self.nominal_encoder.transform(sub_nom)
        features_list.append(nom_encoded)

        feature_matrix = np.hstack(features_list).astype(np.float32)
        return feature_matrix

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)


def split_clinical_cohort(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    stratify_col: Optional[str] = "her2_status",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits clinical cohort into train and test partitions.
    Guarantees patient isolation (grouping by patient_id).
    """
    # Check patient uniqueness
    unique_patients = df["patient_id"].nunique()
    total_records = len(df)

    if unique_patients == total_records:
        # 1:1 patient to record mapping
        strat = df[stratify_col] if stratify_col in df.columns else None
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=strat,
        )
    else:
        # Grouped split to prevent multi-record patient leakage
        patient_list = df["patient_id"].unique()
        train_pids, test_pids = train_test_split(patient_list, test_size=test_size, random_state=random_state)
        train_df = df[df["patient_id"].isin(train_pids)].copy()
        test_df = df[df["patient_id"].isin(test_pids)].copy()

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

