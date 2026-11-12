"""Supervised Clinical Tabular Modelling and Encoder Interface.

Implements baseline clinical classifiers and continuous tabular neural encoders:
- ClinicalLogisticRegressionModel: L2-regularized logistic regression with class-weight balancing.
- ClinicalRandomForestModel: Non-linear tabular ensemble baseline.
- ClinicalMLPEncoder: PyTorch MLP mapping discrete/continuous clinical features into continuous
  embeddings (d=128) designed for downstream multimodal fusion.
- Stratified 5-fold cross-validation with strict train-fold preprocessing fit (zero leakage).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn

from research.multimodal.clinical_preprocessing import ClinicalTabularPreprocessor

logger = logging.getLogger("multimodal.models")


class ClinicalMLPEncoder(nn.Module):
    """
    Continuous neural tabular encoder projecting heterogeneous clinical variables
    into a standardized dense representation space for multimodal fusion.
    """

    def __init__(
        self,
        in_dim: int,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.embedding_dim = embedding_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Preprocessed clinical feature tensor (B, in_dim).

        Returns:
            dense_embedding: Continuous representation (B, embedding_dim).
        """
        return self.encoder(x)


class ClinicalLogisticRegressionModel:
    """L2-penalized baseline logistic regression model for clinical tabular prediction."""

    def __init__(self, c: float = 1.0, random_state: int = 42) -> None:
        self.clf = LogisticRegression(
            C=c,
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ClinicalLogisticRegressionModel":
        self.clf.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.clf.predict(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(x)


def run_clinical_cross_validation(
    df: pd.DataFrame,
    target_name: str = "her2_overexpression",
    model_type: str = "logistic_regression",
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes Stratified 5-Fold Cross-Validation on the 300 clinical records.
    Crucial: Preprocessor is fitted strictly on the training fold in each split.
    """
    # Instantiate clean preprocessor to extract target labels
    probe_prep = ClinicalTabularPreprocessor(target_name=target_name)
    y_all, target_desc = probe_prep.extract_target_vector(df)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics: List[Dict[str, float]] = []

    all_y_true: List[int] = []
    all_y_pred: List[int] = []
    all_y_prob: List[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, y_all)):
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        # 1. Fit preprocessor strictly on training fold
        prep = ClinicalTabularPreprocessor(target_name=target_name)
        x_train = prep.fit_transform(train_df)
        x_val = prep.transform(val_df)

        y_train = y_all[train_idx]
        y_val = y_all[val_idx]

        # 2. Fit model
        if model_type == "logistic_regression":
            model = ClinicalLogisticRegressionModel(c=1.0, random_state=random_state + fold_idx)
            model.fit(x_train, y_train)
            val_preds = model.predict(x_val)
            val_probs = model.predict_proba(x_val)[:, 1]
        elif model_type == "random_forest":
            rf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight="balanced", random_state=random_state + fold_idx)
            rf.fit(x_train, y_train)
            val_preds = rf.predict(x_val)
            val_probs = rf.predict_proba(x_val)[:, 1]
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        all_y_true.extend(y_val.tolist())
        all_y_pred.extend(val_preds.tolist())
        all_y_prob.extend(val_probs.tolist())

        acc = float(accuracy_score(y_val, val_preds))
        bal_acc = float(balanced_accuracy_score(y_val, val_preds))
        f1 = float(f1_score(y_val, val_preds, zero_division=0))
        try:
            auc = float(roc_auc_score(y_val, val_probs))
        except ValueError:
            auc = 0.5
        brier = float(brier_score_loss(y_val, val_probs))

        fold_metrics.append({
            "fold": fold_idx,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "f1_score": f1,
            "auroc": auc,
            "brier_score": brier,
        })

    metrics_df = pd.DataFrame(fold_metrics)
    pooled_y_true = np.array(all_y_true)
    pooled_y_prob = np.array(all_y_prob)

    try:
        pooled_auc = float(roc_auc_score(pooled_y_true, pooled_y_prob))
    except ValueError:
        pooled_auc = 0.5

    return {
        "target_name": target_name,
        "target_description": target_desc,
        "model_type": model_type,
        "cohort_size": len(df),
        "positive_class_count": int(np.sum(y_all)),
        "prevalence": float(np.mean(y_all)),
        "n_splits": n_splits,
        "accuracy_mean": float(metrics_df["accuracy"].mean()),
        "accuracy_std": float(metrics_df["accuracy"].std()),
        "balanced_acc_mean": float(metrics_df["balanced_accuracy"].mean()),
        "balanced_acc_std": float(metrics_df["balanced_accuracy"].std()),
        "f1_mean": float(metrics_df["f1_score"].mean()),
        "f1_std": float(metrics_df["f1_score"].std()),
        "auroc_mean": float(metrics_df["auroc"].mean()),
        "auroc_std": float(metrics_df["auroc"].std()),
        "pooled_auroc": pooled_auc,
        "brier_score_mean": float(metrics_df["brier_score"].mean()),
        "brier_score_std": float(metrics_df["brier_score"].std()),
        "scientific_disclosure": (
            "Model evaluated as a unimodal clinical tabular baseline strictly within HMU-GC-Clinical. "
            "Because GCHTID image tiles lack patient identifiers, this model cannot be fused with "
            "histopathology image representations until verified crosswalk linkage is established."
        ),
    }

