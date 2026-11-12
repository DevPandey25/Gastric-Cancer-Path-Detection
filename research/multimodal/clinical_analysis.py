"""Exploratory Clinical Data Analysis and Demographic Visualization Engine.

Generates descriptive statistics, demographic breakdowns, missingness heatmaps,
and exploratory bivariate association tests for the 300 HMU gastric cancer cohort:
- Age, sex, AJCC pathological stage, and WHO histological subtype distributions.
- Missingness pattern visualization across pathological invasion fields.
- Publication-quality visualization rendering (headless Matplotlib).
- Rigorous statistical disclosures (sample sizes, test statistics, and limitations).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from research.multimodal.clinical_schema import ClinicalSchemaAuditor

logger = logging.getLogger("multimodal.analysis")


def compute_cohort_descriptive_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes comprehensive descriptive clinical statistics for the 300 patients.
    """
    n_patients = len(df)

    # 1. Age
    age_mean = float(df["age"].mean())
    age_std = float(df["age"].std())
    age_median = float(df["age"].median())
    age_q25 = float(df["age"].quantile(0.25))
    age_q75 = float(df["age"].quantile(0.75))

    # 2. Sex
    sex_counts = df["sex"].value_counts().to_dict()
    sex_pcts = (df["sex"].value_counts(normalize=True) * 100.0).round(1).to_dict()

    # 3. Comprehensive Stage
    stage_counts = df["comprehensive_stage"].value_counts().to_dict()
    adv_count = int(df["comprehensive_stage"].isin(["ⅢA", "ⅢB", "ⅢC", "Ⅳ"]).sum())

    # 4. Histological Types
    histo_counts = df["histological_type"].value_counts().to_dict()

    # 5. HER-2 Expression
    her2_counts = df["her2_status"].value_counts().to_dict()
    her2_overexp = int((df["her2_status"] >= 2).sum())  # 2+ or 3+

    # 6. Lauren Classification
    lauren_counts = df["lauren_classification"].value_counts(dropna=False).to_dict()

    return {
        "cohort_size": n_patients,
        "age": {
            "mean": age_mean,
            "std": age_std,
            "median": age_median,
            "iqr": [age_q25, age_q75],
            "min": int(df["age"].min()),
            "max": int(df["age"].max()),
        },
        "sex": {
            "counts": sex_counts,
            "percentages": sex_pcts,
        },
        "comprehensive_stage": {
            "counts": stage_counts,
            "advanced_stage_iii_iv_count": adv_count,
            "advanced_stage_pct": float(round((adv_count / n_patients) * 100.0, 1)),
        },
        "histological_subtypes": histo_counts,
        "her2_expression": {
            "score_counts": {str(k): int(v) for k, v in her2_counts.items()},
            "overexpression_2_or_3_count": her2_overexp,
            "overexpression_pct": float(round((her2_overexp / n_patients) * 100.0, 1)),
        },
        "lauren_classification": {str(k): int(v) for k, v in lauren_counts.items()},
    }


def render_clinical_missingness_heatmap(df: pd.DataFrame, output_path: Union[str, Path]) -> str:
    """
    Renders missingness matrix visualization across all 15 clinical columns.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Boolean matrix (True = missing, False = present)
    missing_mat = df.isna().to_numpy()
    cols = df.columns.tolist()

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(missing_mat.T, aspect="auto", cmap="Blues", interpolation="nearest")

    ax.set_yticks(np.arange(len(cols)))
    ax.set_yticklabels(cols, fontsize=9)
    ax.set_xlabel("Patient Index (N=300)")
    ax.set_title("HMU Gastric Cancer Clinical Cohort Missingness Pattern (Dark = Missing)", fontsize=11, fontweight="bold")

    # Add missing percentage text next to labels
    for idx, col in enumerate(cols):
        pct = (df[col].isna().sum() / len(df)) * 100.0
        if pct > 0:
            ax.text(len(df) + 3, idx, f"{pct:.1f}%", va="center", fontsize=8, color="darkred", fontweight="bold")

    plt.tight_layout()
    fig.savefig(str(out_p), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out_p.resolve()).replace("\\", "/")


def render_clinical_demographics_figure(df: pd.DataFrame, output_path: Union[str, Path]) -> str:
    """
    Renders 4-panel publication-grade demographic summary:
    - Panel A: Age Distribution (Histogram + KDE)
    - Panel B: Sex Distribution (Pie / Bar)
    - Panel C: AJCC Comprehensive Staging (Bar)
    - Panel D: HER-2 Status Distribution (Bar)
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Panel A: Age Distribution
    ax_a = axes[0, 0]
    ax_a.hist(df["age"], bins=15, color="#1f77b4", edgecolor="black", alpha=0.75)
    mean_age = df["age"].mean()
    median_age = df["age"].median()
    ax_a.axvline(mean_age, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_age:.1f}")
    ax_a.axvline(median_age, color="orange", linestyle=":", linewidth=1.5, label=f"Median: {median_age:.1f}")
    ax_a.set_xlabel("Age at Diagnosis (Years)")
    ax_a.set_ylabel("Patient Count (N=300)")
    ax_a.set_title("A: Patient Age Distribution", fontweight="bold")
    ax_a.legend(loc="upper left")
    ax_a.grid(True, linestyle="--", alpha=0.4)

    # Panel B: Sex Breakdown
    ax_b = axes[0, 1]
    sex_c = df["sex"].value_counts()
    colors = ["#4A90E2", "#E24A90"]
    ax_b.bar(sex_c.index, sex_c.values, color=colors, edgecolor="black", alpha=0.85, width=0.5)
    for i, v in enumerate(sex_c.values):
        ax_b.text(i, v + 4, f"{v} ({v/len(df)*100:.1f}%)", ha="center", fontsize=9, fontweight="bold")
    ax_b.set_ylabel("Patient Count")
    ax_b.set_title("B: Biological Sex Distribution", fontweight="bold")
    ax_b.set_ylim(0, max(sex_c.values) + 30)
    ax_b.grid(True, axis="y", linestyle="--", alpha=0.4)

    # Panel C: Comprehensive AJCC Staging
    ax_c = axes[1, 0]
    stage_order = ["ⅠA", "ⅠB", "ⅡA", "ⅡB", "ⅢA", "ⅢB", "ⅢC", "Ⅳ"]
    stage_counts = [df["comprehensive_stage"].value_counts().get(s, 0) for s in stage_order]
    bar_colors = ["#2ca02c" if "Ⅰ" in s or "Ⅱ" in s else "#d62728" for s in stage_order]
    bars = ax_c.bar(stage_order, stage_counts, color=bar_colors, edgecolor="black", alpha=0.8)
    for b in bars:
        h = b.get_height()
        ax_c.text(b.get_x() + b.get_width() / 2, h + 1, str(h), ha="center", fontsize=8)
    ax_c.set_xlabel("AJCC/UICC Comprehensive Tumor Stage")
    ax_c.set_ylabel("Patient Count")
    ax_c.set_title("C: Stage Distribution (Green: Early I/II, Red: Advanced III/IV)", fontweight="bold")
    ax_c.grid(True, axis="y", linestyle="--", alpha=0.4)

    # Panel D: HER-2 Immunohistochemistry Status
    ax_d = axes[1, 1]
    her2_order = [0, 1, 2, 3]
    her2_labels = ["0 (Negative)", "1+ (Negative)", "2+ (Equivocal)", "3+ (Positive)"]
    her2_counts = [int((df["her2_status"] == score).sum()) for score in her2_order]
    her2_colors = ["#7f7f7f", "#aec7e8", "#ffbb78", "#d62728"]
    bars_d = ax_d.bar(her2_labels, her2_counts, color=her2_colors, edgecolor="black", alpha=0.85)
    for b in bars_d:
        h = b.get_height()
        ax_d.text(b.get_x() + b.get_width() / 2, h + 2, f"{h} ({h/len(df)*100:.1f}%)", ha="center", fontsize=8)
    ax_d.set_ylabel("Patient Count")
    ax_d.set_title("D: HER2 Immunohistochemical Status (ToGA Criteria)", fontweight="bold")
    ax_d.set_ylim(0, max(her2_counts) + 20)
    ax_d.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.suptitle("HMU Gastric Cancer Clinical Cohort Demographics (N=300 Patients)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(str(out_p), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out_p.resolve()).replace("\\", "/")


def test_bivariate_clinical_associations(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Conducts exploratory statistical association tests on available clinical variables.
    Reports chi-square statistic, p-value, sample size, and unverified hypothesis disclosure.
    """
    results: Dict[str, Any] = {}

    # Test 1: HER2 Overexpression (2+/3+ vs 0/1+) across Lauren Classification (Intestinal vs Diffuse/Mixed)
    sub_df = df.dropna(subset=["lauren_classification"]).copy()
    sub_df["her2_overexp"] = (sub_df["her2_status"] >= 2).astype(int)
    # Lauren: 0=Intestinal, 1=Mixed, 2=Diffuse
    sub_df["lauren_binary"] = sub_df["lauren_classification"].map({0.0: "Intestinal", 1.0: "Non-Intestinal", 2.0: "Non-Intestinal"})

    contingency = pd.crosstab(sub_df["lauren_binary"], sub_df["her2_overexp"])
    chi2, p_val, dof, _ = stats.chi2_contingency(contingency)

    results["her2_vs_lauren"] = {
        "description": "Association between HER2 Overexpression (2+/3+) and Lauren Histological Type",
        "sample_size": len(sub_df),
        "excluded_missing": len(df) - len(sub_df),
        "chi2_statistic": float(round(chi2, 3)),
        "degrees_of_freedom": int(dof),
        "p_value": float(round(p_val, 4)),
        "statistically_significant": bool(p_val < 0.05),
        "contingency_table": contingency.to_dict(),
        "clinical_disclosure": (
            "Observed association is exploratory and hypothesis-generating. "
            "Higher HER2 amplification is classically reported in intestinal-type gastric cancer."
        ),
    }

    return results

