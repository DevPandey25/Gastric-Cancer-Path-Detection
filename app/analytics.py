import json
from functools import lru_cache

import pandas as pd

from app.core.config import settings
from app.ml.constants import CLASS_COLORS, CLASS_LABELS, TISSUE_CLASSES
from app.ml.inference import inference_service


def _as_chart_items(mapping: dict[str, int | float], use_labels: bool = False) -> list[dict]:
    items = []
    for key, value in mapping.items():
        items.append(
            {
                "key": key,
                "label": CLASS_LABELS.get(key, key) if use_labels else key,
                "value": value,
                "color": CLASS_COLORS.get(key, "#6b7280"),
            }
        )
    return items


@lru_cache(maxsize=1)
def dataset_summary() -> dict:
    counts = {}
    dataset_available = settings.histology_dataset_dir.exists()
    total_images = 0
    snapshot = {}

    if not dataset_available and settings.dataset_snapshot_json.exists():
        snapshot = json.loads(settings.dataset_snapshot_json.read_text(encoding="utf-8"))

    snapshot_counts = snapshot.get("class_distribution", {})
    snapshot_total = int(snapshot.get("total_images", 0))

    for class_name in TISSUE_CLASSES:
        class_dir = settings.histology_dataset_dir / class_name
        if dataset_available and class_dir.exists():
            count = len(list(class_dir.glob("*.png")))
        else:
            count = int(snapshot_counts.get(class_name, 0))
        counts[class_name] = count
        total_images += count

    if not total_images:
        total_images = snapshot_total

    return {
        "dataset_path": str(settings.histology_dataset_dir),
        "dataset_available": dataset_available,
        "using_snapshot": (not dataset_available) and bool(snapshot_counts),
        "total_images": total_images,
        "image_size": settings.image_size,
        "sequence_length": settings.sequence_length,
        "patch_size": settings.patch_size,
        "class_distribution": _as_chart_items(counts, use_labels=True),
    }


@lru_cache(maxsize=1)
def clinical_summary() -> dict:
    df = pd.read_csv(settings.clinical_csv)
    her2 = pd.to_numeric(
        df["HER-2(Negative=0，1=1+，2=2+，3=3+)"],
        errors="coerce",
    )
    histology_counts = df["Histological Type"].fillna("Unknown").value_counts().head(6).to_dict()

    return {
        "total_patients": int(len(df)),
        "mean_age": round(float(pd.to_numeric(df["Age"], errors="coerce").mean()), 1),
        "sex_distribution": [{"label": key, "value": int(value)} for key, value in df["Sex"].value_counts().to_dict().items()],
        "stage_distribution": [
            {"label": key, "value": int(value)}
            for key, value in df["Comprehensive Staging"].astype(str).value_counts().to_dict().items()
        ],
        "top_histologies": [{"label": key, "value": int(value)} for key, value in histology_counts.items()],
        "her2_distribution": [
            {"label": str(int(key)) if pd.notna(key) else "Unknown", "value": int(value)}
            for key, value in her2.value_counts(dropna=True).to_dict().items()
        ],
        "her2_four_tier_counts": {
            "0 (Negative)": int((her2 == 0).sum()),
            "1+ (Negative/Equivocal)": int((her2 == 1).sum()),
            "2+ (Equivocal / Requires ISH)": int((her2 == 2).sum()),
            "3+ (Positive Overexpression)": int((her2 == 3).sum()),
        },
        "her2_exploratory_category_label": "IHC 2+/3+ exploratory category — not confirmed HER2-positive status.",
        "her2_scientific_caveat": "IHC 2+ requires In Situ Hybridization (ISH) confirmation for clinical HER2 positivity. No patient-level ISH assays exist in this cohort.",
        "her2_unimodal_cv_benchmark": {
            "task": "HER-2 Exploratory Category (2+/3+ vs 0/1+)",
            "majority_class_baseline_accuracy": 0.7533,
            "logistic_regression": {
                "balanced_accuracy_mean": 0.4274,
                "balanced_accuracy_std": 0.0666,
                "macro_f1_mean": 0.2576,
                "macro_f1_std": 0.0806,
                "auroc_mean": 0.4212,
                "auroc_std": 0.0710,
            },
            "random_forest": {
                "balanced_accuracy_mean": 0.4930,
                "balanced_accuracy_std": 0.0704,
                "macro_f1_mean": 0.2366,
                "macro_f1_std": 0.1125,
                "auroc_mean": 0.4895,
                "auroc_std": 0.1058,
            },
            "clinical_utility_disclosure": "Model results do not demonstrate clinically useful discrimination.",
        },
        "patient_privacy_enforced": True,
        "privacy_notice": "Individual patient IDs and raw records excluded. Aggregate cohort statistics only.",
    }


@lru_cache(maxsize=1)
def biomarker_summary() -> dict:
    df = pd.read_excel(settings.biomarker_xlsx)
    grouped = (
        df.groupby("target_symbol", as_index=False)[["predicted.sum", "all.sum"]]
        .max()
        .sort_values(by=["predicted.sum", "all.sum"], ascending=False)
        .head(10)
    )
    return {
        "sheet_name": "Table s2",
        "rows": int(len(df)),
        "mirna_id": "hsa-miR-497-5p",
        "network_nature": "In-silico predicted target network",
        "scientific_disclosure": "In-silico predictions across 8 computational tools. Not experimentally validated patient biomarkers.",
        "top_targets": [
            {
                "target_symbol": row["target_symbol"],
                "predicted_sum": int(row["predicted.sum"]),
                "all_sum": int(row["all.sum"]),
            }
            for _, row in grouped.iterrows()
        ],
    }


@lru_cache(maxsize=1)
def benchmark_summary() -> dict:
    payload = json.loads(settings.notebook_metrics_json.read_text(encoding="utf-8"))
    payload["confusion_matrix_image"] = "/static/assets/confusion_matrix.png"
    payload["source_notebook"] = str(settings.notebook_path)
    return payload


def demo_summary() -> dict:
    return {
        "model_status": inference_service.status(),
        "dataset": dataset_summary(),
        "clinical": clinical_summary(),
        "biomarkers": biomarker_summary(),
        "benchmark": benchmark_summary(),
    }
