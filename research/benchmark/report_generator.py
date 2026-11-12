"""Report Generator for GastricPath AI Benchmarks.

Exports metrics.json, epoch_history.csv, predictions_test.csv, confusion_matrix.csv,
classification_report.csv, and experiment_summary.md with Wilson and bootstrap confidence intervals.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from research.benchmark.dataset import CANONICAL_CLASSES


def generate_experiment_reports(
    experiment_dir: Path | str,
    metrics: dict[str, Any],
    predictions: list[dict[str, Any]],
    history: list[dict[str, Any]] | None = None,
    experiment_metadata: dict[str, Any] | None = None,
    class_names: list[str] | None = None,
) -> dict[str, Path]:
    """Saves all standardized research artifacts into the experiment directory."""
    exp_path = Path(experiment_dir)
    exp_path.mkdir(parents=True, exist_ok=True)
    generated_files: dict[str, Path] = {}

    if class_names is None:
        class_names = CANONICAL_CLASSES

    # 1. metrics.json
    metrics_file = exp_path / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    generated_files["metrics_json"] = metrics_file

    # 2. epoch_history.csv
    if history:
        history_file = exp_path / "epoch_history.csv"
        fieldnames = list(history[0].keys())
        with open(history_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in history:
                writer.writerow(row)
        generated_files["epoch_history_csv"] = history_file

    # 3. predictions_test.csv
    if predictions:
        pred_file = exp_path / "predictions_test.csv"
        fieldnames = list(predictions[0].keys())
        with open(pred_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in predictions:
                writer.writerow(row)
        generated_files["predictions_test_csv"] = pred_file

    # 4. confusion_matrix.csv
    if "confusion_matrix" in metrics:
        cm_file = exp_path / "confusion_matrix.csv"
        cm_data = metrics["confusion_matrix"]
        with open(cm_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["actual_class"] + [f"pred_{c}" for c in class_names])
            for idx, row in enumerate(cm_data):
                writer.writerow([class_names[idx]] + row)
        generated_files["confusion_matrix_csv"] = cm_file

    # 5. classification_report.csv
    if "per_class" in metrics:
        report_file = exp_path / "classification_report.csv"
        with open(report_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "class_name",
                "class_index",
                "precision",
                "recall_sensitivity",
                "sensitivity_ci_95_lower",
                "sensitivity_ci_95_upper",
                "specificity",
                "f1_score",
                "support",
                "auroc_ovr",
            ])
            for c_name, c_data in metrics["per_class"].items():
                ci = c_data.get("sensitivity_ci_95", ["N/A", "N/A"])
                writer.writerow([
                    c_name,
                    c_data["class_index"],
                    c_data["precision"],
                    c_data["recall_sensitivity"],
                    ci[0],
                    ci[1],
                    c_data["specificity"],
                    c_data["f1_score"],
                    c_data["support"],
                    c_data.get("auroc_ovr", "N/A"),
                ])
        generated_files["classification_report_csv"] = report_file

    # 6. experiment_summary.md
    summary_file = exp_path / "experiment_summary.md"
    meta = experiment_metadata or {}
    overall = metrics.get("overall", {})

    acc_ci = overall.get("accuracy_ci_95", ["N/A", "N/A"])
    f1_ci = overall.get("macro_f1_ci_95", ["N/A", "N/A"])
    bal_ci = overall.get("balanced_accuracy_ci_95", ["N/A", "N/A"])

    md_lines = [
        f"# Experiment Summary: {meta.get('experiment_id', exp_path.name)}",
        "",
        f"- **Model Architecture:** {meta.get('architecture', 'Unknown')}",
        f"- **Parameter Count:** {meta.get('parameter_count', 'N/A')}",
        f"- **Input Representation:** {meta.get('input_mode', '2D')}",
        f"- **Dataset:** {meta.get('dataset_name', 'GCHTID')} ({meta.get('dataset_split', 'Image-Level Stratified')})",
        f"- **Timestamp (UTC):** {meta.get('timestamp', 'N/A')}",
        f"- **Execution Runtime:** {meta.get('runtime_seconds', 'N/A')} seconds",
        f"- **Git Commit Hash:** `{meta.get('git_commit', 'N/A')}`",
        "",
        "## Overall Test Evaluation Metrics",
        "",
        f"| Metric | Value | 95% Confidence Interval |",
        f"|:---|:---:|:---:|",
        f"| **Test Accuracy** | **{overall.get('accuracy', 'N/A')}** | [{acc_ci[0]}, {acc_ci[1]}] (Wilson) |",
        f"| **Balanced Accuracy** | {overall.get('balanced_accuracy', 'N/A')} | [{bal_ci[0]}, {bal_ci[1]}] (Bootstrap) |",
        f"| **Macro F1-Score** | **{overall.get('macro_f1', 'N/A')}** | [{f1_ci[0]}, {f1_ci[1]}] (Bootstrap) |",
        f"| **Weighted F1-Score** | {overall.get('weighted_f1', 'N/A')} | — |",
        f"| **Macro Precision** | {overall.get('macro_precision', 'N/A')} | — |",
        f"| **Macro Recall** | {overall.get('macro_recall', 'N/A')} | — |",
        f"| **Cohen's Kappa** | {overall.get('cohen_kappa', 'N/A')} | — |",
        f"| **Macro AUROC (OvR)** | {overall.get('macro_auroc', 'N/A')} | — |",
        f"| **Expected Calibration Error (ECE)** | {overall.get('expected_calibration_error', 'N/A')} | — |",
        f"| **Brier Score** | {overall.get('brier_score', 'N/A')} | — |",
        "",
        "## Scientific Integrity Limitation",
        "",
        "> [!IMPORTANT]",
        "> Evaluated on image-level group-stratified splits of the GCHTID dataset. True patient or slide identifiers",
        "> are not encoded in image filenames. Results represent patch classification performance and must not be",
        "> interpreted as patient-level clinical diagnostic accuracy or anatomical tumor area burden.",
        "",
    ]

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    generated_files["experiment_summary_md"] = summary_file

    return generated_files
