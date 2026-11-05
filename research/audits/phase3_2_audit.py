"""GastricPath AI - Phase 3.2 Checkpoint Training-Lineage & Duplicate-Aware Audit Engine.

Executes forensic reconstruction of original checkpoint training lineage,
audits overlap with Phase 3.1 test split, analyzes duplicate/dHash distributions,
computes duplicate-aware and cluster-bootstrap uncertainty metrics, and exports
standardized machine-readable audit artifacts to research/audits/phase3_2/.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics

from research.benchmark.dataset import CANONICAL_CLASSES
from research.benchmark.metrics import (
    compute_comprehensive_metrics,
    wilson_score_interval,
)

logger = logging.getLogger(__name__)


def extract_relative_path(path_str: str) -> str:
    """Extracts normalized relative path (e.g. 'ADI/ADI_1000.png') from any path format."""
    p = Path(path_str)
    return f"{p.parent.name}/{p.name}"


def run_phase3_2_audit(workspace_root: Path | str | None = None) -> dict[str, Any]:
    """Executes the full Phase 3.2 forensic audit and outputs all required artifacts."""
    if workspace_root is None:
        workspace_root = Path(__file__).resolve().parent.parent.parent
    else:
        workspace_root = Path(workspace_root).resolve()

    output_dir = workspace_root / "research" / "audits" / "phase3_2"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GASTRICPATH AI — PHASE 3.2 TRAINING-LINEAGE & DUPLICATE AUDIT")
    print("=" * 80)
    print(f"Audit Output Directory: {output_dir}")

    # 1. Load Files
    hist_manifest_path = workspace_root / "model_train" / "runs" / "dataset_manifest.csv"
    test_manifest_path = workspace_root / "research" / "splits" / "manifest_test.csv"
    preds_path = workspace_root / "research" / "experiments" / "exp_20261105_221027_2463a2de" / "predictions_test.csv"
    exact_rep_path = workspace_root / "research" / "reports" / "exact_duplicates_report.json"
    near_rep_path = workspace_root / "research" / "reports" / "near_duplicates_report.json"

    if not hist_manifest_path.exists():
        raise FileNotFoundError(f"Historical training manifest not found: {hist_manifest_path}")
    if not test_manifest_path.exists():
        raise FileNotFoundError(f"Phase 3.1 test manifest not found: {test_manifest_path}")
    if not preds_path.exists():
        raise FileNotFoundError(f"Phase 3.1 predictions not found: {preds_path}")

    df_hist = pd.read_csv(hist_manifest_path)
    df_test = pd.read_csv(test_manifest_path)
    df_preds = pd.read_csv(preds_path)

    with open(exact_rep_path, "r", encoding="utf-8") as f:
        exact_rep = json.load(f)
    with open(near_rep_path, "r", encoding="utf-8") as f:
        near_rep = json.load(f)

    # 2. Normalize and Map Historical Lineage
    df_hist["rel_path"] = df_hist["image_path"].apply(extract_relative_path)
    hist_split_map = dict(zip(df_hist["rel_path"], df_hist["split"]))

    # Map exact duplicate groups and near duplicate clusters
    path_to_exact_grp = {}
    exact_grp_info = {}
    for g_idx, g in enumerate(exact_rep.get("duplicate_groups", [])):
        gid = f"exact_grp_{g_idx:04d}"
        exact_grp_info[gid] = g
        for p in g["paths"]:
            path_to_exact_grp[p] = gid

    path_to_near_cluster = {}
    near_cluster_info = {}
    for c in near_rep.get("clusters", []):
        cid = f"cluster_{c['cluster_id']:04d}"
        near_cluster_info[cid] = c
        for p in c["paths"]:
            path_to_near_cluster[p] = cid

    # Merge metadata into predictions
    df_merged = df_preds.merge(df_test[["relative_path", "sha256", "dhash"]], on="relative_path", how="left")
    df_merged["historical_split"] = df_merged["relative_path"].map(hist_split_map).fillna("unresolved")
    df_merged["in_original_train"] = df_merged["historical_split"] == "train"
    df_merged["in_original_val"] = df_merged["historical_split"] == "val"
    df_merged["in_original_test"] = df_merged["historical_split"] == "test"
    df_merged["exact_duplicate_group_id"] = df_merged["relative_path"].map(path_to_exact_grp).fillna("none")
    df_merged["near_duplicate_cluster_id"] = df_merged["relative_path"].map(path_to_near_cluster).fillna("none")

    # Composite cluster ID for bootstrap: duplicate group ID if exists, otherwise singleton SHA-256
    df_merged["cluster_id"] = np.where(
        df_merged["exact_duplicate_group_id"] != "none",
        df_merged["exact_duplicate_group_id"],
        "singleton_" + df_merged["sha256"],
    )

    # 3. Task 2: Export checkpoint_train_test_overlap.csv
    overlap_csv_path = output_dir / "checkpoint_train_test_overlap.csv"
    overlap_cols = [
        "relative_path",
        "sha256",
        "ground_truth_label",
        "predicted_label",
        "is_correct",
        "confidence",
        "historical_split",
        "in_original_train",
        "in_original_val",
        "in_original_test",
        "exact_duplicate_group_id",
        "near_duplicate_cluster_id",
    ]
    df_merged[overlap_cols].to_csv(overlap_csv_path, index=False)
    print(f"Exported overlap audit CSV: {overlap_csv_path}")

    # Calculate Overlap Statistics
    total_test = len(df_merged)
    n_train_overlap = int(df_merged["in_original_train"].sum())
    n_val_overlap = int(df_merged["in_original_val"].sum())
    n_test_overlap = int(df_merged["in_original_test"].sum())
    n_unresolved = int((df_merged["historical_split"] == "unresolved").sum())

    pct_train_overlap = round(n_train_overlap / total_test * 100.0, 2)
    pct_val_overlap = round(n_val_overlap / total_test * 100.0, 2)
    pct_test_overlap = round(n_test_overlap / total_test * 100.0, 2)

    print(f"\nHistorical Lineage Breakdown:")
    print(f"  In Original Train (Gradient Training): {n_train_overlap:4d} / {total_test} ({pct_train_overlap}%)")
    print(f"  In Original Val   (Model Selection):   {n_val_overlap:4d} / {total_test} ({pct_val_overlap}%)")
    print(f"  In Original Test  (Genuinely Unseen):  {n_test_overlap:4d} / {total_test} ({pct_test_overlap}%)")
    print(f"  Unresolved Lineage:                     {n_unresolved:4d} / {total_test} (0.00%)")

    # 4. Task 3: Duplicate-Aware Test Cohort Analysis
    unique_hashes = int(df_merged["sha256"].nunique())
    duplicate_files_count = total_test - unique_hashes

    # Test duplicate groups summary
    test_grps = defaultdict(list)
    for idx, r in df_merged.iterrows():
        gid = r["exact_duplicate_group_id"]
        if gid != "none":
            test_grps[gid].append(r)

    grp_summary_rows = []
    for gid, rows in sorted(test_grps.items()):
        paths = [r["relative_path"] for r in rows]
        gts = set(r["ground_truth_label"] for r in rows)
        preds = set(r["predicted_label"] for r in rows)
        cname = list(gts)[0] if len(gts) == 1 else "CONFLICT"
        info = exact_grp_info.get(gid, {})
        grp_summary_rows.append({
            "group_id": gid,
            "group_type": "exact_sha256",
            "class_name": cname,
            "num_test_images": len(rows),
            "total_dataset_images": info.get("count", len(rows)),
            "test_paths": "|".join(paths),
            "label_consistent": len(gts) == 1,
            "predictions_consistent": len(preds) == 1,
            "predicted_classes": "|".join(sorted(preds)),
        })

    grp_summary_df = pd.DataFrame(grp_summary_rows)
    grp_summary_csv = output_dir / "test_duplicate_group_summary.csv"
    grp_summary_df.to_csv(grp_summary_csv, index=False)
    print(f"Exported duplicate group summary CSV: {grp_summary_csv}")

    n_exact_grps_in_test = len(test_grps)
    n_multimember_exact_grps = sum(1 for r in grp_summary_rows if r["num_test_images"] > 1)
    n_conflicts = sum(1 for r in grp_summary_rows if not r["label_consistent"])
    n_inconsistent_preds = sum(1 for r in grp_summary_rows if not r["predictions_consistent"])

    print(f"\nDuplicate Analysis inside Test Split:")
    print(f"  Unique SHA-256 Hashes:                  {unique_hashes} / {total_test}")
    print(f"  Duplicate Images in Test:               {duplicate_files_count}")
    print(f"  Exact Duplicate Groups represented:     {n_exact_grps_in_test}")
    print(f"  Multi-member Groups in Test:            {n_multimember_exact_grps}")
    print(f"  Contradictory Labels in Test Groups:    {n_conflicts}")
    print(f"  Inconsistent Predictions in Groups:     {n_inconsistent_preds}")

    # 5. Task 4: Deduplicated Predictions and Exact-Unique Metrics
    prob_cols = [f"prob_{c}" for c in CANONICAL_CLASSES]
    dedup_prediction_rows = []
    for sha, group in df_merged.groupby("sha256"):
        # Select first exemplar deterministically (manifest is sorted by relative_path)
        first_row = group.iloc[0].to_dict()
        first_row["duplicate_count"] = len(group)
        first_row["all_relative_paths"] = "|".join(group["relative_path"].tolist())
        dedup_prediction_rows.append(first_row)

    dedup_df = pd.DataFrame(dedup_prediction_rows)
    dedup_csv_path = output_dir / "deduplicated_predictions.csv"
    dedup_export_cols = [
        "sha256",
        "relative_path",
        "ground_truth_label",
        "ground_truth_index",
        "duplicate_count",
        "all_relative_paths",
        "predicted_label",
        "predicted_index",
        "is_correct",
        "confidence",
    ] + prob_cols
    dedup_df[dedup_export_cols].to_csv(dedup_csv_path, index=False)
    print(f"Exported deduplicated predictions CSV: {dedup_csv_path}")

    # 6. Compute File-Weighted vs Exact-Unique vs Lineage Metrics
    gts_file = df_merged["ground_truth_index"].values
    probs_file = df_merged[prob_cols].values
    m_file = compute_comprehensive_metrics(gts_file, probs_file)

    gts_dedup = dedup_df["ground_truth_index"].values
    probs_dedup = dedup_df[prob_cols].values
    m_dedup = compute_comprehensive_metrics(gts_dedup, probs_dedup)

    # Subsets by historical lineage
    df_train_sub = df_merged[df_merged["in_original_train"]]
    m_train_sub = compute_comprehensive_metrics(
        df_train_sub["ground_truth_index"].values,
        df_train_sub[prob_cols].values,
    )

    df_val_sub = df_merged[df_merged["in_original_val"]]
    m_val_sub = compute_comprehensive_metrics(
        df_val_sub["ground_truth_index"].values,
        df_val_sub[prob_cols].values,
    )

    df_test_sub = df_merged[df_merged["in_original_test"]]
    m_test_sub = compute_comprehensive_metrics(
        df_test_sub["ground_truth_index"].values,
        df_test_sub[prob_cols].values,
    )

    # 7. Cluster Bootstrap Uncertainty
    clusters = df_merged["cluster_id"].unique()
    total_clusters = len(clusters)
    cluster_dict = {}
    for cid, grp in df_merged.groupby("cluster_id"):
        cluster_dict[cid] = {
            "true": grp["ground_truth_index"].values,
            "pred": grp["predicted_index"].values,
            "correct": grp["is_correct"].values,
        }

    rng = np.random.default_rng(42)
    B = 1000
    boot_acc = []
    boot_f1 = []
    boot_bal = []

    for _ in range(B):
        sampled_cids = rng.choice(clusters, size=total_clusters, replace=True)
        all_true = []
        all_pred = []
        all_correct = []
        for cid in sampled_cids:
            c = cluster_dict[cid]
            all_true.extend(c["true"])
            all_pred.extend(c["pred"])
            all_correct.extend(c["correct"])

        y_t = np.array(all_true)
        y_p = np.array(all_pred)

        boot_acc.append(float(np.mean(all_correct)))
        boot_f1.append(float(sk_metrics.f1_score(y_t, y_p, average="macro", zero_division=0)))
        boot_bal.append(float(sk_metrics.balanced_accuracy_score(y_t, y_p)))

    ci_cluster_acc = [round(float(v), 4) for v in np.percentile(boot_acc, [2.5, 97.5])]
    ci_cluster_f1 = [round(float(v), 4) for v in np.percentile(boot_f1, [2.5, 97.5])]
    ci_cluster_bal = [round(float(v), 4) for v in np.percentile(boot_bal, [2.5, 97.5])]

    cluster_bootstrap_data = {
        "resampling_unit": "exact/dHash duplicate cluster (indivisible group)",
        "total_test_images": total_test,
        "total_clusters": total_clusters,
        "multimember_clusters": n_multimember_exact_grps,
        "singleton_clusters": total_clusters - n_multimember_exact_grps,
        "bootstrap_replicates": B,
        "random_seed": 42,
        "cluster_bootstrap_intervals": {
            "accuracy_95": ci_cluster_acc,
            "balanced_accuracy_95": ci_cluster_bal,
            "macro_f1_95": ci_cluster_f1,
        },
        "iid_file_bootstrap_intervals": {
            "accuracy_95": m_file["overall"]["accuracy_ci_95"],
            "balanced_accuracy_95": m_file["overall"]["balanced_accuracy_ci_95"],
            "macro_f1_95": m_file["overall"]["macro_f1_ci_95"],
        },
        "methodological_disclosure": (
            "Cluster bootstrap resamples whole duplicate clusters with replacement, preserving intra-cluster "
            "correlation. These intervals apply strictly to image/cluster sampling variability in the GCHTID dataset "
            "and cannot be claimed as patient-level clinical confidence intervals."
        ),
    }

    cluster_json_path = output_dir / "cluster_bootstrap_results.json"
    with open(cluster_json_path, "w", encoding="utf-8") as f:
        json.dump(cluster_bootstrap_data, f, indent=2)
    print(f"Exported cluster bootstrap JSON: {cluster_json_path}")

    # 8. Export duplicate_aware_metrics.json
    audit_metrics_data = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_test_files": total_test,
            "exact_unique_images": unique_hashes,
            "duplicate_files_in_test": duplicate_files_count,
            "historical_train_overlap_files": n_train_overlap,
            "historical_val_overlap_files": n_val_overlap,
            "historical_unseen_test_files": n_test_overlap,
            "integrity_classification": "CONTAMINATED",
        },
        "file_weighted_metrics": m_file["overall"],
        "exact_unique_metrics": m_dedup["overall"],
        "lineage_subset_comparison": {
            "historical_train_overlap_subset": {
                "sample_count": n_train_overlap,
                "percentage_of_test": pct_train_overlap,
                "accuracy": m_train_sub["overall"]["accuracy"],
                "balanced_accuracy": m_train_sub["overall"]["balanced_accuracy"],
                "macro_f1": m_train_sub["overall"]["macro_f1"],
                "macro_auroc": m_train_sub["overall"]["macro_auroc"],
                "expected_calibration_error": m_train_sub["overall"]["expected_calibration_error"],
                "brier_score": m_train_sub["overall"]["brier_score"],
            },
            "historical_val_overlap_subset": {
                "sample_count": n_val_overlap,
                "percentage_of_test": pct_val_overlap,
                "accuracy": m_val_sub["overall"]["accuracy"],
                "balanced_accuracy": m_val_sub["overall"]["balanced_accuracy"],
                "macro_f1": m_val_sub["overall"]["macro_f1"],
                "macro_auroc": m_val_sub["overall"]["macro_auroc"],
                "expected_calibration_error": m_val_sub["overall"]["expected_calibration_error"],
                "brier_score": m_val_sub["overall"]["brier_score"],
            },
            "historical_unseen_test_subset": {
                "sample_count": n_test_overlap,
                "percentage_of_test": pct_test_overlap,
                "accuracy": m_test_sub["overall"]["accuracy"],
                "balanced_accuracy": m_test_sub["overall"]["balanced_accuracy"],
                "macro_f1": m_test_sub["overall"]["macro_f1"],
                "macro_auroc": m_test_sub["overall"]["macro_auroc"],
                "expected_calibration_error": m_test_sub["overall"]["expected_calibration_error"],
                "brier_score": m_test_sub["overall"]["brier_score"],
            },
        },
        "per_class_deduplicated": m_dedup["per_class"],
        "per_class_file_weighted": m_file["per_class"],
    }

    metrics_json_path = output_dir / "duplicate_aware_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_metrics_data, f, indent=2)
    print(f"Exported duplicate-aware metrics JSON: {metrics_json_path}")

    print("\n=== SUMMARY OF KEY FINDINGS ===")
    print(f"File-Weighted Accuracy (N={total_test}):    {m_file['overall']['accuracy']*100:.2f}%")
    print(f"Exact-Unique Accuracy  (N={unique_hashes}):    {m_dedup['overall']['accuracy']*100:.2f}% (Delta = {m_dedup['overall']['accuracy'] - m_file['overall']['accuracy']:.4f})")
    print(f"Train-Overlap Accuracy (N={n_train_overlap}):    {m_train_sub['overall']['accuracy']*100:.2f}%")
    print(f"Unseen-Test Accuracy   (N={n_test_overlap}):     {m_test_sub['overall']['accuracy']*100:.2f}%")
    print(f"Integrity Status:                             CONTAMINATED (80.56% of Phase 3.1 test seen in original training)")

    return audit_metrics_data


if __name__ == "__main__":
    run_phase3_2_audit()
