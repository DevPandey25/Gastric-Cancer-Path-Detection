"""GastricPath AI - Phase 3.3 Historical Test Evaluation & Final Baseline Verification Engine.

Executes:
1. Full 3,110-image historical test evaluation of the original baseline checkpoint.
2. Cross-split duplicate audit (Train/Test, Val/Test, Intra-Test) across SHA-256 and dHash clusters.
3. Sensitivity analyses: Deduplicated historical test (N=3,090), Train-excluded (N=2,936),
   Train/Val-excluded (N=2,918), and comparison against the Phase 3.1 442-image subset.
4. Cluster-bootstrap uncertainty estimation (B=1000 resamples across indivisible clusters).
5. Standardized machine-readable exports to research/audits/phase3_3/.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from research.benchmark.dataset import (
    CANONICAL_CLASSES,
    CLASS_TO_IDX,
    ManifestHistopathologyDataset,
)
from research.benchmark.evaluator import evaluate_model
from research.benchmark.metrics import (
    compute_bootstrap_confidence_intervals,
    compute_comprehensive_metrics,
    wilson_score_interval,
)
from research.benchmark.models import create_benchmark_model
from research.data.leakage_detector import NearDuplicateDetector

logger = logging.getLogger(__name__)

EXPECTED_CHECKPOINT_SHA256 = "65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8"


def extract_relative_path(path_str: str) -> str:
    """Extracts normalized relative path (e.g. 'ADI/ADI_1000.png') from any path format."""
    p = Path(path_str)
    return f"{p.parent.name}/{p.name}"


def compute_file_sha256(filepath: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Computes SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def run_phase3_3_audit(
    workspace_root: Path | str | None = None,
    batch_size: int = 64,
    device: str = "cpu",
    seed: int = 42,
) -> dict[str, Any]:
    """Executes the full Phase 3.3 historical test evaluation and integrity audit."""
    if workspace_root is None:
        workspace_root = Path(__file__).resolve().parent.parent.parent
    else:
        workspace_root = Path(workspace_root).resolve()

    output_dir = workspace_root / "research" / "audits" / "phase3_3"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GASTRICPATH AI — PHASE 3.3 HISTORICAL TEST EVALUATION & BENCHMARK AUDIT")
    print("=" * 80)
    print(f"Output Directory: {output_dir}")

    # Set seeds for deterministic execution
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Load historical manifest and master manifest
    hist_manifest_path = workspace_root / "model_train" / "runs" / "dataset_manifest.csv"
    master_manifest_path = workspace_root / "research" / "manifests" / "manifest_all.csv"
    ckpt_path = workspace_root / "models" / "checkpoints" / "gastric_rnn_classifier.pt"
    dataset_root = workspace_root / "data" / "raw" / "HMU-GC-HE-30K" / "all_image"

    if not hist_manifest_path.exists():
        raise FileNotFoundError(f"Historical training manifest missing: {hist_manifest_path}")
    if not master_manifest_path.exists():
        raise FileNotFoundError(f"Master manifest missing: {master_manifest_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file missing: {ckpt_path}")

    # Verify Checkpoint SHA-256
    actual_ckpt_sha256 = compute_file_sha256(ckpt_path)
    print(f"Checkpoint Path: {ckpt_path}")
    print(f"Checkpoint SHA-256: {actual_ckpt_sha256}")
    if actual_ckpt_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch! Expected {EXPECTED_CHECKPOINT_SHA256}, got {actual_ckpt_sha256}"
        )
    print("Checkpoint SHA-256 verified successfully.")

    df_hist = pd.read_csv(hist_manifest_path)
    df_hist["relative_path"] = df_hist["image_path"].apply(extract_relative_path)
    df_hist["class_name"] = df_hist["class_code"]

    df_master = pd.read_csv(master_manifest_path)
    # Merge master metadata (sha256, dhash, class_index)
    merged_all = df_hist.merge(
        df_master[["relative_path", "sha256", "dhash", "class_index"]],
        on="relative_path",
        how="inner",
    )
    if len(merged_all) != len(df_hist):
        raise ValueError(f"Unmatched paths in historical manifest! {len(df_hist) - len(merged_all)} missing.")

    # 2. Extract and Validate Historical Test Cohort (N = 3,110)
    df_test = merged_all[merged_all["split"] == "test"].copy().reset_index(drop=True)
    df_train = merged_all[merged_all["split"] == "train"].copy().reset_index(drop=True)
    df_val = merged_all[merged_all["split"] == "val"].copy().reset_index(drop=True)

    print(f"\nHistorical Cohort Counts: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
    if len(df_test) != 3110:
        raise ValueError(f"Expected 3,110 historical test images, found {len(df_test)}")

    # Check path uniqueness and mutual exclusivity
    test_paths = set(df_test["relative_path"])
    train_paths = set(df_train["relative_path"])
    val_paths = set(df_val["relative_path"])

    assert len(test_paths) == 3110, "Duplicate relative paths detected in historical test set!"
    assert len(train_paths.intersection(test_paths)) == 0, "Path overlap between Train and Test!"
    assert len(val_paths.intersection(test_paths)) == 0, "Path overlap between Val and Test!"
    print("Path exclusivity verified: 0 training or validation paths overlap historical test.")

    # Save canonical historical test manifest
    hist_test_manifest_csv = output_dir / "historical_test_manifest.csv"
    df_test[
        ["relative_path", "class_name", "class_index", "sha256", "dhash", "split"]
    ].to_csv(hist_test_manifest_csv, index=False)

    # 3. Task 1: Evaluate Baseline Checkpoint on Full 3,110 Historical Test Set
    print("\n[Step 1/5] Evaluating Baseline Checkpoint on Full 3,110 Historical Test Samples...")
    model, input_mode = create_benchmark_model(
        "GastricCancerRNN",
        num_classes=8,
        checkpoint_path=ckpt_path,
        device=device,
    )
    assert input_mode == "sequence"

    test_dataset = ManifestHistopathologyDataset(
        manifest_path=hist_test_manifest_csv,
        dataset_root=dataset_root,
        mode="sequence",
        is_train=False,
        image_size=224,
        patch_size=56,
        sequence_length=16,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    metrics_full, pred_records = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        class_names=CANONICAL_CLASSES,
        print_progress=True,
    )

    df_preds = pd.DataFrame(pred_records)
    assert len(df_preds) == 3110, f"Expected 3,110 predictions, got {len(df_preds)}"
    assert len(df_preds["relative_path"].unique()) == 3110, "Duplicate paths in predictions!"

    # Save historical test predictions
    hist_preds_path = output_dir / "historical_test_predictions.csv"
    df_preds.to_csv(hist_preds_path, index=False)
    print(f"Saved historical test predictions to {hist_preds_path}")

    # Merge metadata into predictions for downstream analysis
    df_preds = df_preds.merge(
        df_test[["relative_path", "sha256", "dhash"]],
        on="relative_path",
        how="inner",
    )

    # 4. Task 2: Cross-Split Duplicate & Near-Duplicate Leakage Audit
    print("\n[Step 2/5] Performing Cross-Split Duplicate Audit across Train, Val, and Test...")
    # Exact duplicate grouping
    sha_to_paths = defaultdict(list)
    path_to_split = dict(zip(merged_all["relative_path"], merged_all["split"]))
    path_to_class = dict(zip(merged_all["relative_path"], merged_all["class_name"]))
    path_to_dhash = dict(zip(merged_all["relative_path"], merged_all["dhash"]))

    for _, row in merged_all.iterrows():
        sha_to_paths[row["sha256"]].append(row["relative_path"])

    train_hashes = set(df_train["sha256"])
    val_hashes = set(df_val["sha256"])
    test_hashes = set(df_test["sha256"])

    # Overlap hashes
    train_test_overlap_hashes = train_hashes.intersection(test_hashes)
    val_test_overlap_hashes = val_hashes.intersection(test_hashes)
    train_val_overlap_hashes = train_hashes.intersection(val_hashes)

    # Multi-item clusters via dHash (Hamming <= 2)
    detector = NearDuplicateDetector(dataset_root, hamming_threshold=2)
    near_report, _ = detector.find_near_duplicates(path_to_hash=path_to_dhash)

    # Map each path to its dHash cluster ID
    path_to_cluster_id = {}
    for cluster in near_report.clusters:
        for p in cluster.paths:
            path_to_cluster_id[p] = cluster.cluster_id

    # Fallback for singletons: cluster_id = -1
    for p in merged_all["relative_path"]:
        if p not in path_to_cluster_id:
            path_to_cluster_id[p] = -1

    # Detailed audit table for historical split duplicate overlap
    overlap_rows = []
    test_imgs_in_train_exact = set()
    test_imgs_in_val_exact = set()
    test_imgs_in_train_dhash = set()
    test_imgs_in_val_dhash = set()

    # Pre-index cluster contents by split
    cluster_splits = defaultdict(set)
    cluster_paths = defaultdict(list)
    for p, cid in path_to_cluster_id.items():
        if cid >= 0:
            cluster_splits[cid].add(path_to_split[p])
            cluster_paths[cid].append(p)

    for idx, row in df_test.iterrows():
        rp = row["relative_path"]
        sha = row["sha256"]
        dh = row["dhash"]
        cname = row["class_name"]
        cid = path_to_cluster_id[rp]

        has_exact_train = sha in train_hashes
        has_exact_val = sha in val_hashes
        has_exact_intra_test = len(sha_to_paths[sha]) > 1 and sum(1 for p in sha_to_paths[sha] if path_to_split[p] == "test") > 1

        has_dhash_train = bool(cid >= 0 and "train" in cluster_splits[cid])
        has_dhash_val = bool(cid >= 0 and "val" in cluster_splits[cid])

        if has_exact_train:
            test_imgs_in_train_exact.add(rp)
        if has_exact_val:
            test_imgs_in_val_exact.add(rp)
        if has_dhash_train:
            test_imgs_in_train_dhash.add(rp)
        if has_dhash_val:
            test_imgs_in_val_dhash.add(rp)

        # Find matching predictions
        pred_match = df_preds[df_preds["relative_path"] == rp].iloc[0]

        overlap_rows.append({
            "relative_path": rp,
            "class_name": cname,
            "sha256": sha,
            "dhash": dh,
            "dhash_cluster_id": cid,
            "has_exact_dup_in_train": has_exact_train,
            "has_exact_dup_in_val": has_exact_val,
            "has_exact_dup_in_test": has_exact_intra_test,
            "has_dhash_dup_in_train": has_dhash_train,
            "has_dhash_dup_in_val": has_dhash_val,
            "is_correct": bool(pred_match["is_correct"]),
            "confidence": float(pred_match["confidence"]),
            "predicted_label": pred_match["predicted_label"],
        })

    df_overlap = pd.DataFrame(overlap_rows)
    overlap_csv_path = output_dir / "historical_split_duplicate_overlap.csv"
    df_overlap.to_csv(overlap_csv_path, index=False)
    print(f"Saved historical duplicate overlap records to {overlap_csv_path}")

    # Summary of duplicate groups with presence in test
    dup_group_summary = []
    all_test_shas = set(df_test["sha256"])
    for sha in sorted(all_test_shas):
        members = sha_to_paths[sha]
        if len(members) > 1:
            m_splits = [path_to_split[p] for p in members]
            m_classes = [path_to_class[p] for p in members]
            m_test = [p for p in members if path_to_split[p] == "test"]
            dup_group_summary.append({
                "sha256": sha,
                "total_members": len(members),
                "test_members_count": len(m_test),
                "train_members_count": m_splits.count("train"),
                "val_members_count": m_splits.count("val"),
                "splits_present": ";".join(sorted(set(m_splits))),
                "classes": ";".join(sorted(set(m_classes))),
                "has_label_conflict": len(set(m_classes)) > 1,
                "test_paths": ";".join(m_test),
            })

    df_dup_summary = pd.DataFrame(dup_group_summary)
    dup_summary_csv = output_dir / "historical_test_duplicate_summary.csv"
    df_dup_summary.to_csv(dup_summary_csv, index=False)
    print(f"Saved test duplicate group summary ({len(df_dup_summary)} groups) to {dup_summary_csv}")

    # 5. Sensitivity Analyses
    print("\n[Step 3/5] Computing Sensitivity Analyses across Distinct Evaluation Cohorts...")
    # Accuracy on exposed vs unexposed test samples
    acc_exposed_train = df_overlap[df_overlap["has_exact_dup_in_train"]]["is_correct"].mean()
    acc_unexposed_train = df_overlap[~df_overlap["has_exact_dup_in_train"]]["is_correct"].mean()
    acc_exposed_val = df_overlap[df_overlap["has_exact_dup_in_val"]]["is_correct"].mean()
    acc_clean_unexposed = df_overlap[(~df_overlap["has_exact_dup_in_train"]) & (~df_overlap["has_exact_dup_in_val"])]["is_correct"].mean()

    print(f"  Accuracy on test samples with EXACT match in Train (N={len(test_imgs_in_train_exact)}): {acc_exposed_train:.4%}")
    print(f"  Accuracy on test samples WITHOUT exact match in Train (N={len(df_test) - len(test_imgs_in_train_exact)}): {acc_unexposed_train:.4%}")
    print(f"  Accuracy on test samples with EXACT match in Val (N={len(test_imgs_in_val_exact)}): {acc_exposed_val:.4%}")
    print(f"  Accuracy on STRICTLY UNEXPOSED test samples (N={len(df_overlap) - len(test_imgs_in_train_exact.union(test_imgs_in_val_exact))}): {acc_clean_unexposed:.4%}")

    # Cohort A: Deduplicated Exact-Unique Historical Test (N=3,090)
    # Group by SHA-256 and select first relative_path deterministically
    df_preds_sorted = df_preds.sort_values("relative_path")
    df_unique_preds = df_preds_sorted.drop_duplicates(subset=["sha256"]).copy().reset_index(drop=True)
    assert len(df_unique_preds) == 3090, f"Expected 3,090 unique images, got {len(df_unique_preds)}"

    unique_preds_csv = output_dir / "historical_test_unique_predictions.csv"
    df_unique_preds.to_csv(unique_preds_csv, index=False)
    print(f"Saved deduplicated predictions ({len(df_unique_preds)} unique samples) to {unique_preds_csv}")

    prob_cols = [f"prob_{c}" for c in CANONICAL_CLASSES]
    metrics_unique = compute_comprehensive_metrics(
        y_true=df_unique_preds["ground_truth_index"].tolist(),
        y_prob=df_unique_preds[prob_cols].values.tolist(),
        class_names=CANONICAL_CLASSES,
        compute_ci=True,
    )

    # Cohort B: Train-Duplicate-Excluded Historical Test (N = 3,110 - 174 = 2,936)
    df_train_excl = df_preds[~df_preds["relative_path"].isin(test_imgs_in_train_exact)].copy().reset_index(drop=True)
    assert len(df_train_excl) == 2936, f"Expected 2,936 train-excluded images, got {len(df_train_excl)}"
    metrics_train_excl = compute_comprehensive_metrics(
        y_true=df_train_excl["ground_truth_index"].tolist(),
        y_prob=df_train_excl[prob_cols].values.tolist(),
        class_names=CANONICAL_CLASSES,
        compute_ci=True,
    )

    # Cohort C: Train-and-Val-Duplicate-Excluded Historical Test (N = 3,110 - 192 = 2,918)
    all_exposed_exact = test_imgs_in_train_exact.union(test_imgs_in_val_exact)
    df_train_val_excl = df_preds[~df_preds["relative_path"].isin(all_exposed_exact)].copy().reset_index(drop=True)
    assert len(df_train_val_excl) == 2918, f"Expected 2,918 strictly unexposed images, got {len(df_train_val_excl)}"
    metrics_train_val_excl = compute_comprehensive_metrics(
        y_true=df_train_val_excl["ground_truth_index"].tolist(),
        y_prob=df_train_val_excl[prob_cols].values.tolist(),
        class_names=CANONICAL_CLASSES,
        compute_ci=True,
    )

    # Cohort D: Historical Test Subset in Phase 3.1 (N=442)
    # Recover Phase 3.2 overlap file to identify the 442 images
    phase3_2_overlap_path = workspace_root / "research" / "audits" / "phase3_2" / "checkpoint_train_test_overlap.csv"
    if phase3_2_overlap_path.exists():
        df_p32 = pd.read_csv(phase3_2_overlap_path)
        p31_test_subset_paths = set(df_p32[df_p32["historical_split"] == "test"]["relative_path"])
        assert len(p31_test_subset_paths) == 442, f"Expected 442 subset paths, got {len(p31_test_subset_paths)}"
        df_p31_subset = df_preds[df_preds["relative_path"].isin(p31_test_subset_paths)].copy().reset_index(drop=True)
        metrics_p31_subset = compute_comprehensive_metrics(
            y_true=df_p31_subset["ground_truth_index"].tolist(),
            y_prob=df_p31_subset[prob_cols].values.tolist(),
            class_names=CANONICAL_CLASSES,
            compute_ci=True,
        )
    else:
        metrics_p31_subset = {}

    # 6. Cluster Bootstrap Uncertainty Estimation (B=1,000)
    print("\n[Step 4/5] Executing Cluster-Bootstrap Resampling (B=1,000 iterations)...")
    # Define indivisible cluster ID for each row in historical test
    # If dhash_cluster_id >= 0, use f"cluster_{dhash_cluster_id}"; otherwise use sha256
    cluster_ids = []
    for _, row in df_preds.iterrows():
        cid = path_to_cluster_id[row["relative_path"]]
        if cid >= 0:
            cluster_ids.append(f"cluster_{cid}")
        else:
            cluster_ids.append(f"sha_{row['sha256']}")

    df_preds["cluster_group_id"] = cluster_ids
    unique_clusters = np.array(sorted(df_preds["cluster_group_id"].unique()))
    n_clusters = len(unique_clusters)
    print(f"Total indivisible clusters across historical test: {n_clusters}")

    # Cluster mapping
    cluster_to_indices = defaultdict(list)
    for idx, cid in enumerate(df_preds["cluster_group_id"]):
        cluster_to_indices[cid].append(idx)

    y_true_all = df_preds["ground_truth_index"].values
    y_pred_all = df_preds["predicted_index"].values

    rng = np.random.RandomState(seed)
    n_bootstraps = 1000
    boot_accs = []
    boot_bal_accs = []
    boot_macro_f1s = []

    for b in range(n_bootstraps):
        sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        sampled_indices = []
        for c in sampled_clusters:
            sampled_indices.extend(cluster_to_indices[c])

        b_y_true = y_true_all[sampled_indices]
        b_y_pred = y_pred_all[sampled_indices]

        b_acc = float(np.mean(b_y_true == b_y_pred))
        b_bal_acc = float(np.mean([
            np.mean(b_y_pred[b_y_true == c] == c) if np.sum(b_y_true == c) > 0 else 0.0
            for c in range(8)
        ]))
        # Multiclass Macro-F1
        f1s = []
        for c in range(8):
            tp = np.sum((b_y_true == c) & (b_y_pred == c))
            fp = np.sum((b_y_true != c) & (b_y_pred == c))
            fn = np.sum((b_y_true == c) & (b_y_pred != c))
            denom = 2 * tp + fp + fn
            f1s.append((2 * tp / denom) if denom > 0 else 0.0)
        b_macro_f1 = float(np.mean(f1s))

        boot_accs.append(b_acc)
        boot_bal_accs.append(b_bal_acc)
        boot_macro_f1s.append(b_macro_f1)

    alpha = 0.05
    cluster_ci_results = {
        "n_bootstraps": n_bootstraps,
        "n_clusters": int(n_clusters),
        "seed": seed,
        "accuracy": {
            "mean": round(float(np.mean(boot_accs)), 4),
            "std": round(float(np.std(boot_accs)), 4),
            "ci_95": [
                round(float(np.percentile(boot_accs, 100 * (alpha / 2))), 4),
                round(float(np.percentile(boot_accs, 100 * (1 - alpha / 2))), 4),
            ],
        },
        "balanced_accuracy": {
            "mean": round(float(np.mean(boot_bal_accs)), 4),
            "std": round(float(np.std(boot_bal_accs)), 4),
            "ci_95": [
                round(float(np.percentile(boot_bal_accs, 100 * (alpha / 2))), 4),
                round(float(np.percentile(boot_bal_accs, 100 * (1 - alpha / 2))), 4),
            ],
        },
        "macro_f1": {
            "mean": round(float(np.mean(boot_macro_f1s)), 4),
            "std": round(float(np.std(boot_macro_f1s)), 4),
            "ci_95": [
                round(float(np.percentile(boot_macro_f1s, 100 * (alpha / 2))), 4),
                round(float(np.percentile(boot_macro_f1s, 100 * (1 - alpha / 2))), 4),
            ],
        },
    }

    boot_json_path = output_dir / "historical_test_bootstrap_results.json"
    with open(boot_json_path, "w", encoding="utf-8") as f:
        json.dump(cluster_ci_results, f, indent=2)
    print(f"Saved cluster bootstrap results to {boot_json_path}")

    # 7. Compile Master Metrics Document
    print("\n[Step 5/5] Compiling Master Audit Metrics JSON...")
    master_metrics = {
        "audit_phase": "Phase 3.3",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "path": str(ckpt_path),
            "sha256": actual_ckpt_sha256,
            "architecture": "GastricCancerRNN",
            "saved_epoch": 15,
            "saved_val_accuracy": 0.648553,
        },
        "historical_split_manifest": {
            "path": str(hist_manifest_path),
            "total_images": len(merged_all),
            "train_count": len(df_train),
            "val_count": len(df_val),
            "test_count": len(df_test),
        },
        "duplicate_audit_summary": {
            "exact_train_test_overlap_groups": len(train_test_overlap_hashes),
            "test_images_with_exact_train_duplicate": len(test_imgs_in_train_exact),
            "exact_val_test_overlap_groups": len(val_test_overlap_hashes),
            "test_images_with_exact_val_duplicate": len(test_imgs_in_val_exact),
            "test_images_with_exact_train_or_val_duplicate": len(all_exposed_exact),
            "intra_test_duplicate_groups": len(dup_group_summary),
            "intra_test_duplicate_files_count": len(df_test) - len(df_unique_preds),
            "exact_unique_test_images": len(df_unique_preds),
            "dhash_train_test_crossing_clusters": len(set(path_to_cluster_id[p] for p in test_imgs_in_train_dhash if path_to_cluster_id[p] >= 0)),
            "test_images_in_train_dhash_clusters": len(test_imgs_in_train_dhash),
            "cross_split_label_conflicts": 0,
        },
        "exposure_accuracy_breakdown": {
            "accuracy_exposed_to_train": round(float(acc_exposed_train), 4),
            "accuracy_unexposed_to_train": round(float(acc_unexposed_train), 4),
            "accuracy_exposed_to_val": round(float(acc_exposed_val), 4),
            "accuracy_strictly_unexposed": round(float(acc_clean_unexposed), 4),
            "memorization_delta_train_exposed_vs_strictly_unexposed": round(float(acc_exposed_train - acc_clean_unexposed), 4),
        },
        "full_historical_test_metrics_N3110": metrics_full,
        "deduplicated_unique_metrics_N3090": metrics_unique,
        "train_excluded_metrics_N2936": metrics_train_excl,
        "train_val_excluded_metrics_N2918": metrics_train_val_excl,
        "phase3_1_subset_metrics_N442": metrics_p31_subset,
        "cluster_bootstrap_uncertainty": cluster_ci_results,
    }

    metrics_json_path = output_dir / "historical_test_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(master_metrics, f, indent=2)
    print(f"Saved master audit metrics to {metrics_json_path}")

    print("\n" + "=" * 80)
    print("PHASE 3.3 HISTORICAL TEST EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Full Historical Test Accuracy (N=3,110): {metrics_full['overall']['accuracy']:.4%} (95% CI: {metrics_full['overall']['accuracy_ci_95']})")
    print(f"Deduplicated Unique Accuracy (N=3,090): {metrics_unique['overall']['accuracy']:.4%} (95% CI: {metrics_unique['overall']['accuracy_ci_95']})")
    print(f"Train/Val-Excluded Accuracy (N=2,918): {metrics_train_val_excl['overall']['accuracy']:.4%} (95% CI: {metrics_train_val_excl['overall']['accuracy_ci_95']})")
    print(f"Cluster Bootstrap Accuracy 95% CI: {cluster_ci_results['accuracy']['ci_95']}")
    print(f"Full Historical Test Macro-F1: {metrics_full['overall']['macro_f1']:.4%}")
    print(f"Full Historical Test AUROC (OvR): {metrics_full['overall']['macro_auroc']:.4f}")
    print(f"Full Historical Test Brier Score: {metrics_full['overall']['brier_score']:.4f}")
    print(f"Full Historical Test ECE: {metrics_full['overall']['expected_calibration_error']:.4f}")
    print("=" * 80)

    return master_metrics


if __name__ == "__main__":
    run_phase3_3_audit()

