"""GastricPath AI - Master Phase 2 Execution Script.

Optimized unified single-pass audit across all 31,096 images:
1. Dataset Integrity Scan (Dimensions, formats, corruption, RGB statistics)
2. Exact Duplicate Detection (Cryptographic SHA-256 checksums & cross-class conflict audit)
3. Near-Duplicate Detection (Perceptual dHash multi-index inverted table & cluster analysis)
4. Canonical Manifest Generation (Sorted, reproducible CSV)
5. Reproducible Stratified Splitting (70% Train / 15% Val / 15% Test with zero leakage)
"""

from __future__ import annotations

import io
import time
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image

from research.config import load_research_config
from research.data.integrity_scanner import (
    DatasetIntegrityScanner,
    DatasetIntegrityReport,
    ImageIntegrityRecord,
    SUPPORTED_EXTENSIONS,
)
from research.data.duplicate_detector import (
    ExactDuplicateDetector,
    ExactDuplicateReport,
    DuplicateGroup,
)
from research.data.leakage_detector import (
    NearDuplicateDetector,
    NearDuplicateReport,
    _calculate_dhash_from_img,
)
from research.data.manifest_generator import (
    DatasetManifestGenerator,
    ManifestRecord,
)
from research.data.split_generator import ReproducibleSplitGenerator


def run_phase2_audit():
    print("=" * 80)
    print("GASTRICPATH AI — PHASE 2 DATASET AUDIT & SPLIT PIPELINE")
    print("=" * 80)
    start_total = time.time()

    cfg = load_research_config()
    raw_dir = cfg.paths.raw_dataset_dir
    reports_dir = cfg.paths.reports_dir
    splits_dir = cfg.paths.splits_dir
    manifests_dir = cfg.paths.workspace_root / "research" / "manifests"

    reports_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset Root:    {raw_dir}")
    print(f"Reports Output:  {reports_dir}")
    print(f"Splits Output:   {splits_dir}")
    print(f"Manifest Output: {manifests_dir}")
    print("-" * 80)

    # Enumerate files deterministically
    print("[1/5] Scanning and hashing all images (Unified Streaming Pass)...")
    t0 = time.time()

    expected_classes = cfg.dataset.classes
    expected_size = (cfg.preprocessing.image_size, cfg.preprocessing.image_size)

    subdirs = [d.name for d in raw_dir.iterdir() if d.is_dir()]
    detected_classes = [c for c in expected_classes if c in subdirs]
    missing_classes = [c for c in expected_classes if c not in subdirs]
    unexpected_directories = [d for d in subdirs if d not in expected_classes]

    file_paths: list[Path] = []
    for class_name in expected_classes:
        class_folder = raw_dir / class_name
        if class_folder.exists() and class_folder.is_dir():
            for ext in SUPPORTED_EXTENSIONS:
                file_paths.extend(class_folder.glob(f"*{ext}"))
                file_paths.extend(class_folder.glob(f"*{ext.upper()}"))

    file_paths = sorted(list(set(file_paths)), key=lambda p: p.relative_to(raw_dir).as_posix())
    total_files = len(file_paths)
    print(f"  Found {total_files} candidate images across {len(detected_classes)} class folders.")

    records: list[ImageIntegrityRecord] = []
    manifest_records: list[ManifestRecord] = []
    path_to_sha: dict[str, str] = {}
    path_to_dhash: dict[str, str] = {}
    hash_to_paths: dict[str, list[Path]] = {}

    total_valid = 0
    total_corrupt = 0
    total_invalid_dimension = 0
    total_invalid_channels = 0
    corrupt_files: list[str] = []
    invalid_dimension_files: list[dict[str, Any]] = []

    class_counts: dict[str, int] = {c: 0 for c in expected_classes}
    dimensions_dist: dict[str, int] = {}
    format_dist: dict[str, int] = {}
    mode_dist: dict[str, int] = {}
    file_sizes: list[int] = []

    ch_sums = np.zeros(3, dtype=np.float64)
    ch_sq_sums = np.zeros(3, dtype=np.float64)
    ch_min = np.array([255.0, 255.0, 255.0], dtype=np.float64)
    ch_max = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    sampled_pixel_count = 0
    sample_step = 5  # 20% sample for RGB pixel stats (~6,200 images, >310M pixels)

    class_to_idx = {name: idx for idx, name in enumerate(expected_classes)}

    for idx, file_path in enumerate(file_paths):
        if idx % 5000 == 0 and idx > 0:
            print(f"  ... processed {idx}/{total_files} images ({idx/total_files*100:.1f}%) in {time.time() - t0:.1f}s")

        rel_path = file_path.relative_to(raw_dir).as_posix()
        class_name = file_path.parent.name
        file_size = file_path.stat().st_size
        file_sizes.append(file_size)

        is_valid = True
        error_reason: str | None = None
        fmt: str | None = None
        w, h, channels = 0, 0, 0
        mode: str | None = None
        sha_str = ""
        dh_str = ""

        try:
            file_bytes = file_path.read_bytes()
            if len(file_bytes) == 0:
                raise ValueError("Empty 0-byte file.")

            # Compute SHA-256
            sha_str = hashlib.sha256(file_bytes).hexdigest()
            path_to_sha[rel_path] = sha_str

            if sha_str not in hash_to_paths:
                hash_to_paths[sha_str] = []
            hash_to_paths[sha_str].append(file_path)

            # Open image with PIL
            with Image.open(io.BytesIO(file_bytes)) as img:
                fmt = img.format
                mode = img.mode
                w, h = img.size

                dim_key = f"{w}x{h}"
                dimensions_dist[dim_key] = dimensions_dist.get(dim_key, 0) + 1
                if fmt:
                    format_dist[fmt] = format_dist.get(fmt, 0) + 1
                if mode:
                    mode_dist[mode] = mode_dist.get(mode, 0) + 1

                if (w, h) != expected_size:
                    is_valid = False
                    error_reason = f"Unexpected dimensions {w}x{h}, expected {expected_size[0]}x{expected_size[1]}"
                    total_invalid_dimension += 1
                    invalid_dimension_files.append({"path": rel_path, "actual": f"{w}x{h}"})

                if mode == "RGB":
                    channels = 3
                elif mode == "RGBA":
                    # Fully convertible to 3-channel RGB without information loss (alpha is 255 constant)
                    channels = 4
                elif mode == "L":
                    channels = 1
                    is_valid = False
                    error_reason = "Grayscale mode (L, 1 channel) incompatible with expected RGB"
                    total_invalid_channels += 1
                else:
                    channels = len(img.getbands())
                    is_valid = False
                    error_reason = f"Unsupported image mode {mode}"
                    total_invalid_channels += 1

                # Compute dHash
                dh_str = _calculate_dhash_from_img(img, hash_size=8)
                path_to_dhash[rel_path] = dh_str

                # Sample for RGB stats
                if is_valid and (idx % sample_step == 0):
                    rgb_img = img.convert("RGB") if mode != "RGB" else img
                    arr = np.asarray(rgb_img, dtype=np.float64)
                    if arr.ndim == 3 and arr.shape[2] == 3:
                        ch_sums += arr.sum(axis=(0, 1))
                        ch_sq_sums += (arr ** 2).sum(axis=(0, 1))
                        ch_min = np.minimum(ch_min, arr.min(axis=(0, 1)))
                        ch_max = np.maximum(ch_max, arr.max(axis=(0, 1)))
                        sampled_pixel_count += w * h

        except Exception as exc:
            is_valid = False
            error_reason = f"Corrupt image file: {exc}"
            total_corrupt += 1
            corrupt_files.append(rel_path)

        if is_valid:
            total_valid += 1
            if class_name in class_counts:
                class_counts[class_name] += 1

        rec = ImageIntegrityRecord(
            relative_path=rel_path,
            class_name=class_name,
            is_valid=is_valid,
            file_size_bytes=file_size,
            format=fmt,
            width=w,
            height=h,
            channels=channels,
            mode=mode,
            error_reason=error_reason,
        )
        records.append(rec)

        manifest_records.append(
            ManifestRecord(
                relative_path=rel_path,
                class_name=class_name,
                class_index=class_to_idx.get(class_name, -1),
                width=w,
                height=h,
                channels=channels,
                file_size_bytes=file_size,
                sha256=sha_str,
                dhash=dh_str,
                dataset_id=cfg.dataset.name,
                audit_status="valid" if is_valid else "corrupt" if total_corrupt > 0 else "invalid",
            )
        )

    file_size_stats = {
        "min_bytes": float(np.min(file_sizes)),
        "max_bytes": float(np.max(file_sizes)),
        "mean_bytes": float(np.mean(file_sizes)),
        "median_bytes": float(np.median(file_sizes)),
        "total_megabytes": float(np.sum(file_sizes) / (1024 * 1024)),
    }

    rgb_stats: dict[str, dict[str, float]] = {}
    if sampled_pixel_count > 0:
        means = ch_sums / sampled_pixel_count
        variances = np.maximum(0.0, (ch_sq_sums / sampled_pixel_count) - (means ** 2))
        stds = np.sqrt(variances)
        channels_keys = ["channel_r", "channel_g", "channel_b"]
        for i, k in enumerate(channels_keys):
            rgb_stats[k] = {
                "mean": round(float(means[i]), 4),
                "std": round(float(stds[i]), 4),
                "min": round(float(ch_min[i]), 4),
                "max": round(float(ch_max[i]), 4),
            }

    report_integrity = DatasetIntegrityReport(
        dataset_id=cfg.dataset.name,
        dataset_root=str(raw_dir),
        scanned_at_utc=datetime.now(timezone.utc).isoformat(),
        total_files_scanned=total_files,
        total_valid=total_valid,
        total_corrupt=total_corrupt,
        total_invalid_dimension=total_invalid_dimension,
        total_invalid_channels=total_invalid_channels,
        expected_classes=expected_classes,
        detected_classes=detected_classes,
        missing_classes=missing_classes,
        unexpected_directories=unexpected_directories,
        class_counts=class_counts,
        dimensions_distribution=dimensions_dist,
        format_distribution=format_dist,
        mode_distribution=mode_dist,
        file_size_bytes_stats=file_size_stats,
        rgb_intensity_stats=rgb_stats,
        corrupt_files=corrupt_files,
        invalid_dimension_files=invalid_dimension_files,
    )

    integrity_json = reports_dir / "dataset_integrity_report.json"
    integrity_csv = reports_dir / "dataset_integrity_records.csv"
    report_integrity.to_json(integrity_json)
    report_integrity.to_csv(integrity_csv, records)
    print(f"  [Integrity] Completed in {time.time() - t0:.2f}s")
    print(f"  [Integrity] Total scanned: {report_integrity.total_files_scanned}")
    print(f"  [Integrity] Total valid:   {report_integrity.total_valid}")
    print(f"  [Integrity] Total corrupt: {report_integrity.total_corrupt}")
    print(f"  [Integrity] Formats:       {report_integrity.format_distribution}")
    print(f"  [Integrity] Modes:         {report_integrity.mode_distribution}")
    print(f"  [Integrity] Class counts:  {report_integrity.class_counts}")
    print(f"  [Integrity] RGB means:     R={rgb_stats['channel_r']['mean']}, G={rgb_stats['channel_g']['mean']}, B={rgb_stats['channel_b']['mean']}")

    # 2. Exact Duplicate Detection
    print("\n[2/5] Evaluating Exact Duplicates (SHA-256)...")
    t0 = time.time()
    total_unique_hashes = len(hash_to_paths)
    duplicate_groups: list[DuplicateGroup] = []
    cross_class_conflicts: list[dict[str, Any]] = []
    total_dup_files = 0
    cross_class_groups_count = 0
    intra_class_groups_count = 0

    for sha, paths in hash_to_paths.items():
        if len(paths) > 1:
            total_dup_files += len(paths)
            classes = [p.parent.name for p in paths]
            unique_classes = sorted(list(set(classes)))
            is_cross = len(unique_classes) > 1

            rel_paths = [p.relative_to(raw_dir).as_posix() for p in paths]
            group = DuplicateGroup(
                sha256=sha,
                count=len(paths),
                classes=classes,
                is_cross_class=is_cross,
                paths=rel_paths,
            )
            duplicate_groups.append(group)

            if is_cross:
                cross_class_groups_count += 1
                cross_class_conflicts.append({
                    "sha256": sha,
                    "conflicting_classes": unique_classes,
                    "file_count": len(paths),
                    "paths": rel_paths,
                })
            else:
                intra_class_groups_count += 1

    report_duplicates = ExactDuplicateReport(
        dataset_id=cfg.dataset.name,
        scanned_at_utc=datetime.now(timezone.utc).isoformat(),
        total_files_audited=total_files,
        total_unique_hashes=total_unique_hashes,
        total_duplicate_files=total_dup_files,
        total_duplicate_groups=len(duplicate_groups),
        cross_class_duplicate_groups=cross_class_groups_count,
        intra_class_duplicate_groups=intra_class_groups_count,
        cross_class_conflicts=cross_class_conflicts,
        duplicate_groups=duplicate_groups,
    )
    dup_json = reports_dir / "exact_duplicates_report.json"
    dup_csv = reports_dir / "exact_duplicates_report.csv"
    report_duplicates.to_json(dup_json)
    report_duplicates.to_csv(dup_csv)
    print(f"  [Duplicates] Completed in {time.time() - t0:.2f}s")
    print(f"  [Duplicates] Unique SHA-256 hashes:          {report_duplicates.total_unique_hashes}")
    print(f"  [Duplicates] Duplicate files:                {report_duplicates.total_duplicate_files}")
    print(f"  [Duplicates] Duplicate groups:               {report_duplicates.total_duplicate_groups}")
    print(f"  [Duplicates] Cross-class duplicate groups:   {report_duplicates.cross_class_duplicate_groups}")
    print(f"  [Duplicates] Intra-class duplicate groups:   {report_duplicates.intra_class_duplicate_groups}")

    # 3. Near-Duplicate Detection (dHash)
    print("\n[3/5] Evaluating Perceptual Near-Duplicates (dHash, Hamming threshold <= 2)...")
    t0 = time.time()
    leak_detector = NearDuplicateDetector(
        dataset_dir=raw_dir,
        hamming_threshold=2,
        hash_size=8,
        dataset_id=cfg.dataset.name,
    )
    report_near_dup, _ = leak_detector.find_near_duplicates(path_to_hash=path_to_dhash)
    near_dup_json = reports_dir / "near_duplicates_report.json"
    near_dup_csv = reports_dir / "near_duplicates_report.csv"
    report_near_dup.to_json(near_dup_json)
    report_near_dup.to_csv(near_dup_csv)
    print(f"  [Near-Duplicates] Completed in {time.time() - t0:.2f}s")
    print(f"  [Near-Duplicates] Total candidate edges:          {report_near_dup.total_candidate_pairs}")
    print(f"  [Near-Duplicates] Total near-duplicate clusters:  {report_near_dup.total_clusters}")
    print(f"  [Near-Duplicates] Cross-class clusters:           {report_near_dup.cross_class_clusters_count}")
    print(f"  [Near-Duplicates] Intra-class clusters:           {report_near_dup.intra_class_clusters_count}")

    # 4. Canonical Dataset Manifest Generation
    print("\n[4/5] Generating Canonical Dataset Manifest...")
    t0 = time.time()
    manifest_records.sort(key=lambda r: r.relative_path)
    manifest_csv = manifests_dir / "manifest_all.csv"
    manifest_sha = DatasetManifestGenerator.save_csv(manifest_records, manifest_csv)
    print(f"  [Manifest] Completed in {time.time() - t0:.2f}s")
    print(f"  [Manifest] Total entries: {len(manifest_records)}")
    print(f"  [Manifest] Saved to:      {manifest_csv}")
    print(f"  [Manifest] SHA-256:       {manifest_sha}")

    # 5. Reproducible Stratified Splitting with Zero Leakage
    print("\n[5/5] Generating Reproducible Group-Stratified Splits (70% / 15% / 15%)...")
    t0 = time.time()
    group_map: dict[str, str] = {}
    for cluster in report_near_dup.clusters:
        cid = f"cluster_{cluster.cluster_id}"
        for p in cluster.paths:
            group_map[p] = cid

    splitter = ReproducibleSplitGenerator(
        seed=cfg.dataset.split_seed,
        split_ratios=(0.70, 0.15, 0.15),
        splits_dir=splits_dir,
    )
    summary, train_recs, val_recs, test_recs = splitter.generate_splits(
        manifest_records,
        manifest_source_hash=manifest_sha,
        duplicate_groups_map=group_map,
    )
    print(f"  [Splits] Completed in {time.time() - t0:.2f}s")
    print(f"  [Splits] Train samples:    {summary.train_count} ({summary.actual_proportions['train']}%)")
    print(f"  [Splits] Val samples:      {summary.val_count} ({summary.actual_proportions['val']}%)")
    print(f"  [Splits] Test samples:     {summary.test_count} ({summary.actual_proportions['test']}%)")
    print(f"  [Splits] Excluded samples: {summary.excluded_count}")
    print(f"  [Splits] Split files:      {list(summary.split_manifest_hashes.keys())}")
    print(f"  [Splits] Mutual exclusivity verified: {summary.leakage_safeguards['mutual_exclusivity_verified']}")
    print(f"  [Splits] Split overlap count:        {summary.leakage_safeguards['split_overlap_count']}")

    total_elapsed = time.time() - start_total
    print("\n" + "=" * 80)
    print(f"PHASE 2 AUDIT PIPELINE COMPLETED SUCCESSFULLY IN {total_elapsed:.2f} SECONDS")
    print("=" * 80)


if __name__ == "__main__":
    run_phase2_audit()
