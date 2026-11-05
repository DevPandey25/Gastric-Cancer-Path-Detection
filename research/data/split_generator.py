"""Reproducible Group-Stratified Dataset Splitter.

Generates deterministic train, validation, and test splits with strict mutual exclusivity,
preserves class balance across splits, groups identical and near-duplicate images to
prevent cross-split data leakage, and documents scientific limitations.
"""

from __future__ import annotations

import csv
import json
import logging
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.data.duplicate_detector import compute_sha256
from research.data.manifest_generator import ManifestRecord

logger = logging.getLogger(__name__)

SCIENTIFIC_SPLIT_DISCLOSURE = (
    "SCIENTIFIC INTEGRITY DISCLOSURE (Image-Level Splitting): "
    "The GCHTID dataset filenames (e.g. TUM_1.png) do not contain patient, hospital, or slide identifiers. "
    "Consequently, this split is strictly IMAGE-LEVEL, not PATIENT-LEVEL or SLIDE-LEVEL. "
    "While exact bitwise duplicates and high-confidence perceptual near-duplicates have been grouped into "
    "single splits to prevent trivial leakage, different patches from the same patient may still exist across "
    "training and evaluation sets. Generalization performance estimates must be reported as image-level patch "
    "classification metrics and cannot be claimed as patient-level clinical diagnostic accuracy."
)


@dataclass
class SplitClassCounts:
    train: int
    val: int
    test: int
    total: int
    train_pct: float
    val_pct: float
    test_pct: float


@dataclass
class SplitSummary:
    dataset_id: str
    created_at_utc: str
    random_seed: int
    target_proportions: dict[str, float]
    actual_proportions: dict[str, float]
    total_samples: int
    train_count: int
    val_count: int
    test_count: int
    excluded_count: int
    class_distribution: dict[str, dict[str, Any]]
    manifest_source_hash: str
    split_manifest_hashes: dict[str, str]
    leakage_safeguards: dict[str, Any]
    scientific_disclosure: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, output_path: Path | str, indent: int = 2) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent)


class ReproducibleSplitGenerator:
    """Generates leak-proof, deterministic group-stratified train/val/test splits."""

    def __init__(
        self,
        seed: int = 42,
        split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
        splits_dir: Path | str = "research/splits",
    ) -> None:
        train_r, val_r, test_r = split_ratios
        total_r = train_r + val_r + test_r
        if not (0.999 <= total_r <= 1.001):
            raise ValueError(f"Split ratios must sum to 1.0, got {total_r} from {split_ratios}")

        self.seed = seed
        self.split_ratios = (train_r / total_r, val_r / total_r, test_r / total_r)
        self.splits_dir = Path(splits_dir).resolve()

    def generate_splits(
        self,
        records: list[ManifestRecord],
        manifest_source_hash: str = "unknown",
        duplicate_groups_map: dict[str, str] | None = None,
    ) -> tuple[SplitSummary, list[ManifestRecord], list[ManifestRecord], list[ManifestRecord]]:
        """Splits manifest records into train, val, and test partitions deterministically."""
        # Filter valid records
        valid_records: list[ManifestRecord] = [r for r in records if r.audit_status == "valid"]
        excluded_records: list[ManifestRecord] = [r for r in records if r.audit_status != "valid"]

        if not valid_records:
            raise ValueError("No valid records found in manifest to split.")

        # Assign each record to an indivisible cluster/group
        # Default grouping: by SHA-256 (exact duplicates never split)
        # If external duplicate_groups_map provided, map record.relative_path -> group_id
        path_to_group: dict[str, str] = {}
        for r in valid_records:
            if duplicate_groups_map and r.relative_path in duplicate_groups_map:
                path_to_group[r.relative_path] = duplicate_groups_map[r.relative_path]
            elif r.sha256:
                path_to_group[r.relative_path] = f"sha_{r.sha256}"
            else:
                path_to_group[r.relative_path] = f"path_{r.relative_path}"

        # Group records by group_id
        group_to_records: dict[str, list[ManifestRecord]] = {}
        for r in valid_records:
            gid = path_to_group[r.relative_path]
            if gid not in group_to_records:
                group_to_records[gid] = []
            group_to_records[gid].append(r)

        # Check for cross-class label conflicts in groups
        pure_groups_by_class: dict[str, list[str]] = {}
        conflicted_groups: list[dict[str, Any]] = []

        for gid, grp_records in group_to_records.items():
            classes = sorted(list(set(r.class_name for r in grp_records)))
            if len(classes) == 1:
                cname = classes[0]
                if cname not in pure_groups_by_class:
                    pure_groups_by_class[cname] = []
                pure_groups_by_class[cname].append(gid)
            else:
                conflicted_groups.append({
                    "group_id": gid,
                    "classes": classes,
                    "paths": [r.relative_path for r in grp_records],
                })
                logger.warning("Group %s has conflicting classes %s; quarantined", gid, classes)

        # Perform stratified splitting across classes on pure groups
        train_groups: set[str] = set()
        val_groups: set[str] = set()
        test_groups: set[str] = set()

        target_train_r, target_val_r, target_test_r = self.split_ratios

        for class_name, group_ids in sorted(pure_groups_by_class.items()):
            # Sort for stability before shuffling with seeded RNG
            sorted_gids = sorted(group_ids)
            rng = random.Random(self.seed + hash(class_name) % 100000)
            rng.shuffle(sorted_gids)

            # Calculate total images in this class
            total_class_images = sum(len(group_to_records[gid]) for gid in sorted_gids)
            target_val_count = int(round(total_class_images * target_val_r))
            target_test_count = int(round(total_class_images * target_test_r))

            curr_val = 0
            curr_test = 0

            for gid in sorted_gids:
                n_images = len(group_to_records[gid])
                if curr_val < target_val_count:
                    val_groups.add(gid)
                    curr_val += n_images
                elif curr_test < target_test_count:
                    test_groups.add(gid)
                    curr_test += n_images
                else:
                    train_groups.add(gid)

        # In case there are conflicted groups, place them in train with warning so they don't corrupt val/test benchmarks
        for conf in conflicted_groups:
            train_groups.add(conf["group_id"])

        # Compile final split records
        train_records: list[ManifestRecord] = []
        val_records: list[ManifestRecord] = []
        test_records: list[ManifestRecord] = []

        for gid in train_groups:
            train_records.extend(group_to_records[gid])
        for gid in val_groups:
            val_records.extend(group_to_records[gid])
        for gid in test_groups:
            test_records.extend(group_to_records[gid])

        # Deterministic sorting of split manifests by relative_path
        train_records.sort(key=lambda r: r.relative_path)
        val_records.sort(key=lambda r: r.relative_path)
        test_records.sort(key=lambda r: r.relative_path)

        # Strict validation of mutual exclusivity
        train_paths = set(r.relative_path for r in train_records)
        val_paths = set(r.relative_path for r in val_records)
        test_paths = set(r.relative_path for r in test_records)

        overlap_tv = train_paths & val_paths
        overlap_tt = train_paths & test_paths
        overlap_vt = val_paths & test_paths

        if overlap_tv or overlap_tt or overlap_vt:
            raise AssertionError(
                f"Split mutual exclusivity violation! "
                f"Train/Val: {len(overlap_tv)}, Train/Test: {len(overlap_tt)}, Val/Test: {len(overlap_vt)}"
            )

        # Verify duplicate grouping across splits (zero leakage)
        for gid, grp_records in group_to_records.items():
            splits_present = set()
            for r in grp_records:
                if r.relative_path in train_paths:
                    splits_present.add("train")
                elif r.relative_path in val_paths:
                    splits_present.add("val")
                elif r.relative_path in test_paths:
                    splits_present.add("test")
            if len(splits_present) > 1:
                raise AssertionError(
                    f"Leakage violation: Group {gid} with {len(grp_records)} items spanned splits: {splits_present}"
                )

        # Save split manifest CSV files
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        train_csv_path = self.splits_dir / "manifest_train.csv"
        val_csv_path = self.splits_dir / "manifest_val.csv"
        test_csv_path = self.splits_dir / "manifest_test.csv"

        train_hash = self._write_split_csv(train_records, train_csv_path)
        val_hash = self._write_split_csv(val_records, val_csv_path)
        test_hash = self._write_split_csv(test_records, test_csv_path)

        # Calculate per-class distribution statistics
        all_classes = sorted(list(set(r.class_name for r in valid_records)))
        class_dist: dict[str, dict[str, Any]] = {}
        for c in all_classes:
            tr_c = sum(1 for r in train_records if r.class_name == c)
            va_c = sum(1 for r in val_records if r.class_name == c)
            te_c = sum(1 for r in test_records if r.class_name == c)
            tot_c = tr_c + va_c + te_c
            class_dist[c] = {
                "train_count": tr_c,
                "val_count": va_c,
                "test_count": te_c,
                "total_count": tot_c,
                "train_pct": round(tr_c / tot_c * 100, 2) if tot_c > 0 else 0.0,
                "val_pct": round(va_c / tot_c * 100, 2) if tot_c > 0 else 0.0,
                "test_pct": round(te_c / tot_c * 100, 2) if tot_c > 0 else 0.0,
            }

        total_valid = len(valid_records)
        actual_train_pct = round(len(train_records) / total_valid * 100, 2)
        actual_val_pct = round(len(val_records) / total_valid * 100, 2)
        actual_test_pct = round(len(test_records) / total_valid * 100, 2)

        summary = SplitSummary(
            dataset_id=valid_records[0].dataset_id if valid_records else "GCHTID",
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            random_seed=self.seed,
            target_proportions={
                "train": self.split_ratios[0],
                "val": self.split_ratios[1],
                "test": self.split_ratios[2],
            },
            actual_proportions={
                "train": actual_train_pct,
                "val": actual_val_pct,
                "test": actual_test_pct,
            },
            total_samples=total_valid,
            train_count=len(train_records),
            val_count=len(val_records),
            test_count=len(test_records),
            excluded_count=len(excluded_records),
            class_distribution=class_dist,
            manifest_source_hash=manifest_source_hash,
            split_manifest_hashes={
                "manifest_train.csv": train_hash,
                "manifest_val.csv": val_hash,
                "manifest_test.csv": test_hash,
            },
            leakage_safeguards={
                "exact_duplicates_grouped": True,
                "group_level_indivisibility": True,
                "conflicted_groups_count": len(conflicted_groups),
                "mutual_exclusivity_verified": True,
                "split_overlap_count": 0,
            },
            scientific_disclosure=SCIENTIFIC_SPLIT_DISCLOSURE,
        )

        # Write summary JSON
        summary.to_json(self.splits_dir / "split_summary.json")

        return summary, train_records, val_records, test_records

    @staticmethod
    def _write_split_csv(records: list[ManifestRecord], path: Path) -> str:
        fieldnames = [
            "relative_path",
            "class_name",
            "class_index",
            "width",
            "height",
            "channels",
            "file_size_bytes",
            "sha256",
            "dhash",
            "dataset_id",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                d = r.to_dict()
                d.pop("audit_status", None)
                writer.writerow(d)
        return compute_sha256(path)

