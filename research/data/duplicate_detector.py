"""Exact Duplicate Detection for Histopathology Images.

Computes cryptographic SHA-256 checksums to identify bitwise-identical images,
detects duplicate clusters within classes, and flags critical cross-class
duplicate anomalies with conflicting ground-truth labels.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


def compute_sha256(file_path: Path | str, chunk_size: int = 65536) -> str:
    """Computes SHA-256 hash using chunked streaming to prevent high memory usage."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class DuplicateGroup:
    sha256: str
    count: int
    classes: list[str]
    is_cross_class: bool
    paths: list[str]


@dataclass
class ExactDuplicateReport:
    dataset_id: str
    scanned_at_utc: str
    total_files_audited: int
    total_unique_hashes: int
    total_duplicate_files: int
    total_duplicate_groups: int
    cross_class_duplicate_groups: int
    intra_class_duplicate_groups: int
    cross_class_conflicts: list[dict[str, Any]]
    duplicate_groups: list[DuplicateGroup]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "scanned_at_utc": self.scanned_at_utc,
            "total_files_audited": self.total_files_audited,
            "total_unique_hashes": self.total_unique_hashes,
            "total_duplicate_files": self.total_duplicate_files,
            "total_duplicate_groups": self.total_duplicate_groups,
            "cross_class_duplicate_groups": self.cross_class_duplicate_groups,
            "intra_class_duplicate_groups": self.intra_class_duplicate_groups,
            "cross_class_conflicts": self.cross_class_conflicts,
            "duplicate_groups": [asdict(g) for g in self.duplicate_groups],
        }

    def to_json(self, output_path: Path | str, indent: int = 2) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent)

    def to_csv(self, output_path: Path | str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sha256", "count", "is_cross_class", "classes", "paths"])
            for g in self.duplicate_groups:
                writer.writerow([
                    g.sha256,
                    g.count,
                    g.is_cross_class,
                    ";".join(sorted(set(g.classes))),
                    ";".join(g.paths),
                ])


class ExactDuplicateDetector:
    """Audits image collections for bitwise exact duplicates and conflicting labels."""

    def __init__(self, dataset_dir: Path | str, dataset_id: str = "GCHTID") -> None:
        self.dataset_dir = Path(dataset_dir).resolve()
        self.dataset_id = dataset_id

    def find_duplicates(
        self,
        file_paths: list[Path] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[ExactDuplicateReport, dict[str, str]]:
        """Scans files, computes SHA-256 hashes, and returns duplicate report and hash map."""
        if file_paths is None:
            if not self.dataset_dir.exists():
                raise FileNotFoundError(f"Dataset directory does not exist: {self.dataset_dir}")
            file_paths = []
            for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
                file_paths.extend(self.dataset_dir.rglob(f"*{ext}"))
                file_paths.extend(self.dataset_dir.rglob(f"*{ext.upper()}"))
            file_paths = sorted(list(set(file_paths)))

        total_files = len(file_paths)
        hash_to_paths: dict[str, list[Path]] = {}
        path_to_hash: dict[str, str] = {}

        for idx, path in enumerate(file_paths):
            if progress_callback and idx % 1000 == 0:
                progress_callback(idx, total_files)

            rel_str = path.relative_to(self.dataset_dir).as_posix()
            sha = compute_sha256(path)
            path_to_hash[rel_str] = sha

            if sha not in hash_to_paths:
                hash_to_paths[sha] = []
            hash_to_paths[sha].append(path)

        if progress_callback:
            progress_callback(total_files, total_files)

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

                rel_paths = [p.relative_to(self.dataset_dir).as_posix() for p in paths]
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
                    logger.critical(
                        "Cross-class duplicate detected: SHA %s shared across classes %s",
                        sha[:12],
                        unique_classes,
                    )
                else:
                    intra_class_groups_count += 1

        report = ExactDuplicateReport(
            dataset_id=self.dataset_id,
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

        return report, path_to_hash

