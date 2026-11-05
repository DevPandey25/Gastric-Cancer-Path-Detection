"""Dataset Integrity Scanner for Histopathology Research.

Recursively validates dataset image readability, verifies format, dimensions,
channel depth, validates RGB conversion and channel consistency, detects corrupt
or truncated files, gathers file size and dimension distributions, and computes
channel-wise RGB intensity statistics.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class ImageIntegrityRecord:
    relative_path: str
    class_name: str
    is_valid: bool
    file_size_bytes: int
    format: str | None = None
    width: int | None = None
    height: int | None = None
    channels: int | None = None
    mode: str | None = None
    error_reason: str | None = None


@dataclass
class DatasetIntegrityReport:
    dataset_id: str
    dataset_root: str
    scanned_at_utc: str
    total_files_scanned: int
    total_valid: int
    total_corrupt: int
    total_invalid_dimension: int
    total_invalid_channels: int
    expected_classes: list[str]
    detected_classes: list[str]
    missing_classes: list[str]
    unexpected_directories: list[str]
    class_counts: dict[str, int]
    dimensions_distribution: dict[str, int]
    format_distribution: dict[str, int]
    mode_distribution: dict[str, int]
    file_size_bytes_stats: dict[str, float]
    rgb_intensity_stats: dict[str, dict[str, float]]
    corrupt_files: list[str]
    invalid_dimension_files: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, output_path: Path | str, indent: int = 2) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent)

    def to_csv(self, output_path: Path | str, records: list[ImageIntegrityRecord]) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "relative_path",
            "class_name",
            "is_valid",
            "file_size_bytes",
            "format",
            "width",
            "height",
            "channels",
            "mode",
            "error_reason",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in records:
                writer.writerow(asdict(rec))


class DatasetIntegrityScanner:
    """Read-only scanner to audit histopathology image directories."""

    def __init__(
        self,
        dataset_dir: Path | str,
        expected_classes: list[str] | None = None,
        expected_size: tuple[int, int] = (224, 224),
        expected_channels: int = 3,
        allow_alpha_channel: bool = True,
        dataset_id: str = "GCHTID",
    ) -> None:
        self.dataset_dir = Path(dataset_dir).resolve()
        self.expected_classes = (
            expected_classes
            if expected_classes is not None
            else ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]
        )
        self.expected_size = expected_size
        self.expected_channels = expected_channels
        self.allow_alpha_channel = allow_alpha_channel
        self.dataset_id = dataset_id

    def scan(
        self,
        compute_rgb_stats: bool = True,
        rgb_sample_rate: float = 1.0,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[DatasetIntegrityReport, list[ImageIntegrityRecord]]:
        """Scans the dataset directory, returning the summary report and per-file records."""
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.dataset_dir}")

        # Directory structure verification
        subdirs = [d.name for d in self.dataset_dir.iterdir() if d.is_dir()]
        detected_classes = [c for c in self.expected_classes if c in subdirs]
        missing_classes = [c for c in self.expected_classes if c not in subdirs]
        unexpected_directories = [d for d in subdirs if d not in self.expected_classes]

        # Enumerate candidate files deterministically
        file_paths: list[Path] = []
        for class_name in self.expected_classes:
            class_folder = self.dataset_dir / class_name
            if class_folder.exists() and class_folder.is_dir():
                for ext in SUPPORTED_EXTENSIONS:
                    file_paths.extend(class_folder.glob(f"*{ext}"))
                    file_paths.extend(class_folder.glob(f"*{ext.upper()}"))

        # Sort file paths to eliminate OS enumeration variance
        file_paths = sorted(list(set(file_paths)))
        total_files = len(file_paths)

        records: list[ImageIntegrityRecord] = []
        total_valid = 0
        total_corrupt = 0
        total_invalid_dimension = 0
        total_invalid_channels = 0
        corrupt_files: list[str] = []
        invalid_dimension_files: list[dict[str, Any]] = []

        class_counts: dict[str, int] = {c: 0 for c in self.expected_classes}
        dimensions_dist: dict[str, int] = {}
        format_dist: dict[str, int] = {}
        mode_dist: dict[str, int] = {}
        file_sizes: list[int] = []

        # RGB stats accumulators (int64/float64 for numeric stability)
        ch_sums = np.zeros(3, dtype=np.float64)
        ch_sq_sums = np.zeros(3, dtype=np.float64)
        ch_min = np.array([255.0, 255.0, 255.0], dtype=np.float64)
        ch_max = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        sampled_pixel_count = 0

        # Deterministic sample step for RGB stats if sample rate < 1.0
        step = max(1, int(1.0 / max(0.0001, min(1.0, rgb_sample_rate))))

        for idx, file_path in enumerate(file_paths):
            if progress_callback and idx % 2000 == 0:
                progress_callback(idx, total_files)

            rel_path = file_path.relative_to(self.dataset_dir).as_posix()
            class_name = file_path.parent.name
            file_size = file_path.stat().st_size
            file_sizes.append(file_size)

            is_valid = True
            error_reason: str | None = None
            fmt: str | None = None
            w: int | None = None
            h: int | None = None
            channels: int | None = None
            mode: str | None = None

            try:
                # Read bytes and open with PIL to check readability
                file_bytes = file_path.read_bytes()
                if len(file_bytes) == 0:
                    raise ValueError("File is 0 bytes (empty file).")

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

                    # Check dimensions
                    if (w, h) != self.expected_size:
                        is_valid = False
                        error_reason = f"Unexpected dimensions {w}x{h}, expected {self.expected_size[0]}x{self.expected_size[1]}"
                        total_invalid_dimension += 1
                        invalid_dimension_files.append({"path": rel_path, "actual": f"{w}x{h}"})

                    # Channel & RGB convertibility check
                    if mode == "RGB":
                        channels = 3
                    elif mode == "RGBA" and self.allow_alpha_channel:
                        # Fully convertible to 3-channel RGB without data loss
                        channels = 4
                    elif mode == "RGBA" and not self.allow_alpha_channel:
                        is_valid = False
                        channels = 4
                        err = f"RGBA mode encountered but alpha channel is disallowed"
                        error_reason = f"{error_reason}; {err}" if error_reason else err
                        total_invalid_channels += 1
                    elif mode == "L":
                        is_valid = False
                        channels = 1
                        err = f"Grayscale mode (L, 1 channel) incompatible with expected RGB ({self.expected_channels} channels)"
                        error_reason = f"{error_reason}; {err}" if error_reason else err
                        total_invalid_channels += 1
                    else:
                        is_valid = False
                        channels = len(img.getbands())
                        err = f"Unsupported image mode {mode} with {channels} bands"
                        error_reason = f"{error_reason}; {err}" if error_reason else err
                        total_invalid_channels += 1

                    # Accumulate RGB stats on valid images (converting RGBA to RGB for uniform statistics)
                    if is_valid and compute_rgb_stats and (idx % step == 0):
                        rgb_img = img.convert("RGB") if mode != "RGB" else img
                        arr = np.asarray(rgb_img, dtype=np.float64)
                        if arr.ndim == 3 and arr.shape[2] == 3:
                            ch_sums += arr.sum(axis=(0, 1))
                            ch_sq_sums += (arr ** 2).sum(axis=(0, 1))
                            ch_min = np.minimum(ch_min, arr.min(axis=(0, 1)))
                            ch_max = np.maximum(ch_max, arr.max(axis=(0, 1)))
                            sampled_pixel_count += w * h

            except (UnidentifiedImageError, OSError, ValueError, Exception) as exc:
                is_valid = False
                error_reason = f"Corrupt image file: {exc}"
                total_corrupt += 1
                corrupt_files.append(rel_path)

            if is_valid:
                total_valid += 1
                if class_name in class_counts:
                    class_counts[class_name] += 1
            else:
                logger.warning("Integrity check failed for %s: %s", rel_path, error_reason)

            records.append(
                ImageIntegrityRecord(
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
            )

        if progress_callback:
            progress_callback(total_files, total_files)

        # File size distribution summary
        file_size_stats: dict[str, float] = {}
        if file_sizes:
            file_size_stats = {
                "min_bytes": float(np.min(file_sizes)),
                "max_bytes": float(np.max(file_sizes)),
                "mean_bytes": float(np.mean(file_sizes)),
                "median_bytes": float(np.median(file_sizes)),
                "total_megabytes": float(np.sum(file_sizes) / (1024 * 1024)),
            }

        # Finalize RGB intensity stats
        rgb_stats: dict[str, dict[str, float]] = {
            "channel_r": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
            "channel_g": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
            "channel_b": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
        }
        if compute_rgb_stats and sampled_pixel_count > 0:
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

        report = DatasetIntegrityReport(
            dataset_id=self.dataset_id,
            dataset_root=str(self.dataset_dir),
            scanned_at_utc=datetime.now(timezone.utc).isoformat(),
            total_files_scanned=total_files,
            total_valid=total_valid,
            total_corrupt=total_corrupt,
            total_invalid_dimension=total_invalid_dimension,
            total_invalid_channels=total_invalid_channels,
            expected_classes=self.expected_classes,
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

        return report, records
