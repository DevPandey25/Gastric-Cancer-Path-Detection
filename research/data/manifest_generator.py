"""Dataset Manifest Generator for Histopathology Research.

Creates canonical, sorted, reproducible manifests recording relative file paths,
tissue classes, numeric indices, dimensions, file sizes, SHA-256 hashes,
dHash values, and audit statuses.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from research.data.duplicate_detector import compute_sha256
from research.data.leakage_detector import compute_dhash

logger = logging.getLogger(__name__)

DEFAULT_CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]


@dataclass
class ManifestRecord:
    relative_path: str
    class_name: str
    class_index: int
    width: int
    height: int
    channels: int
    file_size_bytes: int
    sha256: str
    dhash: str
    dataset_id: str
    audit_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatasetManifestGenerator:
    """Generates canonical and reproducible dataset manifests."""

    def __init__(
        self,
        dataset_dir: Path | str,
        class_names: list[str] | None = None,
        dataset_id: str = "GCHTID",
        expected_size: tuple[int, int] = (224, 224),
    ) -> None:
        self.dataset_dir = Path(dataset_dir).resolve()
        self.class_names = class_names if class_names is not None else DEFAULT_CLASSES
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.dataset_id = dataset_id
        self.expected_size = expected_size

    def generate_manifest(
        self,
        file_paths: list[Path] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[ManifestRecord]:
        """Scans and builds deterministic manifest records sorted by relative_path."""
        if file_paths is None:
            if not self.dataset_dir.exists():
                raise FileNotFoundError(f"Dataset directory does not exist: {self.dataset_dir}")
            file_paths = []
            for class_name in self.class_names:
                class_folder = self.dataset_dir / class_name
                if class_folder.exists() and class_folder.is_dir():
                    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
                        file_paths.extend(class_folder.glob(f"*{ext}"))
                        file_paths.extend(class_folder.glob(f"*{ext.upper()}"))

        # Strict deterministic sort by relative POSIX path
        file_paths = sorted(
            list(set(file_paths)),
            key=lambda p: p.relative_to(self.dataset_dir).as_posix(),
        )
        total_files = len(file_paths)
        records: list[ManifestRecord] = []

        for idx, file_path in enumerate(file_paths):
            if progress_callback and idx % 1000 == 0:
                progress_callback(idx, total_files)

            rel_path = file_path.relative_to(self.dataset_dir).as_posix()
            class_name = file_path.parent.name
            class_index = self.class_to_idx.get(class_name, -1)
            file_size = file_path.stat().st_size

            status = "valid"
            w, h, channels = 0, 0, 0
            sha = ""
            dh = ""

            try:
                sha = compute_sha256(file_path)
                file_bytes = file_path.read_bytes()
                with Image.open(io.BytesIO(file_bytes)) as img:
                    w, h = img.size
                    mode = img.mode
                    if mode == "RGB":
                        channels = 3
                    elif mode == "RGBA":
                        channels = 4
                    elif mode == "L":
                        channels = 1
                    else:
                        channels = len(img.getbands())

                    if (w, h) != self.expected_size:
                        status = "invalid_dimension"
                    elif mode not in ("RGB", "RGBA"):
                        status = "invalid_channels"

                    dh = compute_dhash(img, hash_size=8)
            except Exception as exc:
                status = "corrupt"
                logger.warning("Manifest generation error on %s: %s", rel_path, exc)

            record = ManifestRecord(
                relative_path=rel_path,
                class_name=class_name,
                class_index=class_index,
                width=w,
                height=h,
                channels=channels,
                file_size_bytes=file_size,
                sha256=sha,
                dhash=dh,
                dataset_id=self.dataset_id,
                audit_status=status,
            )
            records.append(record)

        if progress_callback:
            progress_callback(total_files, total_files)

        # Invariant: records strictly sorted by relative_path
        records.sort(key=lambda r: r.relative_path)
        return records

    @staticmethod
    def save_csv(records: list[ManifestRecord], output_path: Path | str) -> str:
        """Saves manifest records to CSV and returns file SHA-256 hash."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
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
            "audit_status",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(r.to_dict())

        return compute_sha256(path)

    @staticmethod
    def save_json(records: list[ManifestRecord], output_path: Path | str, indent: int = 2) -> str:
        """Saves manifest records to JSON and returns file SHA-256 hash."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in records], f, indent=indent)
        return compute_sha256(path)

    @staticmethod
    def load_csv(csv_path: Path | str) -> list[ManifestRecord]:
        """Loads manifest records from CSV."""
        path = Path(csv_path)
        records: list[ManifestRecord] = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(
                    ManifestRecord(
                        relative_path=row["relative_path"],
                        class_name=row["class_name"],
                        class_index=int(row["class_index"]),
                        width=int(row["width"]),
                        height=int(row["height"]),
                        channels=int(row["channels"]),
                        file_size_bytes=int(row["file_size_bytes"]),
                        sha256=row["sha256"],
                        dhash=row["dhash"],
                        dataset_id=row["dataset_id"],
                        audit_status=row["audit_status"],
                    )
                )
        return records
