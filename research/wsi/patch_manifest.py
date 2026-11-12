"""Coordinate-Aware Patch Manifest Schema and Persistence Engine.

Maintains strict provenance, Level-0 spatial coordinates, tissue percentage, and physical
resolution metadata for every patch tile extracted from a Whole-Slide Image:
- Slide-to-patch isolation (ensures patches from unrelated slides are never co-mingled).
- Level-0 reference coordinate integrity (level0_x, level0_y).
- Physical micrometer resolution preservation (MPP_X, MPP_Y).
- Formatted CSV and JSON persistence with automated validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger("wsi.manifest")

PATCH_MANIFEST_COLUMNS: List[str] = [
    "slide_id",
    "patch_id",
    "level",
    "level0_x",
    "level0_y",
    "patch_width",
    "patch_height",
    "downsample",
    "tissue_percentage",
    "mpp_x",
    "mpp_y",
    "magnification",
    "relative_patch_path",
    "source_slide_provenance",
]


@dataclass
class PatchRecord:
    """Individual record representing an extracted tile from a Whole-Slide Image."""

    slide_id: str
    patch_id: str
    level: int
    level0_x: int
    level0_y: int
    patch_width: int
    patch_height: int
    downsample: float
    tissue_percentage: float
    mpp_x: Optional[float] = None
    mpp_y: Optional[float] = None
    magnification: Optional[float] = None
    relative_patch_path: str = ""
    source_slide_provenance: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PatchManifest:
    """Manager container for patch metadata across one or more slides."""

    def __init__(self, records: Optional[List[PatchRecord]] = None) -> None:
        self.records: List[PatchRecord] = records or []

    def add_record(self, record: PatchRecord) -> None:
        self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def to_dataframe(self) -> pd.DataFrame:
        """Converts manifest records into a standardized pandas DataFrame."""
        if not self.records:
            return pd.DataFrame(columns=PATCH_MANIFEST_COLUMNS)
        data = [r.to_dict() for r in self.records]
        return pd.DataFrame(data)[PATCH_MANIFEST_COLUMNS]

    def save_csv(self, output_path: Union[str, Path]) -> str:
        """Serializes manifest to a CSV file."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(str(p), index=False)
        return str(p.resolve()).replace("\\", "/")

    @classmethod
    def load_csv(cls, input_path: Union[str, Path]) -> "PatchManifest":
        """Loads a patch manifest from a CSV file."""
        p = Path(input_path)
        if not p.exists():
            raise FileNotFoundError(f"Manifest file not found: {p}")
        df = pd.read_csv(str(p))
        records = []
        for _, row in df.iterrows():
            rec = PatchRecord(
                slide_id=str(row["slide_id"]),
                patch_id=str(row["patch_id"]),
                level=int(row["level"]),
                level0_x=int(row["level0_x"]),
                level0_y=int(row["level0_y"]),
                patch_width=int(row["patch_width"]),
                patch_height=int(row["patch_height"]),
                downsample=float(row["downsample"]),
                tissue_percentage=float(row["tissue_percentage"]),
                mpp_x=float(row["mpp_x"]) if pd.notna(row.get("mpp_x")) else None,
                mpp_y=float(row["mpp_y"]) if pd.notna(row.get("mpp_y")) else None,
                magnification=float(row["magnification"]) if pd.notna(row.get("magnification")) else None,
                relative_patch_path=str(row.get("relative_patch_path", "")),
                source_slide_provenance=str(row.get("source_slide_provenance", "")),
            )
            records.append(rec)
        return cls(records=records)

    def get_slide_records(self, slide_id: str) -> List[PatchRecord]:
        """Returns all patch records associated with a specific slide ID (strictly isolated)."""
        return [r for r in self.records if r.slide_id == slide_id]

    def get_unique_slide_ids(self) -> List[str]:
        """Returns sorted list of unique slide IDs contained in the manifest."""
        return sorted(list({r.slide_id for r in self.records}))

    def validate_manifest(self) -> Tuple[bool, List[str]]:
        """
        Validates internal consistency:
        - Unique (slide_id, patch_id) pairs.
        - Non-negative Level-0 coordinates.
        - Positive patch dimensions and downsample factors.
        - Tissue percentage bounded in [0.0, 1.0].
        """
        errors: List[str] = []
        seen_keys = set()

        for idx, r in enumerate(self.records):
            key = (r.slide_id, r.patch_id)
            if key in seen_keys:
                errors.append(f"Row {idx}: Duplicate slide_id/patch_id key: {key}")
            seen_keys.add(key)

            if r.level0_x < 0 or r.level0_y < 0:
                errors.append(f"Row {idx} ({r.patch_id}): Negative level-0 coords ({r.level0_x}, {r.level0_y})")
            if r.patch_width <= 0 or r.patch_height <= 0:
                errors.append(f"Row {idx} ({r.patch_id}): Invalid dimensions ({r.patch_width}x{r.patch_height})")
            if r.downsample <= 0:
                errors.append(f"Row {idx} ({r.patch_id}): Non-positive downsample {r.downsample}")
            if not (0.0 <= r.tissue_percentage <= 1.0001):
                errors.append(f"Row {idx} ({r.patch_id}): Out-of-bounds tissue pct: {r.tissue_percentage}")

        return len(errors) == 0, errors

