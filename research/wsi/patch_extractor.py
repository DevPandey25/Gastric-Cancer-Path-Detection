"""Coordinate-Aware Patch Extraction Engine for Whole-Slide Histopathology.

Performs deterministic, tissue-filtered patch extraction across pyramidal levels:
- Strict Level-0 reference coordinate calculation and preservation.
- Configurable patch dimensions (e.g. 224x224, 256x256), target level/downsample, and stride.
- O(1) integral tissue coverage filtering (rejecting acellular glass whitespace).
- Precise boundary condition handling (discarding incomplete edges).
- Image export and structured PatchManifest creation with complete provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from research.wsi.patch_manifest import PatchManifest, PatchRecord
from research.wsi.slide_reader import BaseSlideReader
from research.wsi.tissue_detection import TissueDetector

logger = logging.getLogger("wsi.extractor")


@dataclass
class PatchExtractionConfig:
    """Hyperparameters governing patch tiling and coordinate generation."""
    patch_width: int = 224
    patch_height: int = 224
    target_level: int = 0
    stride_x: Optional[int] = None  # None defaults to patch_width (non-overlapping)
    stride_y: Optional[int] = None  # None defaults to patch_height (non-overlapping)
    min_tissue_percentage: float = 0.50  # Retain patches with >= 50% tissue
    discard_partial_boundaries: bool = True  # Reject partial edge patches extending outside canvas
    save_patch_images: bool = True  # Save extracted PNG images to disk
    image_format: str = "PNG"


class CoordinateAwarePatchExtractor:
    """
    Extracts coordinate-tagged tiles from a pyramidal Whole-Slide Image
    while preserving Level-0 reference coordinates and filtering background whitespace.
    """

    def __init__(
        self,
        reader: BaseSlideReader,
        tissue_detector: Optional[TissueDetector] = None,
        config: Optional[PatchExtractionConfig] = None,
    ) -> None:
        self.reader = reader
        self.metadata = reader.metadata
        self.config = config or PatchExtractionConfig()
        self.tissue_detector = tissue_detector

        # Resolve stride
        self.stride_x = self.config.stride_x or self.config.patch_width
        self.stride_y = self.config.stride_y or self.config.patch_height

        # Validate target level
        if self.config.target_level < 0 or self.config.target_level >= self.metadata.level_count:
            raise IndexError(
                f"Target level {self.config.target_level} invalid for slide with "
                f"{self.metadata.level_count} levels."
            )

        self.downsample = self.metadata.get_downsample_for_level(self.config.target_level)
        self.level_dims = self.metadata.get_dimensions_for_level(self.config.target_level)
        self.w0, self.h0 = self.metadata.level0_dimensions

    def generate_grid_coordinates(self) -> Generator[Tuple[int, int, int, int], None, None]:
        """
        Generates deterministic patch grid coordinates across the slide canvas.

        Yields:
            (col_idx, row_idx, level0_x, level0_y)
        """
        lvl_w, lvl_h = self.level_dims
        pw = self.config.patch_width
        ph = self.config.patch_height
        sx = self.stride_x
        sy = self.stride_y

        num_cols = (lvl_w - pw) // sx + 1 if lvl_w >= pw else 0
        num_rows = (lvl_h - ph) // sy + 1 if lvl_h >= ph else 0

        for row_idx in range(num_rows):
            lvl_y = row_idx * sy
            level0_y = int(round(lvl_y * self.downsample))

            # Boundary check in Level-0 space
            level0_span_h = int(round(ph * self.downsample))
            if self.config.discard_partial_boundaries and (level0_y + level0_span_h > self.h0):
                continue

            for col_idx in range(num_cols):
                lvl_x = col_idx * sx
                level0_x = int(round(lvl_x * self.downsample))

                level0_span_w = int(round(pw * self.downsample))
                if self.config.discard_partial_boundaries and (level0_x + level0_span_w > self.w0):
                    continue

                yield (col_idx, row_idx, level0_x, level0_y)

    def extract_all_patches(
        self,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> PatchManifest:
        """
        Executes complete extraction pipeline across the slide:
        1. Enumerates grid locations.
        2. Filters patches by tissue coverage.
        3. Reads pixel data adhering strictly to Level-0 coordinates.
        4. Saves patch images to disk (if configured).
        5. Returns structured PatchManifest.
        """
        slide_id = self.metadata.slide_id
        manifest = PatchManifest()

        out_path: Optional[Path] = None
        if output_dir is not None and self.config.save_patch_images:
            out_path = Path(output_dir) / slide_id
            out_path.mkdir(parents=True, exist_ok=True)

        pw = self.config.patch_width
        ph = self.config.patch_height
        lvl = self.config.target_level
        d = self.downsample

        mpp_x = self.metadata.mpp_x
        mpp_y = self.metadata.mpp_y
        mag = self.metadata.apparent_magnification
        prov = self.metadata.format_or_vendor

        patch_count = 0
        for col_idx, row_idx, level0_x, level0_y in self.generate_grid_coordinates():
            # Tissue coverage assessment
            if self.tissue_detector is not None:
                is_valid, tissue_pct = self.tissue_detector.is_patch_valid_tissue(
                    level0_x=level0_x,
                    level0_y=level0_y,
                    patch_width_level=pw,
                    patch_height_level=ph,
                    downsample=d,
                    min_tissue_percentage=self.config.min_tissue_percentage,
                )
                if not is_valid:
                    continue  # Filter out acellular background whitespace
            else:
                tissue_pct = 1.0

            patch_id = f"{slide_id}_lvl{lvl}_x{level0_x}_y{level0_y}"
            rel_image_path = ""

            # Extract patch image using Level-0 coordinates
            if out_path is not None and self.config.save_patch_images:
                patch_img = self.reader.read_region(
                    location=(level0_x, level0_y),
                    level=lvl,
                    size=(pw, ph),
                )
                # Convert RGBA to RGB for standard histopathology storage
                patch_rgb = patch_img.convert("RGB")
                img_filename = f"{patch_id}.png"
                img_full_path = out_path / img_filename
                patch_rgb.save(str(img_full_path), format=self.config.image_format)
                rel_image_path = str(img_full_path.relative_to(out_path.parent)).replace("\\", "/")

            record = PatchRecord(
                slide_id=slide_id,
                patch_id=patch_id,
                level=lvl,
                level0_x=level0_x,
                level0_y=level0_y,
                patch_width=pw,
                patch_height=ph,
                downsample=d,
                tissue_percentage=tissue_pct,
                mpp_x=mpp_x,
                mpp_y=mpp_y,
                magnification=mag,
                relative_patch_path=rel_image_path,
                source_slide_provenance=prov,
            )
            manifest.add_record(record)
            patch_count += 1

        logger.info(f"Extracted {patch_count} valid tissue patches for slide '{slide_id}' at level {lvl}.")
        return manifest

