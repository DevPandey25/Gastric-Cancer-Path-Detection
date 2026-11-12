"""Structured Slide Metadata Engine for Whole-Slide Histopathology.

Standardizes pyramidal whole-slide metadata extraction across diverse slide vendors
(Aperio .svs, Hamamatsu .ndpi, 3DHISTECH .mrxs, Philips, Generic TIFF) and synthetic fixtures:
- Pyramid levels and downsample factor enumeration.
- Level-0 pixel dimensions and reference coordinate system bounds.
- Physical resolution / Microns Per Pixel (MPP_X, MPP_Y) validation.
- Apparent optical magnification (e.g., 20x, 40x).
- Raw metadata property preservation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class SlideMetadata:
    """Standardized metadata container for a Whole-Slide Image."""

    slide_id: str
    file_path: str
    file_size_bytes: int
    format_or_vendor: str
    level_count: int
    level_dimensions: List[Tuple[int, int]]  # (width, height) at each level
    level_downsamples: List[float]  # Downsample factor relative to level 0
    level0_dimensions: Tuple[int, int]  # (width, height) at level 0
    mpp_x: Optional[float] = None  # Microns per pixel (X)
    mpp_y: Optional[float] = None  # Microns per pixel (Y)
    apparent_magnification: Optional[float] = None  # e.g., 20.0, 40.0
    bounds_x: Optional[int] = None  # Level-0 tissue bounding box X
    bounds_y: Optional[int] = None  # Level-0 tissue bounding box Y
    bounds_width: Optional[int] = None  # Level-0 tissue bounding box W
    bounds_height: Optional[int] = None  # Level-0 tissue bounding box H
    raw_properties: Dict[str, str] = field(default_factory=dict)
    is_genuine_wsi: bool = False  # False for synthetic/mock test fixtures

    def to_dict(self) -> Dict[str, Any]:
        """Converts dataclass to a JSON-serializable dictionary."""
        d = asdict(self)
        # Format tuples as lists for JSON compatibility
        d["level_dimensions"] = [list(dims) for dims in self.level_dimensions]
        d["level0_dimensions"] = list(self.level0_dimensions)
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serializes metadata to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def get_downsample_for_level(self, level: int) -> float:
        """Returns the downsample factor for a specific pyramid level."""
        if level < 0 or level >= self.level_count:
            raise IndexError(f"Level {level} out of range for slide with {self.level_count} levels.")
        return self.level_downsamples[level]

    def get_dimensions_for_level(self, level: int) -> Tuple[int, int]:
        """Returns the (width, height) dimensions for a specific pyramid level."""
        if level < 0 or level >= self.level_count:
            raise IndexError(f"Level {level} out of range for slide with {self.level_count} levels.")
        return self.level_dimensions[level]

    def get_best_level_for_downsample(self, target_downsample: float) -> int:
        """
        Finds the pyramid level closest to but not exceeding the target downsample factor.
        Matches the behavior of OpenSlide's get_best_level_for_downsample().
        """
        if target_downsample <= self.level_downsamples[0]:
            return 0
        for i in range(1, self.level_count):
            if target_downsample < self.level_downsamples[i]:
                return i - 1
        return self.level_count - 1


def parse_openslide_properties(
    slide_id: str,
    file_path: Union[str, Path],
    file_size_bytes: int,
    properties: Dict[str, str],
    level_dimensions: List[Tuple[int, int]],
    level_downsamples: List[float],
    is_genuine: bool = True,
) -> SlideMetadata:
    """
    Parses OpenSlide raw properties dictionary into standardized SlideMetadata.
    Extracts vendor, MPP, magnification, and bounding boxes.
    """
    p = properties

    # Vendor / Format detection
    vendor = p.get("openslide.vendor", p.get("openslide.format", "generic_wsi"))

    # MPP resolution
    mpp_x: Optional[float] = None
    mpp_y: Optional[float] = None
    if "openslide.mpp-x" in p:
        try:
            mpp_x = float(p["openslide.mpp-x"])
        except ValueError:
            pass
    if "openslide.mpp-y" in p:
        try:
            mpp_y = float(p["openslide.mpp-y"])
        except ValueError:
            pass

    # Magnification
    mag: Optional[float] = None
    if "openslide.objective-power" in p:
        try:
            mag = float(p["openslide.objective-power"])
        except ValueError:
            pass
    elif "aperio.AppMag" in p:
        try:
            mag = float(p["aperio.AppMag"])
        except ValueError:
            pass

    # Bounds
    bounds_x = int(p["openslide.bounds-x"]) if "openslide.bounds-x" in p else None
    bounds_y = int(p["openslide.bounds-y"]) if "openslide.bounds-y" in p else None
    bounds_w = int(p["openslide.bounds-width"]) if "openslide.bounds-width" in p else None
    bounds_h = int(p["openslide.bounds-height"]) if "openslide.bounds-height" in p else None

    level0_dims = level_dimensions[0] if level_dimensions else (0, 0)

    return SlideMetadata(
        slide_id=slide_id,
        file_path=str(file_path).replace("\\", "/"),
        file_size_bytes=file_size_bytes,
        format_or_vendor=vendor,
        level_count=len(level_dimensions),
        level_dimensions=level_dimensions,
        level_downsamples=level_downsamples,
        level0_dimensions=level0_dims,
        mpp_x=mpp_x,
        mpp_y=mpp_y,
        apparent_magnification=mag,
        bounds_x=bounds_x,
        bounds_y=bounds_y,
        bounds_width=bounds_w,
        bounds_height=bounds_h,
        raw_properties=properties,
        is_genuine_wsi=is_genuine,
    )

