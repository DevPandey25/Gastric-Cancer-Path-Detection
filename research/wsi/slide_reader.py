"""Pyramidal Whole-Slide Image Reader and OpenSlide Adapter Engine.

Implements unified slide reader interface adhering strictly to the OpenSlide coordinate contract:
- Context manager lifecycle (safe resource allocation and closing).
- Level-0 reference coordinate system invariance:
  read_region(location=(x_0, y_0), level=l, size=(w, h)) takes coordinates strictly in
  level-0 pixels, regardless of the target pyramid level being read.
- Lazy, graceful OpenSlide import with detailed environment diagnostic errors.
- Synthetic multi-level pyramidal reader for software verification and testing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from research.wsi.slide_metadata import SlideMetadata, parse_openslide_properties

logger = logging.getLogger("wsi.reader")


class OpenSlideUnavailableError(RuntimeError):
    """Raised when OpenSlide Python or its native C binaries/DLLs are not installed or cannot be loaded."""
    pass


class SlideReadError(RuntimeError):
    """Raised when a region, thumbnail, or metadata cannot be read from a slide file."""
    pass


class BaseSlideReader(ABC):
    """Abstract base class establishing the contract for pyramidal slide readers."""

    def __init__(self, file_path: Union[str, Path], slide_id: Optional[str] = None) -> None:
        self.file_path = Path(file_path)
        self.slide_id = slide_id or self.file_path.stem
        self._is_closed = False

    @property
    @abstractmethod
    def metadata(self) -> SlideMetadata:
        """Returns the standardized SlideMetadata container."""
        pass

    @abstractmethod
    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
    ) -> Image.Image:
        """
        Reads a rectangular region from the slide pyramid.

        CRITICAL COORDINATE RULE:
        `location` (x, y) is specified in the LEVEL-0 REFERENCE COORDINATE SYSTEM,
        even when reading from a downsampled level (level > 0).
        `size` (width, height) is the pixel dimensions of the returned patch at `level`.

        Args:
            location: (level0_x, level0_y) top-left corner in Level-0 pixels.
            level: Target pyramid level index (0 to level_count - 1).
            size: (patch_width, patch_height) in level pixels.

        Returns:
            PIL.Image.Image in RGBA mode.
        """
        pass

    @abstractmethod
    def get_thumbnail(self, max_size: Tuple[int, int] = (1024, 1024)) -> Image.Image:
        """
        Returns an RGB thumbnail of the complete slide whose dimensions do not exceed max_size.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Safely closes the underlying slide handle and releases resources."""
        pass

    def __enter__(self) -> "BaseSlideReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class OpenSlideAdapter(BaseSlideReader):
    """
    Adapter wrapping native OpenSlide for reading clinical WSI files (.svs, .ndpi, .mrxs, .tif).
    Uses lazy import to prevent application crashes when native DLLs are missing.
    """

    def __init__(self, file_path: Union[str, Path], slide_id: Optional[str] = None) -> None:
        super().__init__(file_path, slide_id)
        self._osr = None
        self._metadata = None
        self._initialize_openslide()

    def _initialize_openslide(self) -> None:
        if not self.file_path.exists():
            raise FileNotFoundError(f"Slide file not found: {self.file_path}")

        try:
            import openslide
        except ImportError as e:
            raise OpenSlideUnavailableError(
                "OpenSlide Python (`openslide-python`) is not installed or native OpenSlide C DLLs "
                "are missing from the system PATH. On Windows 11 with Python 3.13, native OpenSlide "
                "binaries must be compiled and added via `os.add_dll_directory()`. "
                f"Original error: {e}"
            ) from e

        try:
            self._osr = openslide.OpenSlide(str(self.file_path))
        except Exception as e:
            raise SlideReadError(f"Failed to open slide {self.file_path} via OpenSlide: {e}") from e

        # Extract level dimensions and downsamples
        level_dims = [tuple(d) for d in self._osr.level_dimensions]
        level_downs = [float(d) for d in self._osr.level_downsamples]
        file_size = self.file_path.stat().st_size
        raw_props = dict(self._osr.properties)

        self._metadata = parse_openslide_properties(
            slide_id=self.slide_id,
            file_path=self.file_path,
            file_size_bytes=file_size,
            properties=raw_props,
            level_dimensions=level_dims,
            level_downsamples=level_downs,
            is_genuine=True,
        )

    @property
    def metadata(self) -> SlideMetadata:
        if self._is_closed or self._metadata is None:
            raise RuntimeError("Slide reader is closed.")
        return self._metadata

    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
    ) -> Image.Image:
        if self._is_closed or self._osr is None:
            raise RuntimeError("Slide reader is closed.")
        if level < 0 or level >= self.metadata.level_count:
            raise IndexError(f"Level {level} out of bounds (0 to {self.metadata.level_count - 1}).")

        # OpenSlide read_region accepts (level0_x, level0_y), level, (w, h)
        return self._osr.read_region(location, level, size)

    def get_thumbnail(self, max_size: Tuple[int, int] = (1024, 1024)) -> Image.Image:
        if self._is_closed or self._osr is None:
            raise RuntimeError("Slide reader is closed.")
        thumb = self._osr.get_thumbnail(max_size)
        return thumb.convert("RGB")

    def close(self) -> None:
        if not self._is_closed:
            if self._osr is not None:
                try:
                    self._osr.close()
                except Exception:
                    pass
                self._osr = None
            self._is_closed = True


class SyntheticPyramidSlideReader(BaseSlideReader):
    """
    High-fidelity software pyramid slide reader for testing, verification, and offline pipeline execution.
    Creates a multi-resolution image pyramid in memory and strictly implements the Level-0 coordinate contract.
    """

    def __init__(
        self,
        base_image: Image.Image,
        slide_id: str = "synthetic_wsi",
        downsamples: Optional[List[float]] = None,
        mpp_x: Optional[float] = 0.5,
        mpp_y: Optional[float] = 0.5,
        apparent_magnification: Optional[float] = 20.0,
        vendor: str = "Synthetic Test Fixture",
    ) -> None:
        super().__init__(file_path=Path(f"synthetic/{slide_id}.tiff"), slide_id=slide_id)
        if downsamples is None:
            downsamples = [1.0, 2.0, 4.0, 8.0]

        self._base_rgba = base_image.convert("RGBA")
        w0, h0 = self._base_rgba.size

        self._levels: List[Image.Image] = [self._base_rgba]
        self._level_dimensions: List[Tuple[int, int]] = [(w0, h0)]
        self._level_downsamples: List[float] = [1.0]

        for d in downsamples[1:]:
            lvl_w = max(int(round(w0 / d)), 1)
            lvl_h = max(int(round(h0 / d)), 1)
            actual_d = w0 / lvl_w
            lvl_img = self._base_rgba.resize((lvl_w, lvl_h), Image.Resampling.BILINEAR)
            self._levels.append(lvl_img)
            self._level_dimensions.append((lvl_w, lvl_h))
            self._level_downsamples.append(actual_d)

        self._metadata = SlideMetadata(
            slide_id=self.slide_id,
            file_path=str(self.file_path).replace("\\", "/"),
            file_size_bytes=w0 * h0 * 4,
            format_or_vendor=vendor,
            level_count=len(self._levels),
            level_dimensions=self._level_dimensions,
            level_downsamples=self._level_downsamples,
            level0_dimensions=(w0, h0),
            mpp_x=mpp_x,
            mpp_y=mpp_y,
            apparent_magnification=apparent_magnification,
            bounds_x=0,
            bounds_y=0,
            bounds_width=w0,
            bounds_height=h0,
            raw_properties={"test_fixture": "synthetic_multilevel_pyramid"},
            is_genuine_wsi=False,
        )

    @property
    def metadata(self) -> SlideMetadata:
        if self._is_closed:
            raise RuntimeError("Slide reader is closed.")
        return self._metadata

    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
    ) -> Image.Image:
        """
        Extracts region adhering strictly to Level-0 reference coordinate mapping:
        location = (x_0, y_0) in Level-0 pixels.
        Converted to level coordinates via:
        x_level = round(x_0 / downsample)
        y_level = round(y_0 / downsample)
        """
        if self._is_closed:
            raise RuntimeError("Slide reader is closed.")
        if level < 0 or level >= self.metadata.level_count:
            raise IndexError(f"Level {level} out of range (0 to {self.metadata.level_count - 1}).")

        downsample = self._level_downsamples[level]
        lvl_img = self._levels[level]
        lvl_w, lvl_h = lvl_img.size

        # Map Level-0 location to Level location
        x0, y0 = location
        x_lvl = int(round(x0 / downsample))
        y_lvl = int(round(y0 / downsample))

        patch_w, patch_h = size
        crop_box = (x_lvl, y_lvl, x_lvl + patch_w, y_lvl + patch_h)

        # Handle boundary out-of-bounds with whitespace padding (mimicking OpenSlide)
        region = Image.new("RGBA", (patch_w, patch_h), (255, 255, 255, 0))

        # Intersection of crop box with level image
        src_x1 = max(0, x_lvl)
        src_y1 = max(0, y_lvl)
        src_x2 = min(lvl_w, x_lvl + patch_w)
        src_y2 = min(lvl_h, y_lvl + patch_h)

        if src_x2 > src_x1 and src_y2 > src_y1:
            cropped = lvl_img.crop((src_x1, src_y1, src_x2, src_y2))
            dest_x = src_x1 - x_lvl
            dest_y = src_y1 - y_lvl
            region.paste(cropped, (dest_x, dest_y))

        return region

    def get_thumbnail(self, max_size: Tuple[int, int] = (1024, 1024)) -> Image.Image:
        if self._is_closed:
            raise RuntimeError("Slide reader is closed.")
        thumb = self._base_rgba.copy()
        thumb.thumbnail(max_size, Image.Resampling.LANCZOS)
        return thumb.convert("RGB")

    def close(self) -> None:
        self._is_closed = True
        self._levels.clear()


def open_slide(file_path: Union[str, Path], slide_id: Optional[str] = None) -> BaseSlideReader:
    """
    Factory function opening a slide with OpenSlideAdapter.
    If OpenSlide is unavailable or file is unsupported, raises appropriate exception.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Slide file does not exist: {p}")
    return OpenSlideAdapter(file_path=p, slide_id=slide_id)

