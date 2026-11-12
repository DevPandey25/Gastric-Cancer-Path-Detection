"""Lightweight Thumbnail-Based Tissue Detection Engine for Whole-Slide Histopathology.

Performs fast, low-memory tissue masking and background whitespace rejection:
- HSV color space thresholding (saturation & brightness separation).
- Morphological noise cleanup (opening, closing, hole filling).
- O(1) integral image acceleration for evaluating tissue percentages across patch grids.
- Coordinate-aware mapping between Level-0 coordinates and thumbnail pixel space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


@dataclass
class TissueDetectionConfig:
    """Configurable hyperparameters for thumbnail tissue detection."""
    sat_threshold: int = 15  # Minimum saturation in [0, 255] to be considered tissue
    val_threshold: int = 245  # Maximum brightness in [0, 255] to reject pure white glass
    kernel_size: int = 5  # Structuring element size for morphological operations
    min_tissue_percentage: float = 0.50  # Default threshold for keeping a patch (50%)


class TissueDetector:
    """
    Lightweight tissue detection and background filtering engine.
    Constructs a low-resolution binary tissue mask from a slide thumbnail
    and evaluates tissue coverage in arbitrary Level-0 bounding boxes in O(1) time.
    """

    def __init__(
        self,
        thumbnail: Image.Image,
        level0_dimensions: Tuple[int, int],
        config: Optional[TissueDetectionConfig] = None,
    ) -> None:
        self.config = config or TissueDetectionConfig()
        self.level0_width, self.level0_height = level0_dimensions

        # Convert thumbnail to RGB numpy array
        self.thumb_rgb = np.array(thumbnail.convert("RGB"))
        self.thumb_height, self.thumb_width = self.thumb_rgb.shape[:2]

        # Calculate exact downsample factor from Level-0 to thumbnail
        self.scale_x = self.level0_width / float(self.thumb_width)
        self.scale_y = self.level0_height / float(self.thumb_height)

        # Generate binary tissue mask (H_thumb, W_thumb)
        self.tissue_mask = self._compute_tissue_mask(self.thumb_rgb)

        # Compute 2D Integral Image (summed area table) for O(1) region queries
        # Integral image has shape (H+1, W+1) where integral[y, x] is sum of [0:y, 0:x]
        self._integral_mask = cv2.integral(self.tissue_mask.astype(np.uint8))

    def _compute_tissue_mask(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Generates binary tissue mask via HSV thresholding and morphological filtering.
        """
        # Convert RGB to HSV
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        # Tissue has measurable saturation and is not overexposed background whitespace
        raw_mask = (sat >= self.config.sat_threshold) & (val <= self.config.val_threshold)
        mask_uint8 = (raw_mask.astype(np.uint8)) * 255

        # Morphological operations
        k = max(self.config.kernel_size, 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        # 1. Closing: bridges small glandular lumens and intra-tissue fissures
        closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

        # 2. Opening: removes isolated dust particles and background scanning artifacts
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

        # Return boolean mask (True = tissue, False = background glass)
        return opened > 0

    def compute_tissue_percentage_level0(
        self,
        level0_x: int,
        level0_y: int,
        patch_width_level: int,
        patch_height_level: int,
        downsample: float = 1.0,
    ) -> float:
        """
        Calculates the fraction of tissue [0.0, 1.0] within a patch defined in Level-0 space.
        Uses O(1) integral image evaluation.

        Args:
            level0_x: Level-0 top-left X coordinate.
            level0_y: Level-0 top-left Y coordinate.
            patch_width_level: Width of the patch in level pixels.
            patch_height_level: Height of the patch in level pixels.
            downsample: Downsample factor of the level relative to Level-0.

        Returns:
            Fraction of tissue within the region in [0.0, 1.0].
        """
        # Calculate level-0 span
        level0_span_w = patch_width_level * downsample
        level0_span_h = patch_height_level * downsample

        # Map Level-0 bounds to thumbnail pixel coordinates
        tx1 = int(np.floor(level0_x / self.scale_x))
        ty1 = int(np.floor(level0_y / self.scale_y))
        tx2 = int(np.ceil((level0_x + level0_span_w) / self.scale_x))
        ty2 = int(np.ceil((level0_y + level0_span_h) / self.scale_y))

        # Clamp to thumbnail dimensions
        tx1 = max(0, min(tx1, self.thumb_width))
        ty1 = max(0, min(ty1, self.thumb_height))
        tx2 = max(0, min(tx2, self.thumb_width))
        ty2 = max(0, min(ty2, self.thumb_height))

        box_w = tx2 - tx1
        box_h = ty2 - ty1

        if box_w <= 0 or box_h <= 0:
            return 0.0

        # O(1) Integral Image Lookup:
        # Sum = I(y2, x2) - I(y1, x2) - I(y2, x1) + I(y1, x1)
        tissue_pixel_sum = (
            int(self._integral_mask[ty2, tx2])
            - int(self._integral_mask[ty1, tx2])
            - int(self._integral_mask[ty2, tx1])
            + int(self._integral_mask[ty1, tx1])
        )

        total_pixels = box_w * box_h
        return float(np.clip(tissue_pixel_sum / total_pixels, 0.0, 1.0))

    def is_patch_valid_tissue(
        self,
        level0_x: int,
        level0_y: int,
        patch_width_level: int,
        patch_height_level: int,
        downsample: float = 1.0,
        min_tissue_percentage: Optional[float] = None,
    ) -> Tuple[bool, float]:
        """
        Determines whether a patch satisfies the minimum tissue percentage threshold.

        Returns:
            is_valid: bool
            tissue_percentage: float in [0.0, 1.0]
        """
        thresh = min_tissue_percentage if min_tissue_percentage is not None else self.config.min_tissue_percentage
        pct = self.compute_tissue_percentage_level0(
            level0_x=level0_x,
            level0_y=level0_y,
            patch_width_level=patch_width_level,
            patch_height_level=patch_height_level,
            downsample=downsample,
        )
        return pct >= thresh, pct

    def get_tissue_mask_image(self) -> Image.Image:
        """Returns the binary tissue mask as a PIL Image (0=black, 255=white)."""
        return Image.fromarray((self.tissue_mask.astype(np.uint8)) * 255)

    def get_tissue_overlay_image(self, alpha: float = 0.4, overlay_color: Tuple[int, int, int] = (0, 255, 0)) -> Image.Image:
        """
        Renders a visualization overlay of detected tissue over the RGB thumbnail.
        """
        overlay = self.thumb_rgb.copy()
        mask = self.tissue_mask
        overlay[mask] = (
            (1.0 - alpha) * overlay[mask] + alpha * np.array(overlay_color, dtype=np.float32)
        ).astype(np.uint8)
        return Image.fromarray(overlay)

