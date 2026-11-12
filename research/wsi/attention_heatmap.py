"""Slide-Level Attention Heatmap Reconstruction and Spatial Overlay Engine.

Projects instance-level Multiple Instance Learning (MIL) attention weights back onto
the Whole-Slide Image Level-0 coordinate canvas and slide thumbnail:
- Coordinate-preserving tile-to-thumbnail spatial mapping.
- Per-slide attention score percentile normalization.
- Alpha-blended colormapped overlays over tissue regions.
- Preserves background glass transparency (unshaded).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
from PIL import Image

logger = logging.getLogger("wsi.heatmap")


class SlideAttentionHeatmapGenerator:
    """
    Reconstructs 2D spatial attention heatmaps from patch coordinates and attention weights.
    """

    def __init__(
        self,
        thumbnail: Image.Image,
        level0_dimensions: Tuple[int, int],
        tissue_mask: Optional[np.ndarray] = None,
    ) -> None:
        self.thumbnail_rgb = np.array(thumbnail.convert("RGB"))
        self.thumb_h, self.thumb_w = self.thumbnail_rgb.shape[:2]
        self.w0, self.h0 = level0_dimensions
        self.scale_x = self.w0 / float(self.thumb_w)
        self.scale_y = self.h0 / float(self.thumb_h)
        self.tissue_mask = tissue_mask

    def generate_attention_heatmap(
        self,
        patch_coords: List[Tuple[int, int]],  # (level0_x, level0_y) for each patch
        patch_sizes: List[Tuple[int, int]],   # (patch_w, patch_h) at level
        downsamples: List[float],              # downsample factor at level
        attention_weights: np.ndarray,         # (K,) or (K, 1) float array
        colormap_name: str = "jet",
        alpha: float = 0.55,
        normalize_percentile: bool = True,
    ) -> Tuple[Image.Image, np.ndarray]:
        """
        Generates alpha-blended attention overlay and raw 2D heatmap matrix.

        Args:
            patch_coords: List of K (level0_x, level0_y) tuples.
            patch_sizes: List of K (patch_width, patch_height) tuples.
            downsamples: List of K downsample factors.
            attention_weights: Array of shape (K,) or (K, 1).
            colormap_name: Matplotlib colormap (default: 'jet').
            alpha: Transparency of heatmap overlay.
            normalize_percentile: Whether to clip top 1% outliers.

        Returns:
            blended_image: PIL Image (RGB) with overlaid attention.
            raw_heatmap: np.ndarray (thumb_h, thumb_w) float32 in [0, 1].
        """
        weights = np.squeeze(np.asarray(attention_weights, dtype=np.float32))
        k = len(weights)
        if k == 0 or len(patch_coords) == 0:
            return Image.fromarray(self.thumbnail_rgb), np.zeros((self.thumb_h, self.thumb_w), dtype=np.float32)

        # Raw attention accumulator canvas and hit-count canvas
        raw_canvas = np.zeros((self.thumb_h, self.thumb_w), dtype=np.float32)
        count_canvas = np.zeros((self.thumb_h, self.thumb_w), dtype=np.float32)

        for i in range(k):
            x0, y0 = patch_coords[i]
            pw, ph = patch_sizes[i]
            d = downsamples[i]
            w_val = float(weights[i])

            # Convert Level-0 bounds to thumbnail pixels
            span_w = pw * d
            span_h = ph * d

            tx1 = max(0, min(int(np.floor(x0 / self.scale_x)), self.thumb_w))
            ty1 = max(0, min(int(np.floor(y0 / self.scale_y)), self.thumb_h))
            tx2 = max(0, min(int(np.ceil((x0 + span_w) / self.scale_x)), self.thumb_w))
            ty2 = max(0, min(int(np.ceil((y0 + span_h) / self.scale_y)), self.thumb_h))

            if tx2 > tx1 and ty2 > ty1:
                raw_canvas[ty1:ty2, tx1:tx2] += w_val
                count_canvas[ty1:ty2, tx1:tx2] += 1.0

        # Average overlaps
        valid_hits = count_canvas > 0
        raw_canvas[valid_hits] /= count_canvas[valid_hits]

        # Normalization
        heatmap_norm = np.zeros_like(raw_canvas)
        if np.any(valid_hits):
            vals = raw_canvas[valid_hits]
            if normalize_percentile:
                vmin = float(np.percentile(vals, 1))
                vmax = float(np.percentile(vals, 99))
            else:
                vmin = float(np.min(vals))
                vmax = float(np.max(vals))

            if vmax > vmin:
                heatmap_norm[valid_hits] = np.clip((raw_canvas[valid_hits] - vmin) / (vmax - vmin), 0.0, 1.0)
            else:
                heatmap_norm[valid_hits] = 1.0

        # Colormap mapping
        cmap = matplotlib.colormaps[colormap_name]
        color_rgba = cmap(heatmap_norm)  # (H, W, 4) in [0, 1]
        color_rgb_uint8 = (color_rgba[:, :, :3] * 255).astype(np.uint8)

        # Alpha blending strictly on patches with tissue
        blended = self.thumbnail_rgb.copy().astype(np.float32)

        # Mask regions where patches were placed
        mask_to_blend = valid_hits
        if self.tissue_mask is not None:
            mask_to_blend = mask_to_blend & self.tissue_mask

        blended[mask_to_blend] = (
            (1.0 - alpha) * self.thumbnail_rgb[mask_to_blend]
            + alpha * color_rgb_uint8[mask_to_blend]
        )

        blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8)
        return Image.fromarray(blended_uint8), heatmap_norm
