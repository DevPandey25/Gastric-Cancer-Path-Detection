"""Controlled Histopathology Image Perturbations (Phase 6).

Implements four deterministic, single-severity perturbation families:
1. Brightness & Contrast: factor 1.1 brightness, factor 1.1 contrast via PIL.ImageEnhance.
2. Gaussian Blur: kernel 5x5, sigma 1.0 via cv2.GaussianBlur.
3. JPEG Compression: quality 50 via in-memory Pillow encode/decode.
4. H&E-like Stain Variation: optical-density stain decomposition using Ruifrok-Johnston / scikit-image
   HED matrix, scaling hematoxylin and eosin channels by 1.1 while preserving DAB channel,
   followed by optical-density reconstruction and safe RGB clipping.

All transformations preserve the canonical image dimensions (224x224) and RGB mode.
"""

from __future__ import annotations

import io
from typing import Tuple
import cv2
import numpy as np
from PIL import Image, ImageEnhance

# Standard Ruifrok & Johnston (2001) / scikit-image HED stain matrix
# Rows correspond to Hematoxylin, Eosin, and DAB (background/counterstain)
_RAW_RGB_FROM_HED = np.array([
    [0.65, 0.70, 0.29],  # Hematoxylin
    [0.07, 0.99, 0.11],  # Eosin
    [0.27, 0.57, 0.78],  # DAB
], dtype=np.float64)

# Row-normalize to unit Euclidean length
RGB_FROM_HED = _RAW_RGB_FROM_HED / np.linalg.norm(_RAW_RGB_FROM_HED, axis=1, keepdims=True)
HED_FROM_RGB = np.linalg.inv(RGB_FROM_HED)


def apply_brightness_contrast(
    image: Image.Image,
    brightness_factor: float = 1.1,
    contrast_factor: float = 1.1,
) -> Image.Image:
    """Applies fixed brightness and contrast adjustment.

    Args:
        image: PIL Image in RGB format.
        brightness_factor: Brightness multiplier (1.0 = unchanged, 1.1 = +10%).
        contrast_factor: Contrast multiplier (1.0 = unchanged, 1.1 = +10%).

    Returns:
        Transformed PIL Image with identical dimensions and RGB mode.
    """
    img = image.convert("RGB")
    b_enh = ImageEnhance.Brightness(img).enhance(brightness_factor)
    bc_img = ImageEnhance.Contrast(b_enh).enhance(contrast_factor)
    return bc_img


def apply_gaussian_blur(
    image: Image.Image,
    ksize: Tuple[int, int] = (5, 5),
    sigma: float = 1.0,
) -> Image.Image:
    """Applies mild Gaussian blur using OpenCV.

    Args:
        image: PIL Image in RGB format.
        ksize: Blur kernel dimensions (width, height), must be positive and odd.
        sigma: Gaussian kernel standard deviation in X and Y directions.

    Returns:
        Blurred PIL Image with identical dimensions and RGB mode.
    """
    img = image.convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    blurred_arr = cv2.GaussianBlur(arr, ksize=ksize, sigmaX=sigma, sigmaY=sigma)
    return Image.fromarray(blurred_arr, mode="RGB")


def apply_jpeg_compression(
    image: Image.Image,
    quality: int = 50,
) -> Image.Image:
    """Applies lossy JPEG compression and decompression via in-memory buffer.

    Args:
        image: PIL Image in RGB format.
        quality: JPEG compression quality level (1-100, 50 represents standard lossy compression).

    Returns:
        Compressed/decompressed PIL Image with identical dimensions and RGB mode.
    """
    img = image.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    jpeg_img = Image.open(buf).convert("RGB")
    return jpeg_img


def apply_hed_stain_variation(
    image: Image.Image,
    h_factor: float = 1.1,
    e_factor: float = 1.1,
) -> Image.Image:
    """Applies H&E-like stain variation in optical density space.

    Methodology:
    1. Converts RGB to float in [0, 1].
    2. Computes optical density (OD) via Beer-Lambert law: OD = -log10(I / I0) with I0 = 1.0.
    3. Decomposes OD into Hematoxylin, Eosin, and DAB stain channels using HED_FROM_RGB.
    4. Scales Hematoxylin (H) and Eosin (E) channels by fixed factors (1.1).
    5. Preserves DAB channel without modification.
    6. Recombines modified stain channels into OD using RGB_FROM_HED.
    7. Exponentiates back to linear RGB intensity: I = 10^(-OD).
    8. Clips to [0.0, 1.0] and converts safely to uint8 [0, 255] RGB.

    Args:
        image: PIL Image in RGB format.
        h_factor: Multiplier for Hematoxylin channel (1.1 = +10% stain density).
        e_factor: Multiplier for Eosin channel (1.1 = +10% stain density).

    Returns:
        Stain-varied PIL Image with identical dimensions and RGB mode.
    """
    img = image.convert("RGB")
    arr_f = np.asarray(img, dtype=np.float64) / 255.0

    # Beer-Lambert optical density with epsilon to prevent log(0)
    od = -np.log10(np.maximum(arr_f, 1e-6))

    # Project into HED stain space: shape (H, W, 3)
    stains = np.dot(od.reshape(-1, 3), HED_FROM_RGB).reshape(arr_f.shape)

    # Apply documented perturbation factors to H and E channels
    stains[..., 0] *= h_factor
    stains[..., 1] *= e_factor
    # stains[..., 2] (DAB) remains unchanged

    # Recombine to optical density
    od_mod = np.dot(stains.reshape(-1, 3), RGB_FROM_HED).reshape(arr_f.shape)

    # Invert Beer-Lambert law: I = 10^(-OD)
    rgb_recon = np.clip(10.0 ** (-od_mod), 0.0, 1.0)
    uint8_arr = np.round(rgb_recon * 255.0).astype(np.uint8)

    return Image.fromarray(uint8_arr, mode="RGB")


PERMUTATION_DISPATCH = {
    "brightness_contrast": apply_brightness_contrast,
    "gaussian_blur": apply_gaussian_blur,
    "jpeg_compression": apply_jpeg_compression,
    "stain_hed_variation": apply_hed_stain_variation,
}

