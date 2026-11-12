"""Instance Processing and Post-Processing Utilities for Nuclei Segmentation.

Provides contour extraction, bounding box localization, connected component
filtering, re-indexing, and unit-test synthetic fixtures.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def extract_instance_contours(instance_map: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Extracts closed boundary contours for each non-zero instance in the instance map.

    Args:
        instance_map: 2D integer array (H, W).

    Returns:
        Dict mapping integer instance ID to contour array of shape (N, 2).
    """
    contours_dict: Dict[int, np.ndarray] = {}
    unique_ids = [int(uid) for uid in np.unique(instance_map) if uid > 0]

    for uid in unique_ids:
        binary_mask = (instance_map == uid).astype(np.uint8)
        # Find external contours
        cnts, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if cnts:
            # Largest contour if multiple disconnected fragments exist
            largest_cnt = max(cnts, key=cv2.contourArea)
            contours_dict[uid] = largest_cnt.squeeze(axis=1)  # (N, 2)
        else:
            contours_dict[uid] = np.empty((0, 2), dtype=np.int32)

    return contours_dict


def extract_instance_bounding_boxes(
    instance_map: np.ndarray,
) -> Dict[int, Tuple[int, int, int, int]]:
    """
    Computes (x_min, y_min, width, height) bounding boxes for each instance ID.
    """
    bboxes: Dict[int, Tuple[int, int, int, int]] = {}
    unique_ids = [int(uid) for uid in np.unique(instance_map) if uid > 0]

    for uid in unique_ids:
        rows, cols = np.where(instance_map == uid)
        if len(rows) > 0:
            y_min, y_max = int(np.min(rows)), int(np.max(rows))
            x_min, x_max = int(np.min(cols)), int(np.max(cols))
            bboxes[uid] = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
    return bboxes


def filter_and_reindex_instances(
    instance_map: np.ndarray,
    min_area: int = 10,
    max_area: int = 2500,
) -> Tuple[np.ndarray, int]:
    """
    Filters instances by pixel area bounds and re-indexes remaining instances sequentially 1..K.

    Args:
        instance_map: 2D integer array (H, W).
        min_area: Minimum area in pixels^2.
        max_area: Maximum area in pixels^2.

    Returns:
        filtered_map: 2D integer array with instances re-indexed 1..K.
        retained_count: Total number of retained instances.
    """
    filtered = np.zeros_like(instance_map, dtype=np.int32)
    unique_ids = [int(uid) for uid in np.unique(instance_map) if uid > 0]
    valid_count = 0

    for uid in unique_ids:
        pixels = (instance_map == uid)
        area = int(np.sum(pixels))
        if min_area <= area <= max_area:
            valid_count += 1
            filtered[pixels] = valid_count

    return filtered, valid_count


def generate_synthetic_instance_fixture(
    shape: Tuple[int, int] = (224, 224),
    num_nuclei: int = 15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a deterministic synthetic H&E image and paired nuclear instance map.
    NOTE: This is strictly a software unit-testing fixture to verify feature extraction
    and regional analysis math. It is NEVER used as research data or clinical detections.
    """
    rng = np.random.RandomState(seed)
    h, w = shape

    # Synthetic eosin/stroma background (pinkish)
    img = np.full((h, w, 3), (225, 195, 210), dtype=np.uint8)
    noise = rng.randint(-10, 10, size=(h, w, 3))
    img = np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)

    instance_map = np.zeros((h, w), dtype=np.int32)

    # Render distinct elliptical nuclei across quadrants
    for i in range(1, num_nuclei + 1):
        cx = rng.randint(20, w - 20)
        cy = rng.randint(20, h - 20)
        ax1 = rng.randint(5, 12)
        ax2 = rng.randint(4, 9)
        angle = rng.randint(0, 180)

        # Draw filled ellipse on instance map
        cv2.ellipse(
            instance_map,
            (cx, cy),
            (ax1, ax2),
            angle,
            0,
            360,
            color=int(i),
            thickness=-1,
        )

        # Draw dark purple/blue Hematoxylin stain on synthetic image
        cv2.ellipse(
            img,
            (cx, cy),
            (ax1, ax2),
            angle,
            0,
            360,
            color=(70, 40, 110),
            thickness=-1,
        )

    return img, instance_map

