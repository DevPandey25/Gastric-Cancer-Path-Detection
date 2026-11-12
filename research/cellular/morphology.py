"""Cellular Morphology Feature Extraction Engine.

Calculates instance-level geometric and intensity descriptors:
- Geometric: Area, Perimeter, Equivalent Diameter, Major/Minor Axes, Aspect Ratio,
             Eccentricity, Solidity, Circularity, Orientation, Centroid.
- Intensity: Mean RGB, Mean Hematoxylin OD, Mean Eosin OD (via Ruifrok deconvolution).
- Image-level aggregated morphology statistics.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from research.cellular.model_adapter import ClassicalHEDWatershedAdapter


def compute_instance_geometric_features(
    instance_map: np.ndarray,
    instance_id: int,
) -> Dict[str, Any]:
    """
    Computes rigorous geometric descriptors for a single nuclear instance.

    Args:
        instance_map: 2D integer array (H, W).
        instance_id: Positive integer ID of the target nucleus.

    Returns:
        Dict of geometric descriptors.
    """
    binary_mask = (instance_map == instance_id).astype(np.uint8)
    area = float(np.sum(binary_mask))

    if area == 0:
        return {
            "instance_id": instance_id,
            "centroid_x": 0.0,
            "centroid_y": 0.0,
            "area_px2": 0.0,
            "perimeter_px": 0.0,
            "equivalent_diameter_px": 0.0,
            "major_axis_length_px": 0.0,
            "minor_axis_length_px": 0.0,
            "aspect_ratio": 1.0,
            "eccentricity": 0.0,
            "solidity": 0.0,
            "circularity": 0.0,
            "orientation_deg": 0.0,
        }

    # Extract external contour
    cnts, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return {
            "instance_id": instance_id,
            "centroid_x": 0.0,
            "centroid_y": 0.0,
            "area_px2": area,
            "perimeter_px": 0.0,
            "equivalent_diameter_px": math.sqrt(4.0 * area / math.pi),
            "major_axis_length_px": 0.0,
            "minor_axis_length_px": 0.0,
            "aspect_ratio": 1.0,
            "eccentricity": 0.0,
            "solidity": 1.0,
            "circularity": 0.0,
            "orientation_deg": 0.0,
        }

    cnt = max(cnts, key=cv2.contourArea)
    perimeter = float(cv2.arcLength(cnt, closed=True))

    # Moments for centroid
    moments = cv2.moments(cnt)
    if moments["m00"] > 0:
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
    else:
        # Fallback to mean coordinate
        rows, cols = np.where(binary_mask > 0)
        cx, cy = float(np.mean(cols)), float(np.mean(rows))

    # Equivalent diameter: sqrt(4 * area / pi)
    eq_diameter = math.sqrt(4.0 * area / math.pi)

    # Circularity: 4 * pi * area / perimeter^2
    if perimeter > 0:
        circ = (4.0 * math.pi * area) / (perimeter * perimeter)
        circularity = float(np.clip(circ, 0.0, 1.0))
    else:
        circularity = 0.0

    # Convex hull for solidity
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = float(np.clip(area / hull_area, 0.0, 1.0)) if hull_area > 0 else 1.0

    # Major/Minor axes and orientation from fitted ellipse (requires >= 5 points)
    if len(cnt) >= 5:
        try:
            ellipse = cv2.fitEllipse(cnt)
            (center_x, center_y), (d1, d2), angle = ellipse
            major_axis = max(d1, d2)
            minor_axis = min(d1, d2)
            orientation = angle if angle <= 90 else angle - 180
        except Exception:
            # Fallback to second-order central moments
            mu20 = moments["mu20"] / (moments["m00"] + 1e-6)
            mu02 = moments["mu02"] / (moments["m00"] + 1e-6)
            mu11 = moments["mu11"] / (moments["m00"] + 1e-6)
            delta = math.sqrt(4.0 * (mu11 ** 2) + (mu20 - mu02) ** 2)
            major_axis = 2.0 * math.sqrt(2.0 * max(mu20 + mu02 + delta, 0.0))
            minor_axis = 2.0 * math.sqrt(2.0 * max(mu20 + mu02 - delta, 0.0))
            orientation = 0.5 * math.degrees(math.atan2(2.0 * mu11, mu20 - mu02 + 1e-6))
    else:
        # For small 3-4 pixel objects, use minimum area rectangle
        rect = cv2.minAreaRect(cnt)
        (center_x, center_y), (w_box, h_box), angle = rect
        major_axis = max(w_box, h_box)
        minor_axis = min(w_box, h_box)
        orientation = angle

    aspect_ratio = float(major_axis / max(minor_axis, 1.0))

    # Eccentricity = sqrt(1 - (minor/major)^2)
    if major_axis > 0:
        axis_ratio = min(minor_axis / major_axis, 1.0)
        eccentricity = float(math.sqrt(max(1.0 - (axis_ratio ** 2), 0.0)))
    else:
        eccentricity = 0.0

    return {
        "instance_id": instance_id,
        "centroid_x": round(cx, 2),
        "centroid_y": round(cy, 2),
        "area_px2": round(area, 1),
        "perimeter_px": round(perimeter, 2),
        "equivalent_diameter_px": round(eq_diameter, 2),
        "major_axis_length_px": round(float(major_axis), 2),
        "minor_axis_length_px": round(float(minor_axis), 2),
        "aspect_ratio": round(aspect_ratio, 3),
        "eccentricity": round(eccentricity, 3),
        "solidity": round(solidity, 3),
        "circularity": round(circularity, 3),
        "orientation_deg": round(float(orientation), 1),
    }


def compute_instance_intensity_features(
    image: np.ndarray,
    instance_map: np.ndarray,
    instance_id: int,
    h_channel: Optional[np.ndarray] = None,
    e_channel: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Computes mean RGB and optical density intensity features for an instance.
    """
    mask = (instance_map == instance_id)
    if not np.any(mask):
        return {
            "mean_intensity_r": 0.0,
            "mean_intensity_g": 0.0,
            "mean_intensity_b": 0.0,
            "mean_hematoxylin_od": 0.0,
            "mean_eosin_od": 0.0,
        }

    r_mean = float(np.mean(image[mask, 0]))
    g_mean = float(np.mean(image[mask, 1]))
    b_mean = float(np.mean(image[mask, 2]))

    # If HED channels not precomputed, compute them
    if h_channel is None or e_channel is None:
        hed_adapter = ClassicalHEDWatershedAdapter()
        h_channel, e_channel = hed_adapter.deconvolve_hed(image)

    h_mean = float(np.mean(h_channel[mask]))
    e_mean = float(np.mean(e_channel[mask]))

    return {
        "mean_intensity_r": round(r_mean, 2),
        "mean_intensity_g": round(g_mean, 2),
        "mean_intensity_b": round(b_mean, 2),
        "mean_hematoxylin_od": round(h_mean, 4),
        "mean_eosin_od": round(e_mean, 4),
    }


def extract_cellular_morphology(
    image: np.ndarray,
    instance_map: np.ndarray,
    image_id: str = "unknown",
    class_name: str = "unknown",
) -> pd.DataFrame:
    """
    Extracts complete geometric and intensity morphology features for all instances in an image.

    Returns:
        pd.DataFrame containing one row per detected nucleus instance.
    """
    unique_ids = [int(uid) for uid in np.unique(instance_map) if uid > 0]
    if not unique_ids:
        # Return empty DataFrame with correct column schema
        cols = [
            "image_id", "class_name", "instance_id", "centroid_x", "centroid_y",
            "area_px2", "perimeter_px", "equivalent_diameter_px",
            "major_axis_length_px", "minor_axis_length_px", "aspect_ratio",
            "eccentricity", "solidity", "circularity", "orientation_deg",
            "mean_intensity_r", "mean_intensity_g", "mean_intensity_b",
            "mean_hematoxylin_od", "mean_eosin_od"
        ]
        return pd.DataFrame(columns=cols)

    # Precompute HED deconvolution once per image
    hed_adapter = ClassicalHEDWatershedAdapter()
    h_channel, e_channel = hed_adapter.deconvolve_hed(image)

    records: List[Dict[str, Any]] = []
    for uid in unique_ids:
        geo = compute_instance_geometric_features(instance_map, uid)
        inten = compute_instance_intensity_features(
            image, instance_map, uid, h_channel=h_channel, e_channel=e_channel
        )
        row = {"image_id": image_id, "class_name": class_name, **geo, **inten}
        records.append(row)

    return pd.DataFrame(records)


def compute_image_morphology_summary(
    features_df: pd.DataFrame,
    image_shape: Tuple[int, int] = (224, 224),
    image_id: str = "unknown",
    class_name: str = "unknown",
) -> Dict[str, Any]:
    """
    Aggregates instance-level morphology features into image-level summary descriptors.
    """
    total_pixels = float(image_shape[0] * image_shape[1])
    n_nuclei = len(features_df)

    if n_nuclei == 0:
        return {
            "image_id": image_id,
            "class_name": class_name,
            "total_nuclei_count": 0,
            "nuclear_density_per_kpx": 0.0,
            "nuclear_area_fraction": 0.0,
            "mean_nuclear_area_px2": 0.0,
            "median_nuclear_area_px2": 0.0,
            "std_nuclear_area_px2": 0.0,
            "mean_circularity": 0.0,
            "median_circularity": 0.0,
            "std_circularity": 0.0,
            "mean_eccentricity": 0.0,
            "median_eccentricity": 0.0,
            "std_eccentricity": 0.0,
            "mean_hematoxylin_od": 0.0,
            "mean_eosin_od": 0.0,
        }

    total_area = float(features_df["area_px2"].sum())
    return {
        "image_id": image_id,
        "class_name": class_name,
        "total_nuclei_count": n_nuclei,
        "nuclear_density_per_kpx": round((n_nuclei / total_pixels) * 1000.0, 4),
        "nuclear_area_fraction": round(total_area / total_pixels, 4),
        "mean_nuclear_area_px2": round(float(features_df["area_px2"].mean()), 2),
        "median_nuclear_area_px2": round(float(features_df["area_px2"].median()), 2),
        "std_nuclear_area_px2": round(float(features_df["area_px2"].std()), 2) if n_nuclei > 1 else 0.0,
        "mean_circularity": round(float(features_df["circularity"].mean()), 3),
        "median_circularity": round(float(features_df["circularity"].median()), 3),
        "std_circularity": round(float(features_df["circularity"].std()), 3) if n_nuclei > 1 else 0.0,
        "mean_eccentricity": round(float(features_df["eccentricity"].mean()), 3),
        "median_eccentricity": round(float(features_df["eccentricity"].median()), 3),
        "std_eccentricity": round(float(features_df["eccentricity"].std()), 3) if n_nuclei > 1 else 0.0,
        "mean_hematoxylin_od": round(float(features_df["mean_hematoxylin_od"].mean()), 4),
        "mean_eosin_od": round(float(features_df["mean_eosin_od"].mean()), 4),
    }

