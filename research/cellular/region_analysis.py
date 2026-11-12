"""Intra-Patch 2x2 Quadrant Spatial Regional Analysis.

Divides 224x224 tile coordinates into 4 canonical quadrants:
- Top-Left (TL): x in [0, 112), y in [0, 112)
- Top-Right (TR): x in [112, 224), y in [0, 112)
- Bottom-Left (BL): x in [0, 112), y in [112, 224)
- Bottom-Right (BR): x in [112, 224), y in [112, 224)

Computes per-quadrant nuclei count, density, mean area, eccentricity, and circularity.
Measures within-image intra-patch spatial heterogeneity and dispersion.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

QUADRANT_DEFINITIONS: Dict[str, Dict[str, Tuple[int, int]]] = {
    "TL": {"x_range": (0, 112), "y_range": (0, 112)},
    "TR": {"x_range": (112, 224), "y_range": (0, 112)},
    "BL": {"x_range": (0, 112), "y_range": (112, 224)},
    "BR": {"x_range": (112, 224), "y_range": (112, 224)},
}


def assign_centroid_to_quadrant(
    centroid_x: float, centroid_y: float, image_size: Tuple[int, int] = (224, 224)
) -> str:
    """Assigns an (x, y) centroid to one of the 4 quadrants: 'TL', 'TR', 'BL', 'BR'."""
    mid_x = image_size[1] / 2.0
    mid_y = image_size[0] / 2.0

    if centroid_y < mid_y:
        return "TL" if centroid_x < mid_x else "TR"
    else:
        return "BL" if centroid_x < mid_x else "BR"


def assign_nuclei_to_quadrants(
    features_df: pd.DataFrame,
    image_size: Tuple[int, int] = (224, 224),
) -> pd.DataFrame:
    """Adds a 'quadrant' column to the instance features DataFrame."""
    if features_df.empty:
        df = features_df.copy()
        df["quadrant"] = []
        return df

    df = features_df.copy()
    quadrants = [
        assign_centroid_to_quadrant(row["centroid_x"], row["centroid_y"], image_size)
        for _, row in df.iterrows()
    ]
    df["quadrant"] = quadrants
    return df


def analyze_quadrant_regions(
    features_df: pd.DataFrame,
    image_id: str = "unknown",
    class_name: str = "unknown",
    image_size: Tuple[int, int] = (224, 224),
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Computes per-quadrant cellular morphology statistics and intra-patch dispersion.

    Returns:
        regional_df: DataFrame with 4 rows (one per quadrant).
        intra_patch_metrics: Dict with spatial dispersion and heterogeneity metrics.
    """
    quad_area = (image_size[0] / 2.0) * (image_size[1] / 2.0)  # 112 x 112 = 12,544 px^2

    df = assign_nuclei_to_quadrants(features_df, image_size)

    quad_records: List[Dict[str, Any]] = []
    counts_list: List[int] = []

    for q_code, bounds in QUADRANT_DEFINITIONS.items():
        q_subset = df[df["quadrant"] == q_code] if not df.empty else pd.DataFrame()
        q_count = len(q_subset)
        counts_list.append(q_count)

        if q_count > 0:
            q_density = round((q_count / quad_area) * 1000.0, 4)  # per 1000 px^2
            mean_area = round(float(q_subset["area_px2"].mean()), 2)
            mean_ecc = round(float(q_subset["eccentricity"].mean()), 3)
            mean_circ = round(float(q_subset["circularity"].mean()), 3)
        else:
            q_density = 0.0
            mean_area = 0.0
            mean_ecc = 0.0
            mean_circ = 0.0

        quad_records.append({
            "image_id": image_id,
            "class_name": class_name,
            "quadrant": q_code,
            "nuclei_count": q_count,
            "nuclei_density_per_kpx": q_density,
            "mean_nuclear_area_px2": mean_area,
            "mean_eccentricity": mean_ecc,
            "mean_circularity": mean_circ,
            "x_bounds": f"{bounds['x_range'][0]}-{bounds['x_range'][1]}",
            "y_bounds": f"{bounds['y_range'][0]}-{bounds['y_range'][1]}",
        })

    regional_df = pd.DataFrame(quad_records)

    # Calculate intra-patch heterogeneity across the 4 quadrants
    mean_count = float(np.mean(counts_list))
    std_count = float(np.std(counts_list))
    min_count = float(np.min(counts_list))
    max_count = float(np.max(counts_list))
    dispersion_ratio = round((max_count - min_count) / max(mean_count, 1.0), 3)

    intra_patch_metrics = {
        "image_id": image_id,
        "class_name": class_name,
        "total_nuclei": sum(counts_list),
        "quadrant_counts": {r["quadrant"]: r["nuclei_count"] for r in quad_records},
        "density_variation_std": round(std_count, 2),
        "density_dispersion_ratio": dispersion_ratio,
    }

    return regional_df, intra_patch_metrics

