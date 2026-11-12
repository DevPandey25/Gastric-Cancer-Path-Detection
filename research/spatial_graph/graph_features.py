"""Node and Graph Feature Extraction Engine for Histopathology Candidate-Object Graphs.

Constructs standardized continuous feature representations from Phase 7 cellular morphology:
- Normalized spatial centroids in [0, 1].
- Morphological scale & geometry (log-area, perimeter, equivalent diameter, aspect ratio,
  eccentricity, solidity, circularity, harmonic orientation sin/cos).
- Optical density and color intensity descriptors (RGB and Ruifrok HED deconvolution).
- Image-level aggregated morphology statistics for the non-graph baseline.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

NODE_FEATURE_NAMES: List[str] = [
    "centroid_x_norm",
    "centroid_y_norm",
    "log_area",
    "perimeter_norm",
    "equivalent_diameter_norm",
    "aspect_ratio",
    "eccentricity",
    "solidity",
    "circularity",
    "sin_2orient",
    "cos_2orient",
    "intensity_r_norm",
    "intensity_g_norm",
    "intensity_b_norm",
    "hematoxylin_od",
    "eosin_od",
]

TILE_SIZE_PX: float = 224.0
TILE_DIAGONAL_PX: float = math.sqrt(224.0 ** 2 + 224.0 ** 2)  # ~316.78 px


def extract_node_features_from_dataframe(
    instance_df: pd.DataFrame,
    tile_size: float = TILE_SIZE_PX,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extracts standardized (N, 16) node feature matrix from an instance morphology DataFrame.

    Args:
        instance_df: DataFrame containing instance morphology records.
        tile_size: Dimension of square tile in pixels (default: 224.0).

    Returns:
        node_features: np.ndarray of shape (N, 16), float32.
        feature_names: List of string feature names.
    """
    if instance_df.empty:
        return np.empty((0, len(NODE_FEATURE_NAMES)), dtype=np.float32), NODE_FEATURE_NAMES

    n_instances = len(instance_df)
    features = np.zeros((n_instances, len(NODE_FEATURE_NAMES)), dtype=np.float32)

    # 1-2. Normalized Centroids
    cx = instance_df["centroid_x"].to_numpy(dtype=np.float32)
    cy = instance_df["centroid_y"].to_numpy(dtype=np.float32)
    features[:, 0] = np.clip(cx / tile_size, 0.0, 1.0)
    features[:, 1] = np.clip(cy / tile_size, 0.0, 1.0)

    # 3. Log-transformed Area
    area = np.maximum(instance_df["area_px2"].to_numpy(dtype=np.float32), 0.0)
    features[:, 2] = np.log1p(area)

    # 4-5. Perimeter and Equivalent Diameter
    perim = np.maximum(instance_df["perimeter_px"].to_numpy(dtype=np.float32), 0.0)
    eq_d = np.maximum(instance_df["equivalent_diameter_px"].to_numpy(dtype=np.float32), 0.0)
    features[:, 3] = perim / 100.0
    features[:, 4] = eq_d / 50.0

    # 6-9. Shape Descriptors
    ar = np.clip(instance_df["aspect_ratio"].to_numpy(dtype=np.float32), 1.0, 10.0)
    ecc = np.clip(instance_df["eccentricity"].to_numpy(dtype=np.float32), 0.0, 1.0)
    sol = np.clip(instance_df["solidity"].to_numpy(dtype=np.float32), 0.0, 1.0)
    circ = np.clip(instance_df["circularity"].to_numpy(dtype=np.float32), 0.0, 1.0)
    features[:, 5] = ar
    features[:, 6] = ecc
    features[:, 7] = sol
    features[:, 8] = circ

    # 10-11. Harmonic Orientation Representation
    orient_deg = instance_df["orientation_deg"].to_numpy(dtype=np.float32)
    orient_rad = np.radians(orient_deg)
    features[:, 9] = np.sin(2.0 * orient_rad)
    features[:, 10] = np.cos(2.0 * orient_rad)

    # 12-14. Normalized RGB Intensities
    r = np.clip(instance_df["mean_intensity_r"].to_numpy(dtype=np.float32) / 255.0, 0.0, 1.0)
    g = np.clip(instance_df["mean_intensity_g"].to_numpy(dtype=np.float32) / 255.0, 0.0, 1.0)
    b = np.clip(instance_df["mean_intensity_b"].to_numpy(dtype=np.float32) / 255.0, 0.0, 1.0)
    features[:, 11] = r
    features[:, 12] = g
    features[:, 13] = b

    # 15-16. Optical Density Channels
    h_od = np.maximum(instance_df["mean_hematoxylin_od"].to_numpy(dtype=np.float32), 0.0)
    e_od = np.maximum(instance_df["mean_eosin_od"].to_numpy(dtype=np.float32), 0.0)
    features[:, 14] = h_od
    features[:, 15] = e_od

    # Sanitize any NaNs or Infs
    np.nan_to_num(features, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
    return features, NODE_FEATURE_NAMES


def compute_image_aggregated_features(
    node_features: np.ndarray,
) -> np.ndarray:
    """
    Computes summary feature vector (mean, std, median) of all node features in a tile.
    Used for the non-graph baseline model.

    Args:
        node_features: np.ndarray of shape (N, D).

    Returns:
        flat_vector: np.ndarray of shape (D * 3,), float32.
    """
    dim = len(NODE_FEATURE_NAMES)
    if node_features.shape[0] == 0:
        return np.zeros(dim * 3, dtype=np.float32)

    mean_feat = np.mean(node_features, axis=0)
    std_feat = np.std(node_features, axis=0) if node_features.shape[0] > 1 else np.zeros(dim, dtype=np.float32)
    med_feat = np.median(node_features, axis=0)

    return np.concatenate([mean_feat, std_feat, med_feat]).astype(np.float32)

