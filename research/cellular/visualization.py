"""Publication-Quality Visualization Tools for Cellular Morphology and Nuclei Detection.

Generates:
1. Instance segmentation overlays with random distinct colors and clear boundaries.
2. Nuclear centroid scatter plots with 2x2 quadrant boundaries.
3. Intra-patch 2x2 regional density heatmaps.
4. Individual nucleus morphology gallery crops.
5. Class-level descriptive morphology comparison charts.
6. Contact sheet across the eight GCHTID histopathology tissue classes.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from PIL import Image

# Set deterministic colormap for instances
INSTANCE_COLORMAP_SEED = 42


def generate_instance_colormap(max_instances: int = 500) -> np.ndarray:
    """Generates distinct, vibrant RGB colors for instance visualization."""
    rng = np.random.RandomState(INSTANCE_COLORMAP_SEED)
    # Background is black
    colors = [np.array([0, 0, 0], dtype=np.uint8)]
    for _ in range(max_instances):
        # Generate bright, non-dark colors
        c = rng.randint(50, 255, size=3, dtype=np.uint8)
        colors.append(c)
    return np.array(colors)


def render_instance_overlay(
    image: np.ndarray,
    instance_map: np.ndarray,
    alpha: float = 0.40,
    draw_contours: bool = True,
    contour_color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """
    Renders an alpha-blended instance segmentation overlay on the original H&E tile.

    Args:
        image: uint8 RGB array (H, W, 3).
        instance_map: 2D integer array (H, W).
        alpha: Blending transparency.
        draw_contours: Whether to outline each nucleus boundary.
        contour_color: RGB color for instance boundary outlines.

    Returns:
        blended: uint8 RGB array (H, W, 3).
    """
    h, w = instance_map.shape
    unique_ids = [uid for uid in np.unique(instance_map) if uid > 0]
    n_instances = len(unique_ids)

    if n_instances == 0:
        return image.copy()

    # Generate pseudo-random colors for instances
    colors = generate_instance_colormap(max(n_instances + 10, 500))
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for idx, uid in enumerate(unique_ids):
        color = colors[(idx % 499) + 1]
        colored_mask[instance_map == uid] = color

    # Blend image and colored mask where instance > 0
    blended = image.astype(np.float32).copy()
    mask_pixels = (instance_map > 0)
    blended[mask_pixels] = (1.0 - alpha) * blended[mask_pixels] + alpha * colored_mask[mask_pixels]
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # Draw sharp contours
    if draw_contours:
        for uid in unique_ids:
            bin_mask = (instance_map == uid).astype(np.uint8)
            cnts, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(blended, cnts, -1, contour_color, 1)

    return blended


def render_centroid_map(
    image: np.ndarray,
    features_df: pd.DataFrame,
    quadrant_lines: bool = True,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Renders nuclear centroids overlaid on the H&E image, with optional 2x2 quadrant lines.
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.imshow(image)

    if not features_df.empty:
        # Scatter centroids color-coded by circularity or area
        scatter = ax.scatter(
            features_df["centroid_x"],
            features_df["centroid_y"],
            c=features_df["circularity"],
            cmap="plasma",
            s=22,
            edgecolors="black",
            linewidths=0.5,
            alpha=0.9,
        )
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Circularity (4π·Area / P²)", fontsize=10)

    if quadrant_lines:
        h, w = image.shape[:2]
        ax.axvline(w / 2.0, color="#FF3333", linestyle="--", linewidth=1.5, alpha=0.85)
        ax.axhline(h / 2.0, color="#FF3333", linestyle="--", linewidth=1.5, alpha=0.85)
        # Quadrant labels
        ax.text(10, 20, "TL", color="#FF3333", fontsize=11, fontweight="bold")
        ax.text(w - 30, 20, "TR", color="#FF3333", fontsize=11, fontweight="bold")
        ax.text(10, h - 15, "BL", color="#FF3333", fontsize=11, fontweight="bold")
        ax.text(w - 30, h - 15, "BR", color="#FF3333", fontsize=11, fontweight="bold")

    ax.set_title(title or f"Nuclear Centroids (N={len(features_df)})", fontsize=12, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def render_quadrant_density_map(
    image: np.ndarray,
    regional_df: pd.DataFrame,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Renders 2x2 intra-patch quadrant density comparison with numerical counts.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2), dpi=150)

    # Panel 1: Original image with quadrant grid
    ax1.imshow(image)
    h, w = image.shape[:2]
    ax1.axvline(w / 2.0, color="#FF3333", linestyle="--", linewidth=1.5)
    ax1.axhline(h / 2.0, color="#FF3333", linestyle="--", linewidth=1.5)

    # Annotate counts on Panel 1
    coords_map = {
        "TL": (w * 0.25, h * 0.25),
        "TR": (w * 0.75, h * 0.25),
        "BL": (w * 0.25, h * 0.75),
        "BR": (w * 0.75, h * 0.75),
    }
    for _, row in regional_df.iterrows():
        q = row["quadrant"]
        cnt = row["nuclei_count"]
        pos = coords_map.get(q, (0, 0))
        ax1.text(
            pos[0], pos[1], f"{q}: {cnt}",
            color="white", fontsize=12, fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#CC0000", alpha=0.7)
        )
    ax1.set_title("2×2 Quadrant Partition", fontsize=12, fontweight="bold")
    ax1.axis("off")

    # Panel 2: Density Heatmap (2x2 grid)
    density_grid = np.zeros((2, 2), dtype=np.float32)
    grid_pos = {"TL": (0, 0), "TR": (0, 1), "BL": (1, 0), "BR": (1, 1)}
    for _, row in regional_df.iterrows():
        r, c = grid_pos.get(row["quadrant"], (0, 0))
        density_grid[r, c] = row["nuclei_density_per_kpx"]

    im = ax2.imshow(density_grid, cmap="YlOrRd", vmin=0)
    for q, (r, c) in grid_pos.items():
        val = density_grid[r, c]
        ax2.text(
            c, r, f"{q}\n{val:.2f} / kpx²",
            color="black" if val < 0.6 * density_grid.max() else "white",
            fontsize=12, fontweight="bold", ha="center", va="center"
        )
    ax2.set_xticks([0, 1], labels=["Left", "Right"])
    ax2.set_yticks([0, 1], labels=["Top", "Bottom"])
    ax2.set_title("Nuclear Density Heatmap (/1000 px²)", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def render_single_nucleus_gallery(
    image: np.ndarray,
    features_df: pd.DataFrame,
    instance_map: np.ndarray,
    top_k: int = 6,
    crop_size: int = 40,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Renders high-resolution zoom-ins of individual detected nuclei with their morphology metrics.
    """
    if features_df.empty or len(features_df) == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No Nuclei Detected", ha="center", va="center")
        ax.axis("off")
        return fig

    # Select representative nuclei (e.g. median, min, max area)
    k = min(top_k, len(features_df))
    sorted_df = features_df.sort_values(by="area_px2")
    sample_indices = np.linspace(0, len(features_df) - 1, k, dtype=int)
    sampled = sorted_df.iloc[sample_indices]

    fig, axes = plt.subplots(1, k, figsize=(k * 2.5, 3.2), dpi=150)
    if k == 1:
        axes = [axes]

    h_img, w_img = image.shape[:2]
    half_crop = crop_size // 2

    for ax, (_, row) in zip(axes, sampled.iterrows()):
        inst_id = int(row["instance_id"])
        cx, cy = int(row["centroid_x"]), int(row["centroid_y"])

        # Safe bounding crop
        y1 = max(0, cy - half_crop)
        y2 = min(h_img, cy + half_crop)
        x1 = max(0, cx - half_crop)
        x2 = min(w_img, cx + half_crop)

        crop = image[y1:y2, x1:x2].copy()
        inst_crop = instance_map[y1:y2, x1:x2] == inst_id

        # Draw contour on crop
        cnts, _ = cv2.findContours(inst_crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(crop, cnts, -1, (0, 255, 0), 1)

        ax.imshow(crop)
        ax.set_title(
            f"ID #{inst_id}\n"
            f"Area: {row['area_px2']:.0f} px²\n"
            f"Circ: {row['circularity']:.2f}\n"
            f"Ecc: {row['eccentricity']:.2f}",
            fontsize=8.5,
        )
        ax.axis("off")

    fig.suptitle("Representative Individual Nuclei Morphology Crops", fontsize=12, fontweight="bold", y=1.05)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def render_contact_sheet(
    pilot_exemplars: List[Dict[str, Any]],
    save_path: str,
) -> str:
    """
    Renders an 8-class contact sheet displaying for each canonical class:
    Column 1: Original H&E Tile
    Column 2: Detected Instance Map (pseudo-colored)
    Column 3: Segmentation Overlay with Boundaries & Count

    Args:
        pilot_exemplars: List of 8 dicts (one per class) containing:
                         'class_name', 'image', 'instance_map', 'nuclei_count'
        save_path: Output PNG path.

    Returns:
        Absolute saved path.
    """
    n_classes = len(pilot_exemplars)
    fig, axes = plt.subplots(n_classes, 3, figsize=(10, max(n_classes * 3.0, 3.5)), dpi=150)
    if n_classes == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, ex in enumerate(pilot_exemplars):
        c_name = ex["class_name"]
        img = ex["image"]
        inst_map = ex["instance_map"]
        cnt = ex["nuclei_count"]

        # 1. Original
        axes[i, 0].imshow(img)
        axes[i, 0].set_ylabel(f"{c_name}\n(N={cnt})", fontsize=11, fontweight="bold", rotation=0, labelpad=40, va="center")
        if i == 0:
            axes[i, 0].set_title("Input H&E Tile", fontsize=12, fontweight="bold")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        # 2. Pseudo-colored Instance Map
        colors = generate_instance_colormap(max(cnt + 10, 100))
        h, w = inst_map.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        uids = [u for u in np.unique(inst_map) if u > 0]
        for idx, u in enumerate(uids):
            colored[inst_map == u] = colors[(idx % 99) + 1]

        axes[i, 1].imshow(colored)
        if i == 0:
            axes[i, 1].set_title("Nuclear Instances", fontsize=12, fontweight="bold")
        axes[i, 1].axis("off")

        # 3. Overlay
        overlay = render_instance_overlay(img, inst_map, alpha=0.45, draw_contours=True)
        axes[i, 2].imshow(overlay)
        if i == 0:
            axes[i, 2].set_title("Boundary Overlay", fontsize=12, fontweight="bold")
        axes[i, 2].axis("off")

    fig.suptitle(
        "GastricPath AI — Original Phase 7: Pilot Nuclei Detections Across 8 GCHTID Classes",
        fontsize=13, fontweight="bold", y=1.01
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    abs_path = os.path.abspath(save_path)
    return abs_path


def render_morphology_comparison_charts(
    image_summary_df: pd.DataFrame,
    save_path: str,
) -> str:
    """
    Renders class-level comparative charts for nuclei count, mean nuclear area, and circularity.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)

    classes = sorted(image_summary_df["class_name"].unique())

    # 1. Total Nuclei Count per Class
    data_counts = [image_summary_df[image_summary_df["class_name"] == c]["total_nuclei_count"] for c in classes]
    axes[0].boxplot(data_counts, tick_labels=classes, patch_artist=True,
                    boxprops=dict(facecolor="#A6CEE3", color="#1F78B4"),
                    medianprops=dict(color="#E31A1C", linewidth=1.5))
    axes[0].set_title("Nuclei Count per Tile", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Count per 224×224 Tile", fontsize=10)
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)

    # 2. Mean Nuclear Area (px^2)
    data_area = [image_summary_df[image_summary_df["class_name"] == c]["mean_nuclear_area_px2"] for c in classes]
    axes[1].boxplot(data_area, tick_labels=classes, patch_artist=True,
                    boxprops=dict(facecolor="#B2DF8A", color="#33A02C"),
                    medianprops=dict(color="#E31A1C", linewidth=1.5))
    axes[1].set_title("Mean Nuclear Area (px²)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Area (pixels²)", fontsize=10)
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)

    # 3. Mean Circularity
    data_circ = [image_summary_df[image_summary_df["class_name"] == c]["mean_circularity"] for c in classes]
    axes[2].boxplot(data_circ, tick_labels=classes, patch_artist=True,
                    boxprops=dict(facecolor="#FDBF6F", color="#FF7F00"),
                    medianprops=dict(color="#E31A1C", linewidth=1.5))
    axes[2].set_title("Mean Circularity", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Circularity (0..1)", fontsize=10)
    axes[2].grid(axis="y", linestyle="--", alpha=0.6)

    fig.suptitle(
        "GastricPath AI — Original Phase 7: Quantitative Morphology Comparison Across GCHTID Classes (N=64)",
        fontsize=13, fontweight="bold", y=1.03
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    return os.path.abspath(save_path)
