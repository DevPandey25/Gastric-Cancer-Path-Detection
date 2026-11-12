"""Visualization Engine for Histopathology Candidate-Object Spatial Graphs.

Generates publication-quality figures:
1. Candidate-object spatial graph overlay on H&E RGB patches.
2. 8-class exemplar gallery showing spatial graphs across all canonical tissue types.
3. Cohort-wide topological distributions (node degree and edge Euclidean distance).
4. Cross-validation performance comparison across GNN architectures and non-graph baseline.
5. Multi-class confusion matrix heatmaps.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

try:
    from torch_geometric.data import Data as PyGData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from research.spatial_graph.graph_builder import CANONICAL_CLASSES

logger = logging.getLogger("spatial_graph.visualization")

# Publication style configuration
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 12,
    "figure.dpi": 300,
})

CLASS_COLORS = {
    "ADI": "#4A90E2",
    "DEB": "#D0021B",
    "LYM": "#BD10E0",
    "MUC": "#50E3C2",
    "MUS": "#F5A623",
    "NOR": "#7ED321",
    "STR": "#9013FE",
    "TUM": "#E02020",
}


def overlay_graph_on_image(
    image_np: np.ndarray,
    centroids: np.ndarray,
    edge_index_np: np.ndarray,
    ax: plt.Axes,
    title: str = "",
    node_color: str = "#00FFFF",
    edge_color: str = "#FFFF00",
    node_size: float = 12.0,
    edge_alpha: float = 0.55,
    edge_width: float = 0.8,
) -> None:
    """
    Renders an H&E RGB image with an overlaid candidate-object spatial graph.

    Args:
        image_np: np.ndarray of shape (H, W, 3), uint8.
        centroids: np.ndarray of shape (N, 2), float (x, y coordinates).
        edge_index_np: np.ndarray of shape (2, E), int64.
        ax: Matplotlib axes to draw on.
        title: Plot title.
        node_color: Color of candidate object centroids.
        edge_color: Color of Euclidean proximity edges.
        node_size: Point size for nodes.
        edge_alpha: Transparency of edges.
        edge_width: Line width for edges.
    """
    ax.imshow(image_np)

    # Plot undirected edges (avoid double plotting bidirectional edges and skip self-loops)
    if edge_index_np.shape[1] > 0 and centroids.shape[0] > 0:
        drawn_edges = set()
        for idx in range(edge_index_np.shape[1]):
            u, v = int(edge_index_np[0, idx]), int(edge_index_np[1, idx])
            if u == v:
                continue  # Skip self-loop
            pair = (min(u, v), max(u, v))
            if pair in drawn_edges:
                continue
            drawn_edges.add(pair)

            x_coords = [centroids[u, 0], centroids[v, 0]]
            y_coords = [centroids[u, 1], centroids[v, 1]]
            ax.plot(
                x_coords,
                y_coords,
                color=edge_color,
                alpha=edge_alpha,
                linewidth=edge_width,
                solid_capstyle="round",
            )

    # Plot candidate nodes
    if centroids.shape[0] > 0:
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            c=node_color,
            s=node_size,
            alpha=0.85,
            edgecolors="#000000",
            linewidths=0.4,
            zorder=3,
        )

    ax.set_title(title, fontsize=10, pad=4)
    ax.axis("off")


def render_8class_gallery(
    sample_manifest_df: pd.DataFrame,
    instance_df: pd.DataFrame,
    graphs_dict: Dict[str, Any],
    output_path: Union[str, Path],
) -> str:
    """
    Renders a 2x4 exemplar gallery displaying 1 representative patch per canonical class
    with its candidate-object spatial graph overlay.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
    axes = axes.flatten()

    for idx, class_name in enumerate(CANONICAL_CLASSES):
        ax = axes[idx]
        class_samples = sample_manifest_df[sample_manifest_df["class_name"] == class_name]
        if class_samples.empty:
            ax.text(0.5, 0.5, f"No samples: {class_name}", ha="center", va="center")
            ax.axis("off")
            continue

        # Choose the median-complexity sample for representative visualization
        sample_row = class_samples.iloc[len(class_samples) // 2]
        img_id = sample_row["image_id"]
        img_path = sample_row["image_path"]

        img_np = np.array(Image.open(img_path).convert("RGB"))

        sub_inst = instance_df[instance_df["image_id"] == img_id]
        if not sub_inst.empty:
            centroids = sub_inst[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float32)
        else:
            centroids = np.empty((0, 2), dtype=np.float32)

        g = graphs_dict.get(img_id)
        if g is not None:
            ei = g.edge_index.cpu().numpy() if hasattr(g, "edge_index") else g["edge_index"].cpu().numpy()
            num_nodes = int(g.num_nodes if hasattr(g, "num_nodes") else g["num_nodes"])
            # undirected edge count (excluding self-loops)
            non_self = ei[0] != ei[1]
            num_undirected_edges = int(np.sum(non_self) // 2)
        else:
            ei = np.empty((2, 0), dtype=np.int64)
            num_nodes = len(centroids)
            num_undirected_edges = 0

        title = f"{class_name}\n(N={num_nodes} nodes, E={num_undirected_edges} edges)"
        overlay_graph_on_image(
            image_np=img_np,
            centroids=centroids,
            edge_index_np=ei,
            ax=ax,
            title=title,
            node_color="#00E5FF",
            edge_color="#FFEB3B",
            node_size=16.0,
            edge_alpha=0.6,
            edge_width=0.9,
        )

    plt.suptitle(
        "Candidate-Object Spatial Graph Overlays across 8 Canonical GCHTID Classes (k-NN, k=6)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    fig.savefig(str(out_p), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out_p.resolve()).replace("\\", "/")


def render_topology_distributions(
    manifest_df: pd.DataFrame,
    all_edge_distances_px: List[float],
    output_path: Union[str, Path],
) -> str:
    """
    Renders 2-panel figure showing cohort-wide degree distribution and edge distance distribution.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: Node Degree Distribution across classes
    ax1 = axes[0]
    degrees = manifest_df["mean_degree"].dropna().to_numpy()
    ax1.hist(degrees, bins=15, color="#1f77b4", edgecolor="black", alpha=0.75)
    mean_deg = float(np.mean(degrees)) if len(degrees) > 0 else 0.0
    median_deg = float(np.median(degrees)) if len(degrees) > 0 else 0.0
    ax1.axvline(mean_deg, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_deg:.2f}")
    ax1.axvline(median_deg, color="orange", linestyle=":", linewidth=1.5, label=f"Median: {median_deg:.2f}")
    ax1.set_xlabel("Mean Node Degree per Graph (excluding self-loops)")
    ax1.set_ylabel("Number of Patches (N=64)")
    ax1.set_title("Distribution of Mean Node Degrees (k=6 k-NN)", fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Panel B: Edge Euclidean Distance Distribution
    ax2 = axes[1]
    if len(all_edge_distances_px) > 0:
        dists = np.array(all_edge_distances_px)
        ax2.hist(dists, bins=30, color="#2ca02c", edgecolor="black", alpha=0.75)
        mean_dist = float(np.mean(dists))
        median_dist = float(np.median(dists))
        q25, q75 = float(np.percentile(dists, 25)), float(np.percentile(dists, 75))
        ax2.axvline(mean_dist, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_dist:.1f} px")
        ax2.axvline(median_dist, color="orange", linestyle=":", linewidth=1.5, label=f"Median: {median_dist:.1f} px")
        ax2.axvline(q75, color="purple", linestyle="-.", linewidth=1.2, label=f"75th %ile: {q75:.1f} px")
    ax2.set_xlabel("Edge Euclidean Distance (pixels, 224x224 tile)")
    ax2.set_ylabel("Edge Count")
    ax2.set_title("Distribution of Spatial Edge Proximity (k=6)", fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Candidate-Object Spatial Graph Topological Descriptors (N=64 Pilot Cohort)", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(str(out_p), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out_p.resolve()).replace("\\", "/")


def render_cv_benchmark_chart(
    cv_summary_df: pd.DataFrame,
    output_path: Union[str, Path],
) -> str:
    """
    Renders grouped bar chart of 5-fold cross-validation performance
    (Accuracy, Balanced Accuracy, Macro-F1) across models with error bars.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    models = cv_summary_df["model"].tolist()
    x = np.arange(len(models))
    width = 0.25

    # Metrics and stds
    acc_mean = cv_summary_df["accuracy_mean"].to_numpy()
    acc_std = cv_summary_df["accuracy_std"].to_numpy()

    bal_mean = cv_summary_df["balanced_acc_mean"].to_numpy()
    bal_std = cv_summary_df["balanced_acc_std"].to_numpy()

    f1_mean = cv_summary_df["macro_f1_mean"].to_numpy()
    f1_std = cv_summary_df["macro_f1_std"].to_numpy()

    rects1 = ax.bar(x - width, acc_mean, width, yerr=acc_std, capsize=4, label="Accuracy", color="#1f77b4", alpha=0.85)
    rects2 = ax.bar(x, bal_mean, width, yerr=bal_std, capsize=4, label="Balanced Accuracy", color="#2ca02c", alpha=0.85)
    rects3 = ax.bar(x + width, f1_mean, width, yerr=f1_std, capsize=4, label="Macro F1-Score", color="#ff7f0e", alpha=0.85)

    # Reference random baseline (1 / 8 = 12.5%)
    ax.axhline(0.125, color="gray", linestyle="--", linewidth=1.2, label="Random Guess (12.5%)")

    ax.set_ylabel("Score (0.0 to 1.0)")
    ax.set_title("Stratified 5-Fold Cross-Validation Performance Comparison (N=64 Images)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models], fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=True)

    # Add numeric labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    fig.savefig(str(out_p), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out_p.resolve()).replace("\\", "/")


def render_confusion_matrix_heatmap(
    cm_normalized: np.ndarray,
    model_name: str,
    output_path: Union[str, Path],
) -> str:
    """
    Renders normalized confusion matrix heatmap for an evaluated model.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(cm_normalized, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Proportion", rotation=-90, va="bottom")

    ax.set(
        xticks=np.arange(len(CANONICAL_CLASSES)),
        yticks=np.arange(len(CANONICAL_CLASSES)),
        xticklabels=CANONICAL_CLASSES,
        yticklabels=CANONICAL_CLASSES,
        xlabel="Predicted Class",
        ylabel="Ground Truth Class",
        title=f"Normalized Confusion Matrix: {model_name.upper()} (5-Fold CV Pooled)",
    )

    # Format numbers inside matrix cells
    thresh = 0.5
    for i in range(len(CANONICAL_CLASSES)):
        for j in range(len(CANONICAL_CLASSES)):
            val = cm_normalized[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    fig.savefig(str(out_p), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out_p.resolve()).replace("\\", "/")

