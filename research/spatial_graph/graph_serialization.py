"""Graph Serialization, Storage, and Manifest Persistence Engine.

Serializes PyG / PyTorch spatial graph objects to disk, maintains
reproducible image-to-graph manifests, and aggregates topological statistics.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch

try:
    from torch_geometric.data import Data as PyGData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from research.spatial_graph.graph_builder import (
    CANONICAL_CLASSES,
    CLASS_TO_INDEX,
    compute_graph_statistics,
)

logger = logging.getLogger("spatial_graph.serialization")


def serialize_spatial_graph(
    graph: Any,
    output_path: Union[str, Path],
) -> str:
    """
    Serializes a single spatial graph object to a .pt file.

    Args:
        graph: PyG Data object or custom dict.
        output_path: Destination path (.pt).

    Returns:
        abs_path: Absolute saved file path.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, str(p))
    return str(p.resolve()).replace("\\", "/")


def load_spatial_graph(filepath: Union[str, Path]) -> Any:
    """Loads a serialized spatial graph from disk."""
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"Spatial graph file not found: {p}")
    return torch.load(str(p), weights_only=False)


def save_graph_cohort_manifest(
    graphs_list: List[Any],
    output_csv_path: Union[str, Path],
    graphs_dir: Union[str, Path],
) -> pd.DataFrame:
    """
    Serializes all graphs in the cohort, computes topological statistics,
    and saves the master `spatial_graph_manifest.csv`.

    Args:
        graphs_list: List of PyG Data or dict graph objects.
        output_csv_path: Destination CSV path.
        graphs_dir: Directory where individual .pt graph files are saved.

    Returns:
        pd.DataFrame: Cohort manifest.
    """
    g_dir = Path(graphs_dir)
    g_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for g in graphs_list:
        img_id = str(g.image_id if hasattr(g, "image_id") else g["image_id"])
        c_name = str(g.class_name if hasattr(g, "class_name") else g["class_name"])
        safe_id = img_id.replace("/", "_").replace(".png", "")

        pt_file = g_dir / f"{safe_id}_graph.pt"
        saved_path = serialize_spatial_graph(g, pt_file)

        stats = compute_graph_statistics(g)
        records.append({
            "image_id": img_id,
            "class_name": c_name,
            "class_index": CLASS_TO_INDEX.get(c_name, -1),
            "graph_file_path": saved_path,
            **stats,
        })

    manifest_df = pd.DataFrame(records)
    out_csv = Path(output_csv_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(out_csv, index=False)
    logger.info("Saved spatial graph manifest with %d records to %s", len(manifest_df), out_csv)
    return manifest_df


def compute_class_graph_summary(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes class-level aggregated graph statistics (mean ± std) across the 8 classes.
    """
    records = []
    for c in CANONICAL_CLASSES:
        sub = manifest_df[manifest_df["class_name"] == c]
        if sub.empty:
            continue
        records.append({
            "class_name": c,
            "sample_count": len(sub),
            "mean_nodes": round(float(sub["num_nodes"].mean()), 2),
            "std_nodes": round(float(sub["num_nodes"].std()), 2),
            "mean_edges_undirected": round(float(sub["num_edges_undirected"].mean()), 2),
            "mean_degree": round(float(sub["mean_degree"].mean()), 2),
            "mean_density": round(float(sub["graph_density"].mean()), 4),
            "mean_edge_dist_px": round(float(sub["mean_edge_dist_px"].mean()), 2),
            "mean_components": round(float(sub["connected_components"].mean()), 2),
            "mean_clustering_coeff": round(float(sub["clustering_coefficient"].mean()), 4),
        })
    return pd.DataFrame(records)

