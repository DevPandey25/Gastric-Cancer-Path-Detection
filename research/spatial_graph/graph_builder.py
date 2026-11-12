"""Spatial Graph Construction Engine for Histopathology Candidate-Object Graphs.

Builds symmetric k-nearest-neighbor (kNN, k=6) spatial graphs based on Euclidean
centroid distances between candidate segmented objects.

Computes:
- Node feature matrices (N, 16).
- Symmetric bidirectional edge index tensors (2, E).
- Normalized edge distance attributes.
- Self-loops for GNN message-passing stability.
- Graph-level topological and spatial statistics (degrees, density, clustering, components).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.spatial.distance as sp_dist
import torch

try:
    from torch_geometric.data import Data as PyGData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from research.spatial_graph.graph_features import (
    NODE_FEATURE_NAMES,
    TILE_DIAGONAL_PX,
    TILE_SIZE_PX,
    extract_node_features_from_dataframe,
)

CANONICAL_CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]
CLASS_TO_INDEX: Dict[str, int] = {c: i for i, c in enumerate(CANONICAL_CLASSES)}


def build_knn_spatial_graph(
    node_features: np.ndarray,
    centroids: np.ndarray,
    image_id: str = "unknown",
    class_name: str = "unknown",
    k: int = 6,
    add_self_loops: bool = True,
    tile_diagonal: float = TILE_DIAGONAL_PX,
) -> Any:
    """
    Constructs a symmetric k-nearest neighbor spatial graph for candidate segmented objects in an image.

    Args:
        node_features: np.ndarray of shape (N, 16).
        centroids: np.ndarray of shape (N, 2) in pixel coordinates (x, y).
        image_id: Unique string identifier for the tile.
        class_name: Ground-truth patch-level GCHTID class.
        k: Number of nearest neighbors per node (default: 6).
        add_self_loops: Whether to include (i, i) self-edges for GCN stability.
        tile_diagonal: Maximum diagonal normalization constant.

    Returns:
        PyTorch Geometric Data object (or standalone custom dictionary if PyG is not present).
    """
    n_nodes = node_features.shape[0]
    class_idx = CLASS_TO_INDEX.get(class_name, -1)

    # Edge Case: 0 nodes (empty candidate detection)
    if n_nodes == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float32)
        edge_dist_px = torch.empty(0, dtype=torch.float32)
        x_tensor = torch.empty((0, len(NODE_FEATURE_NAMES)), dtype=torch.float32)
        y_tensor = torch.tensor([class_idx], dtype=torch.long) if class_idx >= 0 else torch.tensor([0], dtype=torch.long)

        if HAS_PYG:
            return PyGData(
                x=x_tensor,
                edge_index=edge_index,
                edge_attr=edge_attr,
                edge_dist_px=edge_dist_px,
                y=y_tensor,
                image_id=image_id,
                class_name=class_name,
                num_nodes=0,
            )
        return {
            "x": x_tensor, "edge_index": edge_index, "edge_attr": edge_attr,
            "edge_dist_px": edge_dist_px, "y": y_tensor, "image_id": image_id,
            "class_name": class_name, "num_nodes": 0,
        }

    # Compute pairwise Euclidean centroid distance matrix
    dist_matrix = sp_dist.cdist(centroids, centroids, metric="euclidean")

    # Find k-nearest neighbors for each node (excluding self)
    adj_set = set()
    actual_k = min(k, max(n_nodes - 1, 0))

    if actual_k > 0:
        for i in range(n_nodes):
            # Sort node distances, skipping index 0 (self-distance = 0.0)
            nearest_indices = np.argsort(dist_matrix[i])[1 : actual_k + 1]
            for j in nearest_indices:
                # Add undirected (symmetric) connection
                adj_set.add((min(i, j), max(i, j)))

    # Convert undirected edges to directed bidirectional edges for PyG
    directed_edges: List[Tuple[int, int]] = []
    directed_distances_px: List[float] = []

    for u, v in sorted(adj_set):
        d = float(dist_matrix[u, v])
        # Forward edge (u -> v)
        directed_edges.append((u, v))
        directed_distances_px.append(d)
        # Reverse edge (v -> u)
        directed_edges.append((v, u))
        directed_distances_px.append(d)

    # Add self-loops if requested
    if add_self_loops:
        for i in range(n_nodes):
            directed_edges.append((i, i))
            directed_distances_px.append(0.0)

    # Convert to tensors
    if directed_edges:
        edge_index_np = np.array(directed_edges, dtype=np.int64).T  # (2, E)
        edge_index = torch.from_numpy(edge_index_np).long()
        edge_dist_px = torch.tensor(directed_distances_px, dtype=torch.float32)
        edge_attr = (edge_dist_px / tile_diagonal).unsqueeze(1).float()  # (E, 1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_dist_px = torch.empty(0, dtype=torch.float32)
        edge_attr = torch.empty((0, 1), dtype=torch.float32)

    x_tensor = torch.from_numpy(node_features).float()
    y_tensor = torch.tensor([class_idx], dtype=torch.long)

    if HAS_PYG:
        data = PyGData(
            x=x_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_dist_px=edge_dist_px,
            y=y_tensor,
            image_id=image_id,
            class_name=class_name,
            num_nodes=n_nodes,
        )
        return data

    return {
        "x": x_tensor,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_dist_px": edge_dist_px,
        "y": y_tensor,
        "image_id": image_id,
        "class_name": class_name,
        "num_nodes": n_nodes,
    }


def compute_graph_statistics(graph: Any) -> Dict[str, Any]:
    """
    Computes comprehensive topological and spatial graph descriptors:
    - Node count, Edge count (excluding self-loops)
    - Mean, median, max, min node degree
    - Graph density
    - Mean and standard deviation of edge distance (px)
    - Connected component count
    - Approximate clustering coefficient
    """
    n_nodes = int(graph.num_nodes if hasattr(graph, "num_nodes") else graph["num_nodes"])
    if n_nodes == 0:
        return {
            "num_nodes": 0,
            "num_edges_undirected": 0,
            "mean_degree": 0.0,
            "median_degree": 0.0,
            "max_degree": 0,
            "min_degree": 0,
            "graph_density": 0.0,
            "mean_edge_dist_px": 0.0,
            "std_edge_dist_px": 0.0,
            "connected_components": 0,
            "clustering_coefficient": 0.0,
        }

    edge_index = graph.edge_index if hasattr(graph, "edge_index") else graph["edge_index"]
    edge_dist_px = graph.edge_dist_px if hasattr(graph, "edge_dist_px") else graph["edge_dist_px"]

    if edge_index.ndim == 2 and edge_index.shape[1] > 0:
        ei = edge_index.cpu().numpy()
        # Filter out self-loops for degree and distance calculation
        non_self = ei[0] != ei[1]
        ei_clean = ei[:, non_self]
        dists_clean = edge_dist_px.cpu().numpy()[non_self] if len(edge_dist_px) > 0 else np.array([])
    else:
        ei_clean = np.empty((2, 0), dtype=np.int64)
        dists_clean = np.array([], dtype=np.float32)

    # Number of undirected edges
    n_undirected_edges = int(ei_clean.shape[1] // 2)

    # Degree per node
    degrees = np.zeros(n_nodes, dtype=np.int32)
    if ei_clean.shape[1] > 0:
        src, counts = np.unique(ei_clean[0], return_counts=True)
        degrees[src] = counts

    mean_deg = float(np.mean(degrees))
    med_deg = float(np.median(degrees))
    max_deg = int(np.max(degrees)) if n_nodes > 0 else 0
    min_deg = int(np.min(degrees)) if n_nodes > 0 else 0

    # Graph density: 2 * E_undir / (V * (V - 1))
    if n_nodes > 1:
        density = float(2.0 * n_undirected_edges / (n_nodes * (n_nodes - 1)))
    else:
        density = 0.0

    mean_dist = float(np.mean(dists_clean)) if len(dists_clean) > 0 else 0.0
    std_dist = float(np.std(dists_clean)) if len(dists_clean) > 1 else 0.0

    # Connected Components via BFS
    visited = np.zeros(n_nodes, dtype=bool)
    n_components = 0
    adj_list: Dict[int, List[int]] = {i: [] for i in range(n_nodes)}
    if ei_clean.shape[1] > 0:
        for u, v in zip(ei_clean[0], ei_clean[1]):
            adj_list[u].append(v)

    for i in range(n_nodes):
        if not visited[i]:
            n_components += 1
            queue = [i]
            visited[i] = True
            while queue:
                curr = queue.pop(0)
                for neighbor in adj_list[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)

    # Local Clustering Coefficient
    local_cluster_coeffs = []
    for i in range(n_nodes):
        neighbors = set(adj_list[i])
        k_i = len(neighbors)
        if k_i < 2:
            local_cluster_coeffs.append(0.0)
            continue
        # Count links between neighbors
        links = 0
        for n1 in neighbors:
            for n2 in adj_list[n1]:
                if n2 in neighbors and n1 < n2:
                    links += 1
        possible_links = k_i * (k_i - 1) / 2.0
        local_cluster_coeffs.append(links / possible_links)

    avg_clustering = float(np.mean(local_cluster_coeffs)) if local_cluster_coeffs else 0.0

    return {
        "num_nodes": n_nodes,
        "num_edges_undirected": n_undirected_edges,
        "mean_degree": round(mean_deg, 2),
        "median_degree": round(med_deg, 2),
        "max_degree": max_deg,
        "min_degree": min_deg,
        "graph_density": round(density, 4),
        "mean_edge_dist_px": round(mean_dist, 2),
        "std_edge_dist_px": round(std_dist, 2),
        "connected_components": n_components,
        "clustering_coefficient": round(avg_clustering, 4),
    }

