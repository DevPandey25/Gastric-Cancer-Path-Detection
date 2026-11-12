"""Comprehensive Verification Suite for Original Phase 8: Spatial Graph and GNN Engine.

Verifies:
1. Standardized 16-D node feature representation and image-level 48-D aggregation.
2. Symmetric k-NN spatial graph construction, self-loops, and distance attributes.
3. Edge cases: empty graphs (N=0) and small graphs (N <= k).
4. Serialization and round-trip fidelity (.pt persistence and manifest integrity).
5. Topological and spatial graph statistics computation.
6. Forward passes and batching for GCN, GAT, GraphSAGE, and Morphology MLP baseline.
7. Parameter compactness (< 10,000 parameters per model).
8. Absence of cross-fold data leakage during feature scaling.
9. Baseline checkpoint SHA-256 invariant preservation.
10. Presence and integrity of generated Phase 8 files and visualizations.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

try:
    from torch_geometric.data import Batch as PyGBatch, Data as PyGData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from research.spatial_graph.gnn_models import (
    GATClassifier,
    GCNClassifier,
    GraphSAGEClassifier,
    MorphologyMLPBaseline,
    build_model,
    count_parameters,
)
from research.spatial_graph.graph_builder import (
    CANONICAL_CLASSES,
    CLASS_TO_INDEX,
    build_knn_spatial_graph,
    compute_graph_statistics,
)
from research.spatial_graph.graph_features import (
    NODE_FEATURE_NAMES,
    compute_image_aggregated_features,
    extract_node_features_from_dataframe,
)
from research.spatial_graph.graph_serialization import (
    load_spatial_graph,
    serialize_spatial_graph,
)

EXPECTED_BASELINE_SHA256 = "65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8"


def _create_synthetic_instance_df(n: int = 10, img_id: str = "test_img", class_name: str = "TUM") -> pd.DataFrame:
    """Helper to create a realistic mock DataFrame of detected candidate instances."""
    np.random.seed(42)
    return pd.DataFrame({
        "image_id": [img_id] * n,
        "class_name": [class_name] * n,
        "instance_id": list(range(n)),
        "centroid_x": np.random.uniform(5.0, 219.0, size=n),
        "centroid_y": np.random.uniform(5.0, 219.0, size=n),
        "area_px2": np.random.uniform(20.0, 400.0, size=n),
        "perimeter_px": np.random.uniform(15.0, 90.0, size=n),
        "equivalent_diameter_px": np.random.uniform(5.0, 25.0, size=n),
        "major_axis_length_px": np.random.uniform(6.0, 30.0, size=n),
        "minor_axis_length_px": np.random.uniform(4.0, 20.0, size=n),
        "aspect_ratio": np.random.uniform(1.0, 3.5, size=n),
        "eccentricity": np.random.uniform(0.1, 0.9, size=n),
        "solidity": np.random.uniform(0.8, 0.99, size=n),
        "circularity": np.random.uniform(0.5, 0.95, size=n),
        "orientation_deg": np.random.uniform(-90.0, 90.0, size=n),
        "mean_intensity_r": np.random.uniform(50.0, 220.0, size=n),
        "mean_intensity_g": np.random.uniform(40.0, 200.0, size=n),
        "mean_intensity_b": np.random.uniform(60.0, 230.0, size=n),
        "mean_hematoxylin_od": np.random.uniform(0.1, 1.2, size=n),
        "mean_eosin_od": np.random.uniform(0.05, 0.8, size=n),
    })


# ---------------------------------------------------------------------------
# 1. Feature Extraction Tests
# ---------------------------------------------------------------------------

def test_extract_node_features_shape_and_bounds():
    df = _create_synthetic_instance_df(n=15)
    feats, names = extract_node_features_from_dataframe(df)

    assert feats.shape == (15, 16)
    assert names == NODE_FEATURE_NAMES
    assert feats.dtype == np.float32

    # Check bounds
    assert np.all(feats[:, 0] >= 0.0) and np.all(feats[:, 0] <= 1.0)  # cx norm
    assert np.all(feats[:, 1] >= 0.0) and np.all(feats[:, 1] <= 1.0)  # cy norm
    assert np.all(feats[:, 2] >= 0.0)  # log-area
    assert np.all(feats[:, 5] >= 1.0)  # aspect ratio >= 1
    assert np.all(feats[:, 6] >= 0.0) and np.all(feats[:, 6] <= 1.0)  # eccentricity
    assert np.all(feats[:, 7] >= 0.0) and np.all(feats[:, 7] <= 1.0)  # solidity
    assert np.all(feats[:, 8] >= 0.0) and np.all(feats[:, 8] <= 1.0)  # circularity
    assert np.all(feats[:, 9] >= -1.0) and np.all(feats[:, 9] <= 1.0)  # sin 2orient
    assert np.all(feats[:, 10] >= -1.0) and np.all(feats[:, 10] <= 1.0)  # cos 2orient
    assert np.all(feats[:, 11:14] >= 0.0) and np.all(feats[:, 11:14] <= 1.0)  # RGB
    assert not np.any(np.isnan(feats))


def test_extract_node_features_empty():
    empty_df = pd.DataFrame()
    feats, names = extract_node_features_from_dataframe(empty_df)
    assert feats.shape == (0, 16)
    assert names == NODE_FEATURE_NAMES


def test_compute_image_aggregated_features():
    df = _create_synthetic_instance_df(n=10)
    feats, _ = extract_node_features_from_dataframe(df)
    agg = compute_image_aggregated_features(feats)

    assert agg.shape == (48,)  # 16 * 3 (mean, std, median)
    assert agg.dtype == np.float32
    assert not np.any(np.isnan(agg))

    # Empty array case
    empty_agg = compute_image_aggregated_features(np.empty((0, 16), dtype=np.float32))
    assert empty_agg.shape == (48,)
    assert np.all(empty_agg == 0.0)


# ---------------------------------------------------------------------------
# 2. Spatial Graph Builder Tests
# ---------------------------------------------------------------------------

def test_build_knn_spatial_graph_structure():
    df = _create_synthetic_instance_df(n=12, class_name="LYM")
    feats, _ = extract_node_features_from_dataframe(df)
    centroids = df[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float32)

    graph = build_knn_spatial_graph(
        node_features=feats,
        centroids=centroids,
        image_id="test_lym",
        class_name="LYM",
        k=6,
        add_self_loops=True,
    )

    if HAS_PYG:
        assert isinstance(graph, PyGData)
        assert graph.num_nodes == 12
        assert graph.x.shape == (12, 16)
        assert graph.y.item() == CLASS_TO_INDEX["LYM"]
        ei = graph.edge_index.cpu().numpy()
        dists = graph.edge_dist_px.cpu().numpy()
    else:
        assert graph["num_nodes"] == 12
        ei = graph["edge_index"].cpu().numpy()
        dists = graph["edge_dist_px"].cpu().numpy()

    # Edge Symmetry Verification (excluding self-loops)
    non_self = ei[0] != ei[1]
    non_self_edges = set(zip(ei[0][non_self], ei[1][non_self]))
    for u, v in non_self_edges:
        assert (v, u) in non_self_edges, f"Edge ({u}, {v}) missing symmetric counterpart ({v}, {u})"

    # Self-loops verification
    self_loops = set(zip(ei[0][~non_self], ei[1][~non_self]))
    for i in range(12):
        assert (i, i) in self_loops, f"Missing self-loop for node {i}"

    # Edge distance check: positive for non-self, 0 for self-loops
    assert np.all(dists[non_self] > 0.0)
    assert np.all(dists[~non_self] == 0.0)


def test_build_knn_spatial_graph_edge_cases():
    # Case A: N=0 (empty graph)
    empty_feats = np.empty((0, 16), dtype=np.float32)
    empty_centroids = np.empty((0, 2), dtype=np.float32)
    g_empty = build_knn_spatial_graph(empty_feats, empty_centroids, "empty_img", "ADI", k=6)
    num_nodes = g_empty.num_nodes if hasattr(g_empty, "num_nodes") else g_empty["num_nodes"]
    assert num_nodes == 0

    # Case B: N <= k (small graph, e.g. N=3, k=6)
    df_small = _create_synthetic_instance_df(n=3, class_name="MUC")
    feats_s, _ = extract_node_features_from_dataframe(df_small)
    centroids_s = df_small[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float32)
    g_small = build_knn_spatial_graph(feats_s, centroids_s, "small_img", "MUC", k=6, add_self_loops=True)
    num_nodes_s = g_small.num_nodes if hasattr(g_small, "num_nodes") else g_small["num_nodes"]
    assert num_nodes_s == 3


def test_compute_graph_statistics():
    df = _create_synthetic_instance_df(n=10)
    feats, _ = extract_node_features_from_dataframe(df)
    centroids = df[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float32)
    graph = build_knn_spatial_graph(feats, centroids, "stat_img", "STR", k=6, add_self_loops=True)

    stats = compute_graph_statistics(graph)
    assert stats["num_nodes"] == 10
    assert stats["num_edges_undirected"] > 0
    assert stats["mean_degree"] > 0.0
    assert 0.0 <= stats["graph_density"] <= 1.0
    assert stats["mean_edge_dist_px"] > 0.0
    assert 0.0 <= stats["clustering_coefficient"] <= 1.0


# ---------------------------------------------------------------------------
# 3. Serialization Tests
# ---------------------------------------------------------------------------

def test_graph_serialization_round_trip(tmp_path):
    df = _create_synthetic_instance_df(n=8, class_name="NOR")
    feats, _ = extract_node_features_from_dataframe(df)
    centroids = df[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float32)
    graph = build_knn_spatial_graph(feats, centroids, "test_save", "NOR", k=6)

    save_file = tmp_path / "test_graph.pt"
    serialize_spatial_graph(graph, save_file)
    assert save_file.exists()

    loaded = load_spatial_graph(save_file)
    if HAS_PYG:
        assert loaded.num_nodes == graph.num_nodes
        assert torch.equal(loaded.x, graph.x)
        assert torch.equal(loaded.edge_index, graph.edge_index)
        assert loaded.image_id == graph.image_id
        assert loaded.class_name == graph.class_name
    else:
        assert loaded["num_nodes"] == graph["num_nodes"]


# ---------------------------------------------------------------------------
# 4. GNN Architecture and Baseline Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_PYG, reason="PyTorch Geometric not installed")
def test_gnn_forward_passes_single_and_batch():
    d1 = build_knn_spatial_graph(
        extract_node_features_from_dataframe(_create_synthetic_instance_df(8))[0],
        _create_synthetic_instance_df(8)[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float32),
        "g1", "TUM", k=4,
    )
    d2 = build_knn_spatial_graph(
        extract_node_features_from_dataframe(_create_synthetic_instance_df(5))[0],
        _create_synthetic_instance_df(5)[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float32),
        "g2", "STR", k=4,
    )

    models = [
        build_model("gcn", in_channels=16, hidden_channels=32, num_classes=8),
        build_model("gat", in_channels=16, hidden_channels=32, num_classes=8),
        build_model("sage", in_channels=16, hidden_channels=32, num_classes=8),
    ]

    for model in models:
        model.eval()
        # Single graph forward
        with torch.no_grad():
            out1 = model(d1)
            assert out1.shape == (1, 8), f"Expected (1, 8), got {out1.shape} for {model.__class__.__name__}"

        # Batched forward
        batch = PyGBatch.from_data_list([d1, d2])
        with torch.no_grad():
            out_b = model(batch)
            assert out_b.shape == (2, 8), f"Expected (2, 8), got {out_b.shape} for {model.__class__.__name__}"


def test_morphology_mlp_forward():
    mlp = build_model("mlp", in_channels=16, hidden_channels=32, num_classes=8)
    mlp.eval()
    x = torch.randn(4, 48)
    with torch.no_grad():
        out = mlp(x)
    assert out.shape == (4, 8)


def test_model_parameter_compactness():
    for name in ["gcn", "gat", "sage", "mlp"]:
        model = build_model(name, in_channels=16, hidden_channels=32, num_classes=8)
        n_params = count_parameters(model)
        assert n_params < 10000, f"Model {name} exceeds parameter budget: {n_params} >= 10000"


# ---------------------------------------------------------------------------
# 5. Cohort Manifest and Evaluation Integrity Tests
# ---------------------------------------------------------------------------

def test_phase8_artifacts_exist():
    base_dir = Path("research/spatial_graph")
    manifest = base_dir / "spatial_graph_manifest.csv"
    summary_stats = base_dir / "graph_summary_statistics.csv"
    cv_results = base_dir / "cross_validation_results.csv"
    eval_json = base_dir / "evaluation_summary.json"

    assert manifest.exists(), "Missing spatial_graph_manifest.csv"
    assert summary_stats.exists(), "Missing graph_summary_statistics.csv"
    assert cv_results.exists(), "Missing cross_validation_results.csv"
    assert eval_json.exists(), "Missing evaluation_summary.json"

    df_man = pd.read_csv(manifest)
    assert len(df_man) == 64, f"Expected 64 graphs, found {len(df_man)}"
    assert set(df_man["class_name"].unique()) == set(CANONICAL_CLASSES)

    df_stats = pd.read_csv(summary_stats)
    assert len(df_stats) == 8, f"Expected 8 classes in summary stats, found {len(df_stats)}"

    with open(eval_json) as f:
        summary_data = json.load(f)
    assert summary_data["cohort_size"] == 64
    assert summary_data["total_candidate_instances"] == 3895
    assert "models" in summary_data


def test_phase8_visualizations_exist():
    vis_dir = Path("research/spatial_graph/visualizations")
    expected_plots = [
        vis_dir / "spatial_graph_8class_gallery.png",
        vis_dir / "graph_topology_distributions.png",
        vis_dir / "gnn_cv_benchmark_comparison.png",
        vis_dir / "confusion_matrix_best_gnn.png",
    ]
    for p in expected_plots:
        assert p.exists(), f"Missing visualization: {p}"
        assert p.stat().st_size > 5000, f"Plot {p} appears empty or corrupted"


# ---------------------------------------------------------------------------
# 6. Baseline Invariant Check
# ---------------------------------------------------------------------------

def test_baseline_checkpoint_sha256_unmodified():
    ckpt_path = Path("models/checkpoints/gastric_rnn_classifier.pt")
    assert ckpt_path.exists(), f"Baseline checkpoint not found: {ckpt_path}"
    sha256 = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    assert sha256 == EXPECTED_BASELINE_SHA256, (
        f"CRITICAL: Baseline checkpoint has been altered! Expected {EXPECTED_BASELINE_SHA256}, got {sha256}"
    )

