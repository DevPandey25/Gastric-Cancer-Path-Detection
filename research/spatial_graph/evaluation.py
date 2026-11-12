"""Execution and Cross-Validation Pipeline for Histopathology Candidate-Object Graphs.

Implements end-to-end evaluation:
1. Candidate-object spatial graph generation and serialization for N=64 pilot cohort.
2. Topological and spatial metric extraction across canonical tissue categories.
3. Stratified 5-Fold Cross-Validation across GCN, GAT, GraphSAGE, and non-graph baseline.
4. Strict train-fold-only feature scaling (preventing data leakage).
5. Comprehensive evaluation metrics (accuracy, balanced accuracy, macro-F1, per-class recall).
6. Publication-grade visualization rendering.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.optim import Adam

try:
    from torch_geometric.data import Batch as PyGBatch, Data as PyGData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from research.spatial_graph.gnn_models import build_model, count_parameters
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
    save_graph_cohort_manifest,
    serialize_spatial_graph,
)
from research.spatial_graph.graph_visualization import (
    render_8class_gallery,
    render_confusion_matrix_heatmap,
    render_cv_benchmark_chart,
    render_topology_distributions,
)

logger = logging.getLogger("spatial_graph.evaluation")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = Path("research/spatial_graph")
CELLULAR_DIR = Path("research/cellular")
SAMPLE_MANIFEST_PATH = CELLULAR_DIR / "cellular_sample_manifest.csv"
INSTANCE_FEATURES_PATH = CELLULAR_DIR / "instance_features.csv"
GRAPHS_DIR = BASE_DIR / "graphs"
MANIFEST_PATH = BASE_DIR / "spatial_graph_manifest.csv"
SUMMARY_STATS_PATH = BASE_DIR / "graph_summary_statistics.csv"
CV_RESULTS_PATH = BASE_DIR / "cross_validation_results.csv"
EVAL_SUMMARY_PATH = BASE_DIR / "evaluation_summary.json"
VIS_DIR = BASE_DIR / "visualizations"


def build_and_save_all_graphs(
    sample_manifest_df: pd.DataFrame,
    instance_df: pd.DataFrame,
    graphs_dir: Path = GRAPHS_DIR,
    k: int = 6,
) -> Tuple[List[Any], Dict[str, Any], np.ndarray]:
    """
    Constructs and serializes spatial graphs for all 64 images.

    Returns:
        graphs_list: List of PyG Data objects.
        graphs_dict: Map from image_id to PyG Data object.
        aggregated_features: np.ndarray of shape (64, 48) for non-graph baseline.
    """
    graphs_dir.mkdir(parents=True, exist_ok=True)
    graphs_list: List[Any] = []
    graphs_dict: Dict[str, Any] = {}
    aggregated_list: List[np.ndarray] = []

    for _, row in sample_manifest_df.iterrows():
        img_id = str(row["image_id"])
        c_name = str(row["class_name"])

        # Filter candidate instances for this image
        sub_inst = instance_df[instance_df["image_id"] == img_id]

        if not sub_inst.empty:
            node_feats, _ = extract_node_features_from_dataframe(sub_inst)
            centroids = sub_inst[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float32)
        else:
            node_feats = np.empty((0, len(NODE_FEATURE_NAMES)), dtype=np.float32)
            centroids = np.empty((0, 2), dtype=np.float32)

        # Graph construction
        g = build_knn_spatial_graph(
            node_features=node_feats,
            centroids=centroids,
            image_id=img_id,
            class_name=c_name,
            k=k,
            add_self_loops=True,
        )
        graphs_list.append(g)
        graphs_dict[img_id] = g

        # Non-graph aggregated baseline features (48,)
        agg_feat = compute_image_aggregated_features(node_feats)
        aggregated_list.append(agg_feat)

    aggregated_features = np.vstack(aggregated_list)
    return graphs_list, graphs_dict, aggregated_features


def compute_class_summary_statistics(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates topological statistics grouped by canonical GCHTID class.
    """
    stat_cols = [
        "num_nodes",
        "num_edges_undirected",
        "mean_degree",
        "graph_density",
        "mean_edge_dist_px",
        "clustering_coefficient",
        "connected_components",
    ]
    grouped = manifest_df.groupby("class_name")[stat_cols].agg(["mean", "std"])
    # Flatten column MultiIndex
    grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]
    grouped = grouped.reset_index()
    return grouped


def run_cross_validation(
    graphs_list: List[Any],
    aggregated_features: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    epochs: int = 30,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Performs Stratified 5-Fold Cross-Validation across GCN, GAT, GraphSAGE, and MLP Baseline.
    Node features and aggregated features are strictly scaled on training folds only.
    """
    labels = np.array([int(g.y.item() if hasattr(g, "y") else g["y"].item()) for g in graphs_list])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    models_to_test = ["gcn", "gat", "sage", "mlp"]
    fold_records: List[Dict[str, Any]] = []

    # Store predictions for pooled confusion matrices
    pooled_preds: Dict[str, List[int]] = {m: [] for m in models_to_test}
    pooled_trues: Dict[str, List[int]] = {m: [] for m in models_to_test}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(graphs_list, labels)):
        # 1. Feature Scaling strictly on training fold
        # Collect training node features
        train_node_feats = []
        for i in train_idx:
            g = graphs_list[i]
            x_arr = g.x.numpy() if hasattr(g, "x") else g["x"].numpy()
            if x_arr.shape[0] > 0:
                train_node_feats.append(x_arr)

        if train_node_feats:
            node_scaler = StandardScaler()
            node_scaler.fit(np.vstack(train_node_feats))
        else:
            node_scaler = None

        # Scale graph node features for this fold
        fold_graphs_scaled = []
        for g in graphs_list:
            g_c = g.clone() if hasattr(g, "clone") else dict(g)
            x_arr = g.x.numpy() if hasattr(g, "x") else g["x"].numpy()
            if node_scaler is not None and x_arr.shape[0] > 0:
                x_scaled = node_scaler.transform(x_arr)
            else:
                x_scaled = x_arr
            g_c.x = torch.from_numpy(x_scaled).float()
            fold_graphs_scaled.append(g_c)

        train_graphs = [fold_graphs_scaled[i] for i in train_idx]
        val_graphs = [fold_graphs_scaled[i] for i in val_idx]

        train_pyg_batch = PyGBatch.from_data_list(train_graphs) if HAS_PYG else None
        val_pyg_batch = PyGBatch.from_data_list(val_graphs) if HAS_PYG else None

        # 2. Scale Aggregated Features for MLP Baseline
        agg_scaler = StandardScaler()
        agg_train_scaled = agg_scaler.fit_transform(aggregated_features[train_idx])
        agg_val_scaled = agg_scaler.transform(aggregated_features[val_idx])

        y_train_t = torch.from_numpy(labels[train_idx]).long()
        y_val_t = torch.from_numpy(labels[val_idx]).long()
        y_val_np = labels[val_idx]

        # Train and evaluate each architecture
        for model_name in models_to_test:
            # Seed for fold reproducibility
            torch.manual_seed(random_state + fold_idx * 10)
            model = build_model(model_name=model_name, in_channels=16, hidden_channels=32, num_classes=8)
            optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = nn.CrossEntropyLoss()

            # Train loop
            model.train()
            for ep in range(epochs):
                optimizer.zero_grad()
                if model_name == "mlp":
                    x_in = torch.from_numpy(agg_train_scaled).float()
                    out = model(x_in)
                else:
                    out = model(train_pyg_batch)

                loss = criterion(out, y_train_t)
                loss.backward()
                optimizer.step()

            # Evaluate on validation fold
            model.eval()
            with torch.no_grad():
                if model_name == "mlp":
                    x_val = torch.from_numpy(agg_val_scaled).float()
                    logits = model(x_val)
                else:
                    logits = model(val_pyg_batch)

                preds = torch.argmax(logits, dim=1).cpu().numpy()

            # Record predictions
            pooled_preds[model_name].extend(preds.tolist())
            pooled_trues[model_name].extend(y_val_np.tolist())

            acc = float(accuracy_score(y_val_np, preds))
            bal_acc = float(balanced_accuracy_score(y_val_np, preds))
            f1 = float(f1_score(y_val_np, preds, average="macro", zero_division=0))

            fold_records.append({
                "fold": fold_idx,
                "model": model_name,
                "num_train": len(train_idx),
                "num_val": len(val_idx),
                "accuracy": acc,
                "balanced_acc": bal_acc,
                "macro_f1": f1,
            })

    cv_df = pd.DataFrame(fold_records)

    # Compute summary across folds
    cv_summary_records = []
    model_summaries = {}

    for model_name in models_to_test:
        sub_cv = cv_df[cv_df["model"] == model_name]
        m_acc = float(sub_cv["accuracy"].mean())
        s_acc = float(sub_cv["accuracy"].std())
        m_bal = float(sub_cv["balanced_acc"].mean())
        s_bal = float(sub_cv["balanced_acc"].std())
        m_f1 = float(sub_cv["macro_f1"].mean())
        s_f1 = float(sub_cv["macro_f1"].std())

        t_true = np.array(pooled_trues[model_name])
        t_pred = np.array(pooled_preds[model_name])

        cm = confusion_matrix(t_true, t_pred, labels=list(range(len(CANONICAL_CLASSES))))
        cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1e-8)

        # Per-class recall (sensitivity)
        per_class_rec = {
            c_name: float(cm_norm[c_i, c_i]) for c_i, c_name in enumerate(CANONICAL_CLASSES)
        }

        model_summaries[model_name] = {
            "model_name": model_name,
            "parameter_count": count_parameters(build_model(model_name)),
            "accuracy_mean": m_acc,
            "accuracy_std": s_acc,
            "balanced_acc_mean": m_bal,
            "balanced_acc_std": s_bal,
            "macro_f1_mean": m_f1,
            "macro_f1_std": s_f1,
            "per_class_recall": per_class_rec,
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_norm": cm_norm.tolist(),
        }

        cv_summary_records.append({
            "model": model_name,
            "accuracy_mean": m_acc,
            "accuracy_std": s_acc,
            "balanced_acc_mean": m_bal,
            "balanced_acc_std": s_bal,
            "macro_f1_mean": m_f1,
            "macro_f1_std": s_f1,
        })

    summary_df = pd.DataFrame(cv_summary_records)
    return cv_df, {"summary_df": summary_df, "models": model_summaries}


def run_spatial_graph_pipeline() -> Dict[str, Any]:
    """
    Master driver orchestrating Phase 8 spatial graph construction,
    5-fold cross validation, and publication visualizations.
    """
    logger.info("Starting Original Phase 8: Spatial Graph and GNN Engine...")

    # Load data
    if not SAMPLE_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing sample manifest: {SAMPLE_MANIFEST_PATH}")
    if not INSTANCE_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing instance features: {INSTANCE_FEATURES_PATH}")

    sample_manifest_df = pd.read_csv(SAMPLE_MANIFEST_PATH)
    instance_df = pd.read_csv(INSTANCE_FEATURES_PATH)
    logger.info(f"Loaded {len(sample_manifest_df)} image samples and {len(instance_df)} candidate instances.")

    # 1. Build and serialize graphs
    logger.info("Constructing symmetric k-NN (k=6) candidate-object spatial graphs...")
    graphs_list, graphs_dict, aggregated_features = build_and_save_all_graphs(
        sample_manifest_df=sample_manifest_df,
        instance_df=instance_df,
        graphs_dir=GRAPHS_DIR,
        k=6,
    )

    # Save manifest and topology statistics
    logger.info("Persisting graph manifest and topological summary statistics...")
    manifest_df = save_graph_cohort_manifest(
        graphs_list=graphs_list,
        output_csv_path=MANIFEST_PATH,
        graphs_dir=GRAPHS_DIR,
    )
    class_stats_df = compute_class_summary_statistics(manifest_df)
    class_stats_df.to_csv(SUMMARY_STATS_PATH, index=False)

    # Collect all edge distances for distribution visualization
    all_edge_distances_px: List[float] = []
    for g in graphs_list:
        if hasattr(g, "edge_dist_px") and len(g.edge_dist_px) > 0:
            ei = g.edge_index.cpu().numpy()
            non_self = ei[0] != ei[1]
            if np.any(non_self):
                dists = g.edge_dist_px.cpu().numpy()[non_self]
                # Filter one direction to avoid duplicating undirected distances
                dists_half = dists[::2] if len(dists) > 1 else dists
                all_edge_distances_px.extend(dists_half.tolist())

    # 2. Run Stratified 5-Fold Cross-Validation
    logger.info("Executing Stratified 5-Fold Cross-Validation across GCN, GAT, GraphSAGE, and MLP...")
    cv_df, eval_results = run_cross_validation(
        graphs_list=graphs_list,
        aggregated_features=aggregated_features,
        n_splits=5,
        random_state=42,
        epochs=30,
        lr=0.01,
        weight_decay=1e-4,
    )
    cv_df.to_csv(CV_RESULTS_PATH, index=False)

    # 3. Render Visualizations
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Rendering publication-quality figures...")

    # A. 8-Class Gallery
    gallery_path = VIS_DIR / "spatial_graph_8class_gallery.png"
    render_8class_gallery(
        sample_manifest_df=sample_manifest_df,
        instance_df=instance_df,
        graphs_dict=graphs_dict,
        output_path=gallery_path,
    )

    # B. Topological Distributions
    dist_path = VIS_DIR / "graph_topology_distributions.png"
    render_topology_distributions(
        manifest_df=manifest_df,
        all_edge_distances_px=all_edge_distances_px,
        output_path=dist_path,
    )

    # C. CV Benchmark Comparison
    cv_chart_path = VIS_DIR / "gnn_cv_benchmark_comparison.png"
    render_cv_benchmark_chart(
        cv_summary_df=eval_results["summary_df"],
        output_path=cv_chart_path,
    )

    # D. Confusion Matrix for Best Model
    # Determine best model by balanced accuracy
    best_model_name = eval_results["summary_df"].sort_values(by="balanced_acc_mean", ascending=False).iloc[0]["model"]
    best_cm_norm = np.array(eval_results["models"][best_model_name]["confusion_matrix_norm"])
    cm_path = VIS_DIR / "confusion_matrix_best_gnn.png"
    render_confusion_matrix_heatmap(
        cm_normalized=best_cm_norm,
        model_name=best_model_name,
        output_path=cm_path,
    )

    # 4. Save Master Evaluation Summary JSON
    eval_summary = {
        "phase": "Original Phase 8: Spatial Graph and GNN Engine",
        "cohort_size": len(sample_manifest_df),
        "total_candidate_instances": len(instance_df),
        "classes": CANONICAL_CLASSES,
        "best_model_by_balanced_accuracy": best_model_name,
        "cross_validation_protocol": "Stratified 5-Fold Cross-Validation (N=64)",
        "models": eval_results["models"],
        "mean_nodes_per_graph": float(manifest_df["num_nodes"].mean()),
        "std_nodes_per_graph": float(manifest_df["num_nodes"].std()),
        "mean_edges_per_graph": float(manifest_df["num_edges_undirected"].mean()),
        "mean_degree_cohort": float(manifest_df["mean_degree"].mean()),
        "mean_edge_dist_px_cohort": float(np.mean(all_edge_distances_px)) if all_edge_distances_px else 0.0,
        "visualizations": {
            "gallery": str(gallery_path.resolve()).replace("\\", "/"),
            "topology_distributions": str(dist_path.resolve()).replace("\\", "/"),
            "benchmark_comparison": str(cv_chart_path.resolve()).replace("\\", "/"),
            "confusion_matrix": str(cm_path.resolve()).replace("\\", "/"),
        },
    }

    with open(EVAL_SUMMARY_PATH, "w") as f:
        json.dump(eval_summary, f, indent=2)

    logger.info(f"Original Phase 8 Pipeline Completed Successfully. Summary written to {EVAL_SUMMARY_PATH}")
    return eval_summary


if __name__ == "__main__":
    summary = run_spatial_graph_pipeline()
    print("Execution complete. Best model:", summary["best_model_by_balanced_accuracy"])
    for m, res in summary["models"].items():
        print(f"Model: {m.upper()} | Acc: {res['accuracy_mean']:.4f} +/- {res['accuracy_std']:.4f} | "
              f"BalAcc: {res['balanced_acc_mean']:.4f} +/- {res['balanced_acc_std']:.4f} | "
              f"MacroF1: {res['macro_f1_mean']:.4f} +/- {res['macro_f1_std']:.4f}")

