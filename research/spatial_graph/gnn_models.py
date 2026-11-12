"""Graph Neural Network Architectures and Non-Graph Baseline for Histopathology.

Implements lightweight spatial graph classifiers and a non-graph morphological baseline:
1. GCNClassifier: Graph Convolutional Network (Kipf & Welling, 2017)
2. GATClassifier: Graph Attention Network (Veličković et al., 2018)
3. GraphSAGEClassifier: Inductive Graph Representation Learning (Hamilton et al., 2017)
4. MorphologyMLPBaseline: Non-graph multi-layer perceptron over aggregated morphology

Architectures operate on candidate-object graphs with 16 continuous node features and
weak graph-level tissue class supervision across the 8 canonical GCHTID categories.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Batch as PyGBatch, Data as PyGData
    from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_mean_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from research.spatial_graph.graph_features import NODE_FEATURE_NAMES


def count_parameters(model: nn.Module) -> int:
    """Returns the total number of trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class GCNClassifier(nn.Module):
    """
    2-Layer Graph Convolutional Network (GCN) for graph-level histopathology classification.
    Reference: Kipf & Welling (ICLR 2017), "Semi-Supervised Classification with Graph Convolutional Networks".
    """

    def __init__(
        self,
        in_channels: int = 16,
        hidden_channels: int = 32,
        num_classes: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not HAS_PYG:
            raise ImportError("torch_geometric is required for GCNClassifier.")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.dropout = dropout

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(
        self,
        x_or_data: Any,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass accepting either a PyG Data/Batch object or separate tensors.
        """
        if hasattr(x_or_data, "x"):
            x = x_or_data.x
            edge_index = x_or_data.edge_index
            batch = getattr(x_or_data, "batch", None)
            batch_size = getattr(x_or_data, "num_graphs", 1)
        else:
            x = x_or_data
            batch_size = 1 if batch is None else int(batch.max().item() + 1)

        # Handle empty graph (no detected candidate objects)
        if x.shape[0] == 0:
            device = x.device
            return torch.zeros((batch_size, self.num_classes), device=device, dtype=torch.float32)

        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        # Layer 1
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2
        h = self.conv2(h, edge_index)
        h = F.relu(h)

        # Global graph readout
        pooled = global_mean_pool(h, batch, size=batch_size)

        # Classification logits
        out = self.fc(pooled)
        return out


class GATClassifier(nn.Module):
    """
    2-Layer Graph Attention Network (GAT) for graph-level histopathology classification.
    Reference: Veličković et al. (ICLR 2018), "Graph Attention Networks".
    """

    def __init__(
        self,
        in_channels: int = 16,
        hidden_channels: int = 32,
        num_classes: int = 8,
        heads: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not HAS_PYG:
            raise ImportError("torch_geometric is required for GATClassifier.")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.heads = heads
        self.dropout = dropout

        out_head_dim = hidden_channels // heads
        self.conv1 = GATConv(
            in_channels,
            out_head_dim,
            heads=heads,
            concat=True,
            dropout=dropout,
        )
        self.conv2 = GATConv(
            out_head_dim * heads,
            hidden_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(
        self,
        x_or_data: Any,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hasattr(x_or_data, "x"):
            x = x_or_data.x
            edge_index = x_or_data.edge_index
            batch = getattr(x_or_data, "batch", None)
            batch_size = getattr(x_or_data, "num_graphs", 1)
        else:
            x = x_or_data
            batch_size = 1 if batch is None else int(batch.max().item() + 1)

        if x.shape[0] == 0:
            device = x.device
            return torch.zeros((batch_size, self.num_classes), device=device, dtype=torch.float32)

        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        h = self.conv1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index)
        h = F.elu(h)

        pooled = global_mean_pool(h, batch, size=batch_size)
        out = self.fc(pooled)
        return out


class GraphSAGEClassifier(nn.Module):
    """
    2-Layer GraphSAGE Network with mean aggregation for graph-level histopathology classification.
    Reference: Hamilton et al. (NeurIPS 2017), "Inductive Representation Learning on Large Graphs".
    """

    def __init__(
        self,
        in_channels: int = 16,
        hidden_channels: int = 32,
        num_classes: int = 8,
        aggr: str = "mean",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not HAS_PYG:
            raise ImportError("torch_geometric is required for GraphSAGEClassifier.")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.dropout = dropout

        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr=aggr)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr=aggr)
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(
        self,
        x_or_data: Any,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hasattr(x_or_data, "x"):
            x = x_or_data.x
            edge_index = x_or_data.edge_index
            batch = getattr(x_or_data, "batch", None)
            batch_size = getattr(x_or_data, "num_graphs", 1)
        else:
            x = x_or_data
            batch_size = 1 if batch is None else int(batch.max().item() + 1)

        if x.shape[0] == 0:
            device = x.device
            return torch.zeros((batch_size, self.num_classes), device=device, dtype=torch.float32)

        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index)
        h = F.relu(h)

        pooled = global_mean_pool(h, batch, size=batch_size)
        out = self.fc(pooled)
        return out


class MorphologyMLPBaseline(nn.Module):
    """
    Non-graph baseline operating on image-level aggregated morphological descriptors
    (mean, std, and median across candidate objects, 16 * 3 = 48 continuous features).
    """

    def __init__(
        self,
        in_dim: int = 48,
        hidden_dim: int = 64,
        num_classes: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass accepting aggregated feature matrix of shape (B, 48).
        """
        return self.mlp(x)


def build_model(
    model_name: str,
    in_channels: int = 16,
    hidden_channels: int = 32,
    num_classes: int = 8,
    dropout: float = 0.2,
) -> nn.Module:
    """
    Factory function instantiating a GNN model or non-graph baseline.

    Args:
        model_name: One of 'gcn', 'gat', 'sage', 'mlp'.
        in_channels: Number of node features (16).
        hidden_channels: Hidden representation dimension (32).
        num_classes: Number of target classes (8).
        dropout: Dropout probability (0.2).

    Returns:
        Instantiated nn.Module.
    """
    m = model_name.lower().strip()
    if m == "gcn":
        return GCNClassifier(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            dropout=dropout,
        )
    elif m == "gat":
        return GATClassifier(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            heads=2,
            dropout=dropout,
        )
    elif m in ("sage", "graphsage"):
        return GraphSAGEClassifier(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            dropout=dropout,
        )
    elif m in ("mlp", "morphology_mlp", "baseline"):
        return MorphologyMLPBaseline(
            in_dim=in_channels * 3,  # mean, std, median = 48
            hidden_dim=hidden_channels * 2,
            num_classes=num_classes,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model name: '{model_name}'. Choose from 'gcn', 'gat', 'sage', 'mlp'.")

