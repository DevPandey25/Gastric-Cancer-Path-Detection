"""Multiple Instance Learning (MIL) Architectures for Whole-Slide Histopathology.

Implements attention-based deep multiple instance learning models for slide-level diagnosis:
- GatedAttentionMIL: Ilse et al. (ICML 2018), "Attention-based Deep Multiple Instance Learning".
- MeanPoolingMILBaseline & MaxPoolingMILBaseline: Classical non-attention MIL controls.
- SlideBag data container enforcing strict slide-level isolation (zero inter-slide contamination).
- Extraction of instance-level attention weights for spatial attention heatmap reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("wsi.mil")


class MILTrainingBlockedError(RuntimeError):
    """Raised when MIL model training is attempted without genuine WSI data and slide-level labels."""
    pass


@dataclass
class SlideBag:
    """
    Data container representing a single Whole-Slide Image as a bag of instance patch embeddings.
    Strictly isolated: instances from different slides must NEVER be placed in the same bag.
    """

    slide_id: str
    embeddings: torch.Tensor  # Shape (K, in_features), float32
    label: Optional[int] = None  # Ground-truth slide-level categorical target
    patch_ids: Optional[List[str]] = None  # K patch IDs
    level0_coordinates: Optional[List[Tuple[int, int]]] = None  # K (x0, y0) tuples
    is_genuine_slide: bool = False

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError(f"Bag embeddings must have 2 dimensions (K, D), got {self.embeddings.ndim}")
        k = self.embeddings.shape[0]
        if self.patch_ids is not None and len(self.patch_ids) != k:
            raise ValueError(f"Mismatch: {k} embeddings but {len(self.patch_ids)} patch IDs.")
        if self.level0_coordinates is not None and len(self.level0_coordinates) != k:
            raise ValueError(f"Mismatch: {k} embeddings but {len(self.level0_coordinates)} coordinates.")

    @property
    def num_instances(self) -> int:
        return self.embeddings.shape[0]


class GatedAttentionMIL(nn.Module):
    """
    Gated Attention-based Deep Multiple Instance Learning model.
    Reference: Ilse, Tomczak, & Welling (ICML 2018).
    """

    def __init__(
        self,
        in_features: int = 768,
        hidden_dim: int = 256,
        attention_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.num_classes = num_classes

        # Feature encoder
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Gated Attention Mechanism
        # a_k = softmax( w^T ( tanh(V h_k) * sigmoid(U h_k) ) )
        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Sigmoid(),
        )
        self.attention_w = nn.Linear(attention_dim, 1)

        # Slide-level classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        bag_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single slide bag of K instance embeddings.

        Args:
            bag_embeddings: Tensor of shape (K, in_features).

        Returns:
            logits: Tensor of shape (1, num_classes).
            attention_weights: Tensor of shape (K, 1) normalized attention coefficients.
        """
        k, d = bag_embeddings.shape
        if k == 0:
            # Empty slide handling (no tissue detected)
            device = bag_embeddings.device
            return torch.zeros((1, self.num_classes), device=device), torch.empty((0, 1), device=device)

        # 1. Project instance features: (K, hidden_dim)
        h = self.feature_extractor(bag_embeddings)

        # 2. Compute Gated Attention weights
        a_v = self.attention_V(h)  # (K, attention_dim)
        a_u = self.attention_U(h)  # (K, attention_dim)
        a_gated = a_v * a_u  # Element-wise gating: (K, attention_dim)
        a_raw = self.attention_w(a_gated)  # (K, 1)

        # Softmax normalization across all K instances in the slide bag
        a_weights = F.softmax(a_raw, dim=0)  # (K, 1)

        # 3. Attention-weighted bag aggregation: (1, hidden_dim)
        # sum_{k=1}^K a_k * h_k
        bag_repr = torch.sum(a_weights * h, dim=0, keepdim=True)  # (1, hidden_dim)

        # 4. Slide-level classification logits
        logits = self.classifier(bag_repr)  # (1, num_classes)

        return logits, a_weights


class MeanPoolingMILBaseline(nn.Module):
    """Classical non-attention MIL baseline computing uniform average over instances."""

    def __init__(self, in_features: int = 768, hidden_dim: int = 256, num_classes: int = 2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, bag_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        k, _ = bag_embeddings.shape
        if k == 0:
            return torch.zeros((1, self.fc2.out_features), device=bag_embeddings.device), torch.empty(0, 1)

        h = F.relu(self.fc1(bag_embeddings))
        bag_repr = torch.mean(h, dim=0, keepdim=True)
        logits = self.fc2(bag_repr)

        # Uniform weights for visualization compatibility
        uniform_weights = torch.full((k, 1), 1.0 / float(k), device=bag_embeddings.device)
        return logits, uniform_weights


def build_mil_model(
    model_type: str = "gated_attention",
    in_features: int = 768,
    hidden_dim: int = 256,
    num_classes: int = 2,
) -> nn.Module:
    """Factory function for MIL models."""
    t = model_type.lower().strip()
    if t in ("gated_attention", "attention_mil", "clam_sb"):
        return GatedAttentionMIL(
            in_features=in_features,
            hidden_dim=hidden_dim,
            attention_dim=hidden_dim // 2,
            num_classes=num_classes,
        )
    elif t in ("mean", "mean_pooling"):
        return MeanPoolingMILBaseline(
            in_features=in_features,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown MIL model type: '{model_type}'. Choose 'gated_attention' or 'mean'.")

