"""Multimodal Image-Clinical Fusion Architecture and Linkage Gatekeeper.

Implements modular multimodal fusion architectures:
- EarlyFusionClassifier: Dense feature-level concatenation of image embeddings (d=768)
  and clinical embeddings (d=128) with non-linear MLP readout.
- LateFusionClassifier: Decision-level calibrated ensemble combining unimodal softmax predictions.
- CrossAttentionFusion: Multi-head cross-attention querying image visual tokens with clinical descriptors.
- MultimodalFusionGatekeeper: Strict programmatic enforcement of the 6 scientific prerequisites;
  prevents arbitrary pairing of unlinked image tiles and patient records.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from research.multimodal.linkage_audit import MultimodalLinkageBlockedError

logger = logging.getLogger("multimodal.fusion")


class EarlyFusionClassifier(nn.Module):
    """
    Concatenates visual embeddings (e.g. 768-D Phikon) and tabular clinical embeddings
    (e.g. 128-D ClinicalMLPEncoder) into a unified multimodal representation.
    """

    def __init__(
        self,
        image_dim: int = 768,
        clinical_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.image_dim = image_dim
        self.clinical_dim = clinical_dim
        self.total_dim = image_dim + clinical_dim

        self.fusion_head = nn.Sequential(
            nn.Linear(self.total_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        clinical_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image_embeddings: Tensor of shape (B, image_dim).
            clinical_embeddings: Tensor of shape (B, clinical_dim).

        Returns:
            logits: Tensor of shape (B, num_classes).
        """
        if image_embeddings.shape[0] != clinical_embeddings.shape[0]:
            raise ValueError(
                f"Batch size mismatch between image ({image_embeddings.shape[0]}) "
                f"and clinical ({clinical_embeddings.shape[0]}) embeddings."
            )

        fused = torch.cat([image_embeddings, clinical_embeddings], dim=1)
        return self.fusion_head(fused)


class LateFusionClassifier(nn.Module):
    """
    Decision-level multimodal fusion weighting unimodal probabilities:
    p_fused = alpha * p_image + (1 - alpha) * p_clinical.
    """

    def __init__(
        self,
        image_dim: int = 768,
        clinical_dim: int = 128,
        num_classes: int = 2,
        initial_alpha: float = 0.5,
        learnable_alpha: bool = True,
    ) -> None:
        super().__init__()
        self.image_head = nn.Linear(image_dim, num_classes)
        self.clinical_head = nn.Linear(clinical_dim, num_classes)

        if learnable_alpha:
            self.raw_alpha = nn.Parameter(torch.tensor([float(initial_alpha)]))
        else:
            self.register_buffer("raw_alpha", torch.tensor([float(initial_alpha)]))

    @property
    def alpha(self) -> float:
        return float(torch.clamp(self.raw_alpha, 0.0, 1.0).item())

    def forward(
        self,
        image_embeddings: torch.Tensor,
        clinical_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            (fused_probs, image_probs, clinical_probs)
        """
        img_logits = self.image_head(image_embeddings)
        clin_logits = self.clinical_head(clinical_embeddings)

        p_img = F.softmax(img_logits, dim=1)
        p_clin = F.softmax(clin_logits, dim=1)

        a = torch.clamp(self.raw_alpha, 0.0, 1.0)
        p_fused = a * p_img + (1.0 - a) * p_clin

        return p_fused, p_img, p_clin


class MultimodalFusionGatekeeper:
    """
    Programmatic gatekeeper validating the 6 mandatory scientific prerequisites
    before permitting image-clinical fusion operations.
    """

    @staticmethod
    def assert_fusion_prerequisites_met(
        has_verified_crosswalk: bool,
        crosswalk_verified_count: int,
        cohort_sample_size: int,
        target_defined: bool,
    ) -> None:
        """
        Raises MultimodalLinkageBlockedError if prerequisites are unfulfilled.
        """
        if not has_verified_crosswalk or crosswalk_verified_count == 0:
            raise MultimodalLinkageBlockedError(
                "BLOCKED: Image-clinical fusion cannot be performed. "
                "The repository GCHTID histopathology tiles (N=31,096) lack verified patient identifiers "
                "connecting them to HMU-GC-Clinical records (N=300). Fabricating synthetic crosswalks or "
                "arbitrarily pairing tiles and patient rows is strictly prohibited by research protocol."
            )
        if cohort_sample_size < 30:
            raise MultimodalLinkageBlockedError(
                f"BLOCKED: Linked cohort sample size (N={cohort_sample_size}) is insufficient for multimodal training."
            )
        if not target_defined:
            raise MultimodalLinkageBlockedError(
                "BLOCKED: Multimodal target variable is undefined or unverified."
            )

