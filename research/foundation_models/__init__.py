"""GastricPath AI - Pathology Foundation Models Subsystem.

Provides frozen vision transformer backbones, deterministic pathology preprocessing,
chunked embedding extraction, cache persistence, and split-aware similarity retrieval.
"""

from research.foundation_models.config import FoundationModelConfig
from research.foundation_models.metadata import ImageEmbeddingRecord, RetrievalMatch

__all__ = [
    "FoundationModelConfig",
    "ImageEmbeddingRecord",
    "RetrievalMatch",
]

