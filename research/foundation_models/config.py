"""Configuration schema for GastricPath AI Foundation Model Subsystem."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field


class FoundationModelConfig(BaseModel):
    """Configuration for foundation model feature extraction and embedding retrieval."""

    model_name: str = Field(
        default="dinov2_vits14",
        description="Encoder backbone architecture (e.g. dinov2_vits14, dinov2_vitb14, phikon)",
    )
    embedding_dim: int = Field(
        default=384,
        description="Expected embedding dimension of the encoder [CLS] representation",
    )
    batch_size: int = Field(
        default=16,
        description="Batch size for feature extraction (conservative for CPU)",
    )
    device: str = Field(
        default="cpu",
        description="Target compute device ('cpu' or 'cuda')",
    )
    normalize_l2: bool = Field(
        default=True,
        description="Whether to L2-normalize feature embeddings for cosine similarity indexing",
    )
    image_size: int = Field(
        default=224,
        description="Target square spatial resolution for input pathology tiles",
    )
    mean: tuple[float, float, float] = Field(
        default=(0.485, 0.456, 0.406),
        description="RGB channel mean for z-score normalization",
    )
    std: tuple[float, float, float] = Field(
        default=(0.229, 0.224, 0.225),
        description="RGB channel standard deviation for z-score normalization",
    )
    cache_dir: str = Field(
        default="research/foundation_models/cache",
        description="Directory for persisting embedding arrays and metadata manifests",
    )
    model_revision: str = Field(
        default="dinov2_vits14_pretrain.pth",
        description="Specific model checkpoint hash or revision string",
    )
    preprocessing_version: str = Field(
        default="v1.0-bilinear-224",
        description="Version string tracking the deterministic preprocessing pipeline",
    )

    def resolve_cache_dir(self, workspace_root: Path | str | None = None) -> Path:
        """Resolves absolute path to cache directory."""
        if workspace_root is None:
            workspace_root = Path(__file__).resolve().parent.parent.parent
        p = Path(workspace_root) / self.cache_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

