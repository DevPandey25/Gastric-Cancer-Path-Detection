"""Data models and metadata schemas for Foundation Model Embeddings and Retrieval."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class ImageEmbeddingRecord:
    """Metadata tracking the provenance of a single extracted patch embedding."""

    image_id: str
    relative_path: str
    sha256: str
    class_name: str
    dataset_split: str
    model_name: str
    model_revision: str
    preprocessing_version: str
    embedding_dim: int
    extracted_at_utc: str
    l2_normalized: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImageEmbeddingRecord:
        return cls(
            image_id=str(data["image_id"]),
            relative_path=str(data["relative_path"]),
            sha256=str(data["sha256"]),
            class_name=str(data["class_name"]),
            dataset_split=str(data["dataset_split"]),
            model_name=str(data["model_name"]),
            model_revision=str(data.get("model_revision", "unknown")),
            preprocessing_version=str(data.get("preprocessing_version", "v1.0")),
            embedding_dim=int(data["embedding_dim"]),
            extracted_at_utc=str(data.get("extracted_at_utc", datetime.now(timezone.utc).isoformat())),
            l2_normalized=bool(data.get("l2_normalized", True)),
        )


@dataclass
class RetrievalMatch:
    """Represents a single nearest-neighbor retrieval match from embedding space.

    DISCLOSURE: Similarity reflects distance in representation space, not clinical certainty.
    """

    query_image_id: str
    neighbor_image_id: str
    relative_path: str
    sha256: str
    class_name: str
    dataset_split: str
    similarity_score: float
    rank: int
    is_same_class: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

