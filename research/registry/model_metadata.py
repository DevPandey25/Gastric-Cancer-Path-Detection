from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, Field


class ModelMetadata(BaseModel):
    """
    Standardized computational pathology model metadata specification.
    Follows medical AI model-card guidelines.
    """
    model_id: str
    name: str
    architecture: str
    checkpoint_path: str
    checkpoint_hash: str
    dataset_info: dict[str, Any] = Field(default_factory=dict)
    training_config: dict[str, Any] = Field(default_factory=dict)
    evaluation_metrics: dict[str, Any] = Field(default_factory=dict)
    class_labels: list[str] = Field(default_factory=list)
    input_preprocessing: dict[str, Any] = Field(default_factory=dict)
    git_commit: str | None = None
    model_limitations: list[str] = Field(default_factory=list)
    status: Literal["baseline", "candidate", "production", "deprecated"] = "candidate"
    registered_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
