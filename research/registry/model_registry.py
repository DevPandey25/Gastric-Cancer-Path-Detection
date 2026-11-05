from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from research.registry.model_metadata import ModelMetadata
from research.reproducibility import get_git_commit_hash


def compute_file_sha256(file_path: Path | str) -> str:
    """Computes SHA-256 hash of a file using 64KB chunked reads."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


class ModelRegistry:
    """Manages the registration, querying, and verification of computational pathology models."""

    def __init__(self, registry_dir: Path | str) -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_dir / "models.json"

    def _load_registry_raw(self) -> dict[str, Any]:
        if not self.registry_file.exists():
            return {}
        try:
            return json.loads(self.registry_file.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_registry_raw(self, data: dict[str, Any]) -> None:
        self.registry_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def register_model(self, metadata: ModelMetadata) -> ModelMetadata:
        """Registers a model metadata entry into the catalog."""
        data = self._load_registry_raw()
        data[metadata.model_id] = metadata.model_dump()
        self._save_registry_raw(data)
        return metadata

    def get_model(self, model_id: str) -> ModelMetadata | None:
        """Retrieves model metadata by model ID."""
        data = self._load_registry_raw()
        raw = data.get(model_id)
        if raw:
            return ModelMetadata.model_validate(raw)
        return None

    def list_models(self, status: str | None = None) -> list[ModelMetadata]:
        """Lists all registered models, optionally filtered by status."""
        data = self._load_registry_raw()
        models = [ModelMetadata.model_validate(val) for val in data.values()]
        if status:
            models = [m for m in models if m.status == status]
        return models

    def register_baseline_checkpoint(
        self,
        checkpoint_path: Path | str | None = None,
    ) -> ModelMetadata:
        """
        Registers the existing repository baseline model checkpoint (gastric_rnn_classifier.pt).
        Computes the actual SHA256 hash and includes true validated metrics.
        """
        if checkpoint_path is None:
            checkpoint_path = Path("models/checkpoints/gastric_rnn_classifier.pt")
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Baseline checkpoint not found at: {checkpoint_path}")

        file_hash = compute_file_sha256(checkpoint_path)

        metadata = ModelMetadata(
            model_id="baseline_gastric_rnn",
            name="GastricCancerRNN Baseline Classifier",
            architecture="GastricCancerRNN (CNN + BiLSTM + Additive Attention)",
            checkpoint_path=str(checkpoint_path),
            checkpoint_hash=file_hash,
            dataset_info={
                "name": "GCHTID",
                "total_classes": 8,
                "classes": ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"],
                "total_images": 31096,
                "split_level": "patch-level (patient IDs not available in filenames)",
            },
            training_config={
                "optimizer": "Adam",
                "loss_function": "CrossEntropyLoss",
                "patch_size": 56,
                "sequence_length": 16,
                "feature_dim": 128,
                "hidden_dim": 128,
                "num_layers": 2,
            },
            evaluation_metrics={
                "val_accuracy": 0.6486,
                "test_accuracy": 0.5682,
                "test_loss": 1.2732,
                "source": "Trained PyTorch Checkpoint & Notebook Benchmark",
            },
            class_labels=["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"],
            input_preprocessing={
                "image_size": 224,
                "patch_size": 56,
                "sequence_length": 16,
                "normalization": "min_max_255",
            },
            git_commit=get_git_commit_hash(),
            model_limitations=[
                "Trained on patch-level GCHTID images without patient-level clustering.",
                "Does not compute anatomical tumor area; class probabilities reflect model confidence.",
                "Intended solely for computational pathology research; not a medical diagnostic device.",
            ],
            status="baseline",
        )
        return self.register_model(metadata)
