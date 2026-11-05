from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal
import yaml
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ProjectConfig(BaseModel):
    name: str = "GastricPath AI"
    description: str = "Explainable, Spatially-Aware Computational Pathology Platform for Gastric Cancer"
    version: str = "1.0.0"


class EnvironmentConfig(BaseModel):
    seed: int = 42
    deterministic: bool = True
    device: Literal["auto", "cpu", "cuda"] = "auto"
    num_workers: int = 0
    pin_memory: bool = False
    mixed_precision: bool = False


class PathsConfig(BaseModel):
    workspace_root: Path = Field(default_factory=lambda: PROJECT_ROOT)
    raw_dataset_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "raw" / "HMU-GC-HE-30K" / "all_image")
    support_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "support")
    manifest_output_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "model_train" / "runs")
    experiment_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "research" / "experiments")
    registry_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "research" / "registry")
    checkpoint_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "models" / "checkpoints")
    test_artifacts_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "research" / "test_artifacts")
    splits_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "research" / "splits")
    reports_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "research" / "reports")


class SplitRatios(BaseModel):
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15


class DatasetConfig(BaseModel):
    name: str = "GCHTID"
    full_name: str = "Gastric Cancer Histopathology Tissue Image Dataset"
    total_classes: int = 8
    classes: list[str] = Field(default_factory=lambda: ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"])
    split_ratios: SplitRatios = Field(default_factory=SplitRatios)
    stratified: bool = True
    split_seed: int = 42


class PreprocessingConfig(BaseModel):
    image_size: int = 224
    channels: int = 3
    patch_size: int = 56
    sequence_length: int = 16
    normalization: str = "min_max_255"


class BaselineModelConfig(BaseModel):
    model_id: str = "baseline_gastric_rnn"
    architecture: str = "GastricCancerRNN"
    feature_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    checkpoint_file: Path = Field(default_factory=lambda: PROJECT_ROOT / "models" / "checkpoints" / "gastric_rnn_classifier.pt")


class EarlyStoppingConfig(BaseModel):
    enabled: bool = True
    patience: int = 3
    monitor: str = "val_accuracy"
    mode: Literal["min", "max"] = "max"


class TrainingConfig(BaseModel):
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.00001
    epochs: int = 5
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)


class UncertaintyConfig(BaseModel):
    calibration_bins: int = 10
    temperature_scaling: bool = False


class EvaluationConfig(BaseModel):
    metrics: list[str] = Field(default_factory=lambda: [
        "accuracy", "macro_f1", "balanced_accuracy", "per_class_precision", "per_class_recall", "confusion_matrix"
    ])
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)


class ResearchConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    baseline_model: BaselineModelConfig = Field(default_factory=BaselineModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    def resolve_paths(self) -> None:
        """Ensure all relative paths are resolved to absolute paths based on workspace_root."""
        root = self.paths.workspace_root
        if not root.is_absolute():
            root = (PROJECT_ROOT / root).resolve()
            self.paths.workspace_root = root

        for field_name in PathsConfig.model_fields:
            if field_name == "workspace_root":
                continue
            path_val = getattr(self.paths, field_name)
            if isinstance(path_val, Path) and not path_val.is_absolute():
                setattr(self.paths, field_name, (root / path_val).resolve())

        if not self.baseline_model.checkpoint_file.is_absolute():
            self.baseline_model.checkpoint_file = (root / self.baseline_model.checkpoint_file).resolve()


def _deep_update(mapping: dict[str, Any], updating_mapping: dict[str, Any]) -> dict[str, Any]:
    for key, val in updating_mapping.items():
        if isinstance(val, dict) and key in mapping and isinstance(mapping[key], dict):
            mapping[key] = _deep_update(mapping[key], val)
        else:
            mapping[key] = val
    return mapping


def load_research_config(
    config_path: Path | str | None = None,
    overrides: dict[str, Any] | None = None,
) -> ResearchConfig:
    """Load research configuration from YAML file with optional dictionary overrides."""
    if config_path is None:
        config_path = PROJECT_ROOT / "research" / "configs" / "base_research.yaml"
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = (PROJECT_ROOT / config_path).resolve()

    raw_dict: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                raw_dict = loaded

    if overrides:
        raw_dict = _deep_update(raw_dict, overrides)

    config = ResearchConfig.model_validate(raw_dict)
    config.resolve_paths()
    return config
