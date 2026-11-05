from pathlib import Path
from research.config import ResearchConfig, load_research_config


def test_default_config_loading():
    config = load_research_config()
    assert isinstance(config, ResearchConfig)
    assert config.project.name == "GastricPath AI"
    assert config.dataset.total_classes == 8
    assert "TUM" in config.dataset.classes
    assert config.preprocessing.image_size == 224
    assert config.preprocessing.patch_size == 56
    assert config.preprocessing.sequence_length == 16
    assert config.paths.workspace_root.is_absolute()
    assert config.baseline_model.architecture == "GastricCancerRNN"


def test_config_overrides():
    overrides = {
        "training": {
            "batch_size": 16,
            "epochs": 10,
        },
        "environment": {
            "seed": 100,
        },
    }
    config = load_research_config(overrides=overrides)
    assert config.training.batch_size == 16
    assert config.training.epochs == 10
    assert config.environment.seed == 100
    # Untouched parameters retain defaults
    assert config.dataset.total_classes == 8


def test_config_path_resolution(tmp_path: Path):
    custom_yaml = tmp_path / "custom.yaml"
    custom_yaml.write_text(
        """
project:
  name: "Custom Project"
paths:
  workspace_root: "."
  experiment_dir: "custom_experiments"
""",
        encoding="utf-8",
    )
    config = load_research_config(config_path=custom_yaml)
    assert config.project.name == "Custom Project"
    assert config.paths.experiment_dir.is_absolute()
