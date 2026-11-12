from pathlib import Path
from research.registry.model_metadata import ModelMetadata
from research.registry.model_registry import ModelRegistry, compute_file_sha256


def test_sha256_computation(tmp_path: Path):
    test_file = tmp_path / "sample.bin"
    test_file.write_bytes(b"gastricpath_ai_test_data")
    h1 = compute_file_sha256(test_file)
    assert isinstance(h1, str)
    assert len(h1) == 64
    # Re-computing on same content yields identical hash
    h2 = compute_file_sha256(test_file)
    assert h1 == h2


def test_model_registry_crud(tmp_path: Path):
    registry = ModelRegistry(registry_dir=tmp_path / "registry")

    meta = ModelMetadata(
        model_id="test_model_v1",
        name="Test ResNet50",
        architecture="ResNet50",
        checkpoint_path="models/test.pt",
        checkpoint_hash="abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        dataset_info={"name": "GCHTID"},
        evaluation_metrics={"val_accuracy": 0.75},
        class_labels=["TUM", "NOR"],
        status="candidate",
    )

    registry.register_model(meta)

    # Retrieval
    retrieved = registry.get_model("test_model_v1")
    assert retrieved is not None
    assert retrieved.name == "Test ResNet50"
    assert retrieved.architecture == "ResNet50"
    assert retrieved.evaluation_metrics["val_accuracy"] == 0.75

    # List & filter
    all_models = registry.list_models()
    assert len(all_models) == 1

    candidate_models = registry.list_models(status="candidate")
    assert len(candidate_models) == 1

    baseline_models = registry.list_models(status="baseline")
    assert len(baseline_models) == 0


def test_register_actual_baseline_checkpoint(tmp_path: Path):
    registry = ModelRegistry(registry_dir=tmp_path / "registry")
    meta = registry.register_baseline_checkpoint(Path("models/checkpoints/gastric_rnn_classifier.pt"))
    assert meta.model_id == "baseline_gastric_rnn"
    assert meta.status == "baseline"
    assert len(meta.checkpoint_hash) == 64
    assert meta.evaluation_metrics["val_accuracy"] == 0.6486
    assert len(meta.class_labels) == 8

