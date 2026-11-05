from pathlib import Path
from research.tracking.experiment_tracker import ExperimentTracker


def test_experiment_tracker_workflow(tmp_path: Path):
    tracker = ExperimentTracker(experiment_dir=tmp_path / "experiments")
    
    # 1. Start experiment
    run = tracker.start_experiment(
        name="test_run_1",
        config={"lr": 0.001, "batch_size": 32},
        dataset_info={"name": "GCHTID", "classes": 8},
        model_architecture="GastricCancerRNN",
        seed=42,
    )
    assert run.run_id.startswith("exp_")
    assert run.status == "RUNNING"
    assert run.manifest_file.exists()

    # 2. Log metrics
    run.log_metric("train_loss", 0.85, step=1)
    run.log_metrics({"val_loss": 0.72, "val_accuracy": 0.68}, step=1)

    assert run.metrics_file.exists()
    assert run.metrics_summary["train_loss"] == 0.85
    assert run.metrics_summary["val_accuracy"] == 0.68

    # 3. Log artifact
    dummy_plot = tmp_path / "test_plot.png"
    dummy_plot.write_text("fake png content", encoding="utf-8")
    saved_artifact = run.log_artifact(dummy_plot, "confusion_matrix.png")
    assert saved_artifact.exists()
    assert len(run.artifacts) == 1

    # 4. Finish experiment
    tracker.finish_experiment(run, status="COMPLETED", summary={"final_score": 0.68})
    assert run.status == "COMPLETED"
    assert run.completed_at is not None

    # 5. Query index and manifest
    all_runs = tracker.list_experiments()
    assert len(all_runs) == 1
    assert all_runs[0]["run_id"] == run.run_id
    assert all_runs[0]["status"] == "COMPLETED"

    manifest = tracker.get_run_manifest(run.run_id)
    assert manifest is not None
    assert manifest["name"] == "test_run_1"
    assert manifest["seed"] == 42
    assert "provenance" in manifest

