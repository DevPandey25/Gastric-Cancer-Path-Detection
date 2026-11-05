"""Benchmark Runner and Experiment Orchestrator for GastricPath AI.

Coordinates evaluation of the historical baseline model, execution of CPU-safe
smoke tests, full comparative benchmark training, experiment tracking, and model registry updates.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn

from research.config import load_research_config
from research.reproducibility import get_system_provenance, set_seed
from research.tracking.experiment_tracker import ExperimentTracker
from research.registry.model_registry import ModelRegistry
from research.registry.model_metadata import ModelMetadata
from research.benchmark.dataloaders import create_benchmark_dataloaders
from research.benchmark.models import create_benchmark_model
from research.benchmark.evaluator import evaluate_model
from research.benchmark.trainer import BenchmarkTrainer
from research.benchmark.checkpoint_manager import CheckpointManager
from research.benchmark.report_generator import generate_experiment_reports
from research.data.duplicate_detector import compute_sha256


def estimate_training_resources(
    architecture: str,
    num_samples: int,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Estimates CPU runtime per epoch based on empirical FLOPs and parameter counts."""
    # Approximate empirical forward+backward time on modern multicore CPU per image
    rates = {
        "GastricCancerRNN": 0.045,   # 16 patches * CNN + BiLSTM
        "ResNet50": 0.035,           # 25.5M parameters
        "EfficientNet-B0": 0.015,    # 5.3M parameters
    }
    rate_per_img = rates.get(architecture, 0.030)
    seconds_per_epoch = num_samples * rate_per_img
    minutes_per_epoch = seconds_per_epoch / 60.0

    return {
        "architecture": architecture,
        "sample_count": num_samples,
        "batch_size": batch_size,
        "batches_per_epoch": (num_samples + batch_size - 1) // batch_size,
        "estimated_seconds_per_epoch": round(seconds_per_epoch, 1),
        "estimated_minutes_per_epoch": round(minutes_per_epoch, 2),
        "estimated_hours_for_5_epochs": round((seconds_per_epoch * 5) / 3600.0, 2),
    }


def evaluate_baseline_model(
    val_limit: int | None = None,
    test_limit: int | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Evaluates the original GastricCancerRNN checkpoint on verified validation and test splits."""
    print("=" * 80)
    print("GASTRICPATH AI — EVALUATION OF ORIGINAL BASELINE (GastricCancerRNN)")
    print("=" * 80)
    cfg = load_research_config()
    set_seed(cfg.environment.seed)
    provenance = get_system_provenance()

    tracker = ExperimentTracker(experiment_dir=cfg.paths.workspace_root / "research" / "experiments")
    run = tracker.start_experiment(
        name="GastricCancerRNN Baseline Evaluation",
        model_architecture="GastricCancerRNN",
        dataset_info={"name": cfg.dataset.name, "split": "test"},
        config=cfg.model_dump(mode="json"),
        seed=cfg.environment.seed,
    )
    exp_dir = run.run_dir

    train_m = cfg.paths.splits_dir / "manifest_train.csv"
    val_m = cfg.paths.splits_dir / "manifest_val.csv"
    test_m = cfg.paths.splits_dir / "manifest_test.csv"
    raw_root = cfg.paths.raw_dataset_dir

    print(f"Loading checkpoint: {cfg.baseline_model.checkpoint_file}")
    cp_sha = compute_sha256(cfg.baseline_model.checkpoint_file)
    print(f"Checkpoint SHA-256: {cp_sha}")

    model, input_mode = create_benchmark_model(
        architecture="GastricCancerRNN",
        num_classes=cfg.dataset.total_classes,
        checkpoint_path=cfg.baseline_model.checkpoint_file,
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters:   {param_count:,} (Mode: {input_mode})")

    _, val_loader, test_loader = create_benchmark_dataloaders(
        train_manifest=train_m,
        val_manifest=val_m,
        test_manifest=test_m,
        dataset_root=raw_root,
        mode=input_mode,
        batch_size=32,
        eval_limit=test_limit,
    )

    t0 = time.time()
    print("\nRunning evaluation over Test Split...")
    test_metrics, test_predictions = evaluate_model(
        model,
        test_loader,
        device="cpu",
        class_names=cfg.dataset.classes,
    )
    elapsed = round(time.time() - t0, 2)

    overall = test_metrics["overall"]
    print(f"Evaluation completed in {elapsed}s")
    print(f"Test Accuracy:         {overall['accuracy'] * 100:.2f}%")
    print(f"Test Balanced Accuracy:{overall['balanced_accuracy'] * 100:.2f}%")
    print(f"Test Macro-F1:         {overall['macro_f1'] * 100:.2f}%")
    print(f"Test Weighted-F1:      {overall['weighted_f1'] * 100:.2f}%")
    print(f"Cohen's Kappa:         {overall['cohen_kappa']:.4f}")
    print(f"Expected Calib Error:  {overall['expected_calibration_error']:.4f}")

    meta = {
        "experiment_id": run.run_id,
        "architecture": "GastricCancerRNN",
        "parameter_count": param_count,
        "input_mode": input_mode,
        "dataset_name": cfg.dataset.name,
        "dataset_split": "GCHTID Group-Stratified",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": elapsed,
        "git_commit": provenance.get("git_commit_hash", "unknown"),
        "checkpoint_sha256": cp_sha,
    }

    report_files = generate_experiment_reports(
        experiment_dir=exp_dir,
        metrics=test_metrics,
        predictions=test_predictions,
        experiment_metadata=meta,
        class_names=cfg.dataset.classes,
    )

    # Log metrics to experiment tracker
    run.log_metric("test_accuracy", overall["accuracy"])
    run.log_metric("test_macro_f1", overall["macro_f1"])
    run.log_metric("test_balanced_accuracy", overall["balanced_accuracy"])
    run.log_metric("test_brier_score", overall["brier_score"])
    run.log_metric("test_ece", overall["expected_calibration_error"])
    for f in report_files.values():
        if f.parent != run.artifacts_dir:
            run.log_artifact(f)
    tracker.complete_experiment(run, status="COMPLETED")

    # Register in local model registry
    registry = ModelRegistry(registry_dir=cfg.paths.registry_dir)
    meta_card = ModelMetadata(
        model_id=f"baseline_evaluated_{run.run_id}",
        name="GastricCancerRNN Baseline Verified",
        architecture="GastricCancerRNN (CNN-BiLSTM-Attention)",
        checkpoint_path=str(cfg.baseline_model.checkpoint_file),
        checkpoint_hash=cp_sha,
        dataset_info={
            "name": cfg.dataset.name,
            "total_classes": cfg.dataset.total_classes,
            "classes": cfg.dataset.classes,
            "test_samples": overall["total_samples"],
        },
        evaluation_metrics={
            "test_accuracy": overall["accuracy"],
            "test_macro_f1": overall["macro_f1"],
            "test_balanced_accuracy": overall["balanced_accuracy"],
            "cohen_kappa": overall["cohen_kappa"],
            "ece": overall["expected_calibration_error"],
        },
        class_labels=cfg.dataset.classes,
        status="baseline",
        git_commit=provenance.get("git_commit_hash", "unknown"),
        model_limitations=[
            "Evaluated strictly on image-level group-stratified splits.",
            "Does not output clinical tumor burden; probabilities represent model confidence.",
        ],
    )
    registry.register_model(meta_card)
    print(f"\nSaved all artifacts to: {exp_dir}")
    print(f"Registered in model registry: {meta_card.model_id}")
    return test_metrics, report_files


def run_smoke_experiment(architecture: str = "ResNet50") -> tuple[dict[str, Any], dict[str, Path]]:
    """Executes a resource-safe smoke test validating the complete training lifecycle."""
    print("=" * 80)
    print(f"GASTRICPATH AI — RESOURCE-AWARE SMOKE TEST ({architecture})")
    print("=" * 80)
    cfg = load_research_config()
    set_seed(cfg.environment.seed)
    provenance = get_system_provenance()

    tracker = ExperimentTracker(experiment_dir=cfg.paths.workspace_root / "research" / "experiments")
    run = tracker.start_experiment(
        name=f"Smoke Test {architecture}",
        model_architecture=architecture,
        dataset_info={"name": cfg.dataset.name, "split": "smoke_subset"},
        config=cfg.model_dump(mode="json"),
        seed=cfg.environment.seed,
    )
    exp_dir = run.run_dir

    train_m = cfg.paths.splits_dir / "manifest_train.csv"
    val_m = cfg.paths.splits_dir / "manifest_val.csv"
    test_m = cfg.paths.splits_dir / "manifest_test.csv"
    raw_root = cfg.paths.raw_dataset_dir

    model, input_mode = create_benchmark_model(
        architecture=architecture,
        num_classes=cfg.dataset.total_classes,
        pretrained=False,
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,} (Input Mode: {input_mode})")

    # Use a small subset (e.g. 24 train samples, 16 val, 16 test) for fast CPU-safe smoke verification
    train_loader, val_loader, test_loader = create_benchmark_dataloaders(
        train_manifest=train_m,
        val_manifest=val_m,
        test_manifest=test_m,
        dataset_root=raw_root,
        mode=input_mode,
        batch_size=8,
        train_limit=24,
        eval_limit=16,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    checkpoint_mgr = CheckpointManager(checkpoints_dir=exp_dir / "checkpoints")

    trainer = BenchmarkTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_manager=checkpoint_mgr,
        device="cpu",
        class_names=cfg.dataset.classes,
    )

    t0 = time.time()
    print("Training 1 smoke epoch...")
    best_val_metrics, history = trainer.train(epochs=1)
    elapsed = round(time.time() - t0, 2)
    print(f"Smoke training complete in {elapsed}s")

    # Evaluate on test split
    print("Evaluating on test split...")
    test_metrics, test_predictions = evaluate_model(
        model,
        test_loader,
        device="cpu",
        class_names=cfg.dataset.classes,
    )

    meta = {
        "experiment_id": run.run_id,
        "architecture": architecture,
        "parameter_count": param_count,
        "input_mode": input_mode,
        "dataset_name": cfg.dataset.name,
        "dataset_split": "Smoke Test Subset",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": elapsed,
        "git_commit": provenance.get("git_commit_hash", "unknown"),
    }

    report_files = generate_experiment_reports(
        experiment_dir=exp_dir,
        metrics=test_metrics,
        predictions=test_predictions,
        history=history,
        experiment_metadata=meta,
        class_names=cfg.dataset.classes,
    )

    run.log_metric("test_accuracy", test_metrics["overall"]["accuracy"])
    run.log_metric("test_macro_f1", test_metrics["overall"]["macro_f1"])
    for f in report_files.values():
        if f.parent != run.artifacts_dir:
            run.log_artifact(f)
    tracker.complete_experiment(run, status="COMPLETED")

    print(f"Smoke test successful! Artifacts written to: {exp_dir}")
    return test_metrics, report_files


def main():
    parser = argparse.ArgumentParser(description="GastricPath AI Benchmark Runner")
    parser.add_argument("command", choices=["eval_baseline", "smoke_test", "estimate_resources"], default="eval_baseline", nargs="?")
    parser.add_argument("--arch", type=str, default="ResNet50")
    parser.add_argument("--test-limit", type=int, default=None)
    args = parser.parse_args()

    if args.command == "eval_baseline":
        evaluate_baseline_model(test_limit=args.test_limit)
    elif args.command == "smoke_test":
        run_smoke_experiment(architecture=args.arch)
    elif args.command == "estimate_resources":
        cfg = load_research_config()
        est = estimate_training_resources(args.arch, num_samples=21767)
        print(f"Resource Estimate for {args.arch}:")
        for k, v in est.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

