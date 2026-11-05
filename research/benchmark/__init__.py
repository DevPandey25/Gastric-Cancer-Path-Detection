"""GastricPath AI - Benchmark and Classification Research Package."""

from research.benchmark.dataset import ManifestHistopathologyDataset, CANONICAL_CLASSES
from research.benchmark.dataloaders import create_benchmark_dataloaders
from research.benchmark.models import (
    create_benchmark_model,
    ResNet50Classifier,
    EfficientNetB0Classifier,
    GastricCancerRNNWrapper,
)
from research.benchmark.metrics import (
    compute_comprehensive_metrics,
    compute_expected_calibration_error,
    compute_multiclass_brier_score,
)
from research.benchmark.checkpoint_manager import CheckpointManager
from research.benchmark.evaluator import evaluate_model
from research.benchmark.trainer import BenchmarkTrainer
from research.benchmark.report_generator import generate_experiment_reports
from research.benchmark.benchmark_runner import (
    evaluate_baseline_model,
    run_smoke_experiment,
    estimate_training_resources,
)

__all__ = [
    "ManifestHistopathologyDataset",
    "CANONICAL_CLASSES",
    "create_benchmark_dataloaders",
    "create_benchmark_model",
    "ResNet50Classifier",
    "EfficientNetB0Classifier",
    "GastricCancerRNNWrapper",
    "compute_comprehensive_metrics",
    "compute_expected_calibration_error",
    "compute_multiclass_brier_score",
    "CheckpointManager",
    "evaluate_model",
    "BenchmarkTrainer",
    "generate_experiment_reports",
    "evaluate_baseline_model",
    "run_smoke_experiment",
    "estimate_training_resources",
]

