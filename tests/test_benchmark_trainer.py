"""Tests for Benchmark Trainer, Evaluator, and Report Generator."""

from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from research.benchmark.checkpoint_manager import CheckpointManager
from research.benchmark.evaluator import evaluate_model
from research.benchmark.trainer import BenchmarkTrainer
from research.benchmark.report_generator import generate_experiment_reports


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 8)

    def forward(self, x):
        return self.fc(x)


def test_checkpoint_manager(tmp_path: Path):
    mgr = CheckpointManager(checkpoints_dir=tmp_path / "checkpoints")
    model = DummyModel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    save_path, sha = mgr.save_checkpoint(
        model=model,
        filename="test.pt",
        epoch=1,
        val_macro_f1=0.85,
        optimizer=opt,
    )
    assert save_path.exists()
    assert len(sha) == 64

    # Load back
    new_model = DummyModel()
    loaded_meta = mgr.load_checkpoint(save_path, new_model)
    assert loaded_meta["epoch"] == 1
    assert loaded_meta["val_macro_f1"] == 0.85


def test_trainer_and_evaluator_lifecycle(tmp_path: Path):
    model = DummyModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    chk_mgr = CheckpointManager(checkpoints_dir=tmp_path / "ckps")

    # Synthetic dataset
    X_train = torch.randn(20, 10)
    y_train = torch.randint(0, 8, (20,))
    train_paths = [f"path_{i}.png" for i in range(20)]

    X_val = torch.randn(8, 10)
    y_val = torch.randint(0, 8, (8,))
    val_paths = [f"val_path_{i}.png" for i in range(8)]

    # Custom wrapper to return (x, y, path)
    class TupleDataset(torch.utils.data.Dataset):
        def __init__(self, X, y, paths):
            self.X, self.y, self.paths = X, y, paths

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            return self.X[i], self.y[i], self.paths[i]

    train_loader = DataLoader(TupleDataset(X_train, y_train, train_paths), batch_size=4, shuffle=True)
    val_loader = DataLoader(TupleDataset(X_val, y_val, val_paths), batch_size=4, shuffle=False)

    trainer = BenchmarkTrainer(
        model=model,
        optimizer=opt,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_manager=chk_mgr,
        device="cpu",
    )

    best_val, history = trainer.train(epochs=2)
    assert len(history) == 2
    assert "macro_f1" in best_val["overall"]
    assert "val_macro_f1" in history[0]
    assert (tmp_path / "ckps" / "best_checkpoint.pt").exists()
    assert (tmp_path / "ckps" / "latest_checkpoint.pt").exists()

    # Evaluator test
    test_metrics, test_preds = evaluate_model(model, val_loader, device="cpu")
    assert "accuracy" in test_metrics["overall"]
    assert len(test_preds) == 8
    assert "ground_truth_label" in test_preds[0]
    assert "confidence" in test_preds[0]

    # Report Generator test
    report_files = generate_experiment_reports(
        experiment_dir=tmp_path / "exp_out",
        metrics=test_metrics,
        predictions=test_preds,
        history=history,
        experiment_metadata={"architecture": "DummyModel", "experiment_id": "test_exp"},
    )
    for key, fpath in report_files.items():
        assert fpath.exists(), f"Missing expected output: {key}"
        assert fpath.stat().st_size > 0

