"""Deterministic DataLoader Factory for GastricPath AI.

Constructs seeded, memory-conscious DataLoaders for Training, Validation, and Testing.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from research.benchmark.dataset import ManifestHistopathologyDataset


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_benchmark_dataloaders(
    train_manifest: Path | str,
    val_manifest: Path | str,
    test_manifest: Path | str,
    dataset_root: Path | str,
    mode: Literal["2d", "sequence"] = "2d",
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    train_limit: int | None = None,
    eval_limit: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Creates reproducible PyTorch DataLoaders for Train, Val, and Test splits."""
    g = torch.Generator()
    g.manual_seed(seed)

    train_ds = ManifestHistopathologyDataset(
        manifest_path=train_manifest,
        dataset_root=dataset_root,
        mode=mode,
        is_train=True,
        limit=train_limit,
    )
    val_ds = ManifestHistopathologyDataset(
        manifest_path=val_manifest,
        dataset_root=dataset_root,
        mode=mode,
        is_train=False,
        limit=eval_limit,
    )
    test_ds = ManifestHistopathologyDataset(
        manifest_path=test_manifest,
        dataset_root=dataset_root,
        mode=mode,
        is_train=False,
        limit=eval_limit,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=g,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader

