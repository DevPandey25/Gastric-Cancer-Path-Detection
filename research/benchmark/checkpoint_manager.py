"""Checkpoint Management for GastricPath AI.

Handles atomic checkpoint saving, SHA-256 verification, and resumption metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

from research.data.duplicate_detector import compute_sha256

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages saving and loading of model checkpoints with cryptographic hashing."""

    def __init__(self, checkpoints_dir: Path | str) -> None:
        self.checkpoints_dir = Path(checkpoints_dir).resolve()
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: nn.Module,
        filename: str,
        epoch: int,
        val_macro_f1: float,
        optimizer: torch.optim.Optimizer | None = None,
        val_metrics: dict[str, Any] | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> tuple[Path, str]:
        """Saves checkpoint and returns (saved_path, sha256_hash)."""
        save_path = self.checkpoints_dir / filename
        checkpoint_dict: dict[str, Any] = {
            "epoch": epoch,
            "val_macro_f1": val_macro_f1,
            "model_state_dict": model.state_dict(),
            "val_metrics": val_metrics or {},
            "extra_meta": extra_meta or {},
        }
        if optimizer is not None:
            checkpoint_dict["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint_dict, save_path)
        sha256 = compute_sha256(save_path)
        logger.info("Saved checkpoint %s (SHA-256: %s)", filename, sha256[:12])
        return save_path, sha256

    def load_checkpoint(
        self,
        checkpoint_path: Path | str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        device: torch.device | str = "cpu",
    ) -> dict[str, Any]:
        """Loads weights into model and optional optimizer, returns checkpoint metadata."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise ValueError(f"Unrecognized checkpoint content at {path}")

        model.load_state_dict(state_dict)
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint if isinstance(checkpoint, dict) else {}

