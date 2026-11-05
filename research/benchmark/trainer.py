"""Training Engine for GastricPath AI Classification Benchmarks.

Provides epoch-level training, validation monitoring, early stopping,
AdamW optimization, learning-rate scheduling, and automated checkpointing.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from research.benchmark.checkpoint_manager import CheckpointManager
from research.benchmark.evaluator import evaluate_model

logger = logging.getLogger(__name__)


class BenchmarkTrainer:
    """Configurable trainer for histopathology classification models."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_manager: CheckpointManager,
        device: torch.device | str = "cpu",
        scheduler: Any | None = None,
        early_stopping_patience: int = 3,
        monitor: str = "val_macro_f1",
        class_names: list[str] | None = None,
        epoch_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_manager = checkpoint_manager
        self.device = device
        self.scheduler = scheduler
        self.patience = early_stopping_patience
        self.monitor = monitor
        self.class_names = class_names
        self.epoch_callback = epoch_callback

    def train(self, epochs: int = 5) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Runs the training and validation loop for the specified number of epochs."""
        self.model.to(self.device)
        history: list[dict[str, Any]] = []

        best_score = -1.0
        best_val_metrics: dict[str, Any] = {}
        best_epoch = 0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            t_epoch_start = time.time()
            self.model.train()

            running_loss = 0.0
            total_train_samples = 0
            correct_train = 0

            for batch_idx, batch in enumerate(self.train_loader):
                inputs, targets, _ = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)
                loss.backward()

                # Gradient clipping to prevent exploding gradients in recurrent backbones
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                total_train_samples += batch_size
                preds = logits.argmax(dim=1)
                correct_train += int((preds == targets).sum().item())

            train_loss = running_loss / max(total_train_samples, 1)
            train_acc = correct_train / max(total_train_samples, 1)

            if self.scheduler is not None:
                self.scheduler.step()

            # Validation evaluation
            val_metrics, _ = evaluate_model(
                self.model,
                self.val_loader,
                device=self.device,
                class_names=self.class_names,
            )
            val_loss = val_metrics["overall"]["brier_score"]  # proxy/loss
            val_acc = val_metrics["overall"]["accuracy"]
            val_macro_f1 = val_metrics["overall"]["macro_f1"]

            epoch_duration = round(time.time() - t_epoch_start, 2)

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_accuracy": round(train_acc, 4),
                "val_accuracy": val_acc,
                "val_macro_f1": val_macro_f1,
                "val_balanced_accuracy": val_metrics["overall"]["balanced_accuracy"],
                "duration_seconds": epoch_duration,
            }
            history.append(epoch_record)

            if self.epoch_callback:
                self.epoch_callback(epoch_record)

            logger.info(
                "Epoch %d/%d — Train Loss: %.4f, Train Acc: %.4f | Val Acc: %.4f, Val Macro-F1: %.4f (%ss)",
                epoch,
                epochs,
                train_loss,
                train_acc,
                val_acc,
                val_macro_f1,
                epoch_duration,
            )

            # Check if this is the best checkpoint based on validation monitor (strictly never test set)
            current_score = val_macro_f1 if self.monitor == "val_macro_f1" else val_acc

            if current_score > best_score:
                best_score = current_score
                best_val_metrics = val_metrics
                best_epoch = epoch
                patience_counter = 0

                # Save best checkpoint
                self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    filename="best_checkpoint.pt",
                    epoch=epoch,
                    val_macro_f1=val_macro_f1,
                    optimizer=self.optimizer,
                    val_metrics=val_metrics,
                )
            else:
                patience_counter += 1

            # Always save latest checkpoint for crash recovery
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                filename="latest_checkpoint.pt",
                epoch=epoch,
                val_macro_f1=val_macro_f1,
                optimizer=self.optimizer,
                val_metrics=val_metrics,
            )

            # Early stopping check
            if patience_counter >= self.patience:
                logger.info(
                    "Early stopping triggered at epoch %d (patience of %d reached without improvement).",
                    epoch,
                    self.patience,
                )
                break

        return best_val_metrics, history

