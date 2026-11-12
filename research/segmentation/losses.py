"""Segmentation Loss Functions: Soft Dice Loss and Combined Dice-Cross-Entropy.

Optimizes overlap metrics directly while stabilizing optimization gradients with cross-entropy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Multi-class Soft Dice Loss."""

    def __init__(
        self,
        smooth: float = 1e-6,
        include_background: bool = True,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) unnormalized model outputs.
            targets: (B, H, W) ground-truth integer class labels in {0, ..., C-1}.
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # Convert targets to one-hot: (B, C, H, W)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        start_idx = 0 if self.include_background else 1
        dice_scores = []

        for c in range(start_idx, num_classes):
            p = probs[:, c, :, :].reshape(probs.shape[0], -1)
            t = targets_one_hot[:, c, :, :].reshape(targets_one_hot.shape[0], -1)

            intersection = torch.sum(p * t, dim=1)
            denominator = torch.sum(p ** 2, dim=1) + torch.sum(t ** 2, dim=1)

            dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
            dice_scores.append(dice)

        dice_tensor = torch.stack(dice_scores, dim=1)  # (B, num_evaluated_classes)
        loss = 1.0 - dice_tensor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CombinedDiceCrossEntropyLoss(nn.Module):
    """Weighted combination of Cross-Entropy and Multi-Class Dice Loss."""

    def __init__(
        self,
        weight_ce: float = 1.0,
        weight_dice: float = 1.0,
        class_weights: torch.Tensor | None = None,
        include_background_dice: bool = True,
    ) -> None:
        super().__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = DiceLoss(include_background=include_background_dice)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce_loss(logits, targets)
        loss_dice = self.dice_loss(logits, targets)
        return self.weight_ce * loss_ce + self.weight_dice * loss_dice

