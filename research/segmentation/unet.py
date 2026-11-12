"""Lightweight 2D U-Net Baseline Architecture for Histopathology Tissue Segmentation.

Follows the classic Ronneberger et al. (2015) contracting-expanding path design with
skip connections, BatchNorm, and configurable feature channels.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("tissue_segmentation.unet")


class DoubleConv(nn.Module):
    """Two sequential Conv2d -> BatchNorm2d -> ReLU layers."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """Configurable 2D U-Net for tissue segmentation on histopathology tiles."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 9,
        features: Optional[List[int]] = None,
    ) -> None:
        """
        Args:
            in_channels: Number of input image channels (3 for RGB).
            num_classes: Number of segmentation classes (9 for Background + 8 GCHTID classes).
            features: List of channel dimensions at each scale (default: [32, 64, 128, 256]).
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.features = features or [32, 64, 128, 256]

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (Contracting Path)
        curr_in = in_channels
        for feature in self.features:
            self.downs.append(DoubleConv(curr_in, feature))
            curr_in = feature

        # Bottleneck
        self.bottleneck = DoubleConv(self.features[-1], self.features[-1] * 2)

        # Decoder (Expanding Path)
        for feature in reversed(self.features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Final Classifier Head
        self.final_conv = nn.Conv2d(self.features[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw unnormalized logits of shape (B, num_classes, H, W)."""
        skip_connections = []

        # Downward contracting path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Upward expanding path
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Up-conv
            skip_connection = skip_connections[idx // 2]

            # Handle odd dimensions if any
            if x.shape != skip_connection.shape:
                x = F.interpolate(
                    x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True
                )

            concat_x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_x)  # DoubleConv

        return self.final_conv(x)

    def predict_mask(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs inference returning (predicted_classes, probabilities).
        Args:
            x: Input tensor of shape (B, C, H, W).
        Returns:
            predicted_mask: torch.LongTensor of shape (B, H, W)
            probabilities: torch.FloatTensor of shape (B, num_classes, H, W)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            if self.num_classes == 1:
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long().squeeze(1)
            else:
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
        return preds, probs

    def count_parameters(self) -> int:
        """Counts total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(
        self,
        filepath: Union[str, Path],
        epoch: Optional[int] = None,
        val_dice: Optional[float] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Saves model weights and training metadata."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.state_dict(),
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "features": self.features,
            "epoch": epoch,
            "val_dice": val_dice,
            "extra_metadata": extra_metadata or {},
        }
        torch.save(payload, p)
        logger.info("Saved U-Net checkpoint to %s", p)

    def load_checkpoint(
        self, filepath: Union[str, Path], map_location: str = "cpu"
    ) -> Dict[str, Any]:
        """Loads model weights from checkpoint."""
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        checkpoint = torch.load(p, map_location=map_location)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.eval()
        logger.info("Loaded U-Net checkpoint from %s", p)
        return checkpoint


def create_unet_baseline(
    in_channels: int = 3,
    num_classes: int = 9,
    features: Optional[List[int]] = None,
) -> UNet:
    """Factory function to instantiate the standard 2D U-Net baseline (9 classes)."""
    return UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        features=features or [32, 64, 128, 256],
    )

