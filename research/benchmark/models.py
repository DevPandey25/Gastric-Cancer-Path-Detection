"""Model Architecture Factory for GastricPath AI Benchmarking.

Provides Model A (GastricCancerRNN), Model B (ResNet50), and Model C (EfficientNet-B0).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch import nn
import torchvision.models as tv_models

from app.ml.model import GastricCancerRNN


class ResNet50Classifier(nn.Module):
    """Standard ResNet-50 2D Convolutional Pathology Classifier."""

    def __init__(self, num_classes: int = 8, pretrained: bool = False) -> None:
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        self.backbone = tv_models.resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class EfficientNetB0Classifier(nn.Module):
    """Lightweight EfficientNet-B0 Classifier optimized for CPU pathology inference."""

    def __init__(self, num_classes: int = 8, pretrained: bool = False) -> None:
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        self.backbone = tv_models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class GastricCancerRNNWrapper(nn.Module):
    """Wrapper around GastricCancerRNN ensuring uniform logits return."""

    def __init__(
        self,
        num_classes: int = 8,
        feature_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.rnn_model = GastricCancerRNN(
            num_classes=num_classes,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.rnn_model(x)
        return logits

    def forward_with_attention(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rnn_model(x)


def create_benchmark_model(
    architecture: str,
    num_classes: int = 8,
    pretrained: bool = False,
    checkpoint_path: Path | str | None = None,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, Literal["2d", "sequence"]]:
    """Instantiates a benchmark model, returns (model, input_mode)."""
    arch_lower = architecture.lower().replace("-", "").replace("_", "")

    if arch_lower in ("gastriccancerrnn", "baselinegastricrnn", "rnn"):
        model: nn.Module = GastricCancerRNNWrapper(num_classes=num_classes)
        input_mode: Literal["2d", "sequence"] = "sequence"
    elif arch_lower in ("resnet50", "resnet"):
        model = ResNet50Classifier(num_classes=num_classes, pretrained=pretrained)
        input_mode = "2d"
    elif arch_lower in ("efficientnetb0", "efficientnet"):
        model = EfficientNetB0Classifier(num_classes=num_classes, pretrained=pretrained)
        input_mode = "2d"
    else:
        raise ValueError(
            f"Unsupported architecture: '{architecture}'. "
            f"Supported options: 'GastricCancerRNN', 'ResNet50', 'EfficientNet-B0'"
        )

    if checkpoint_path is not None:
        cp_path = Path(checkpoint_path)
        if not cp_path.exists():
            raise FileNotFoundError(f"Checkpoint file missing: {cp_path}")
        checkpoint = torch.load(cp_path, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise ValueError(f"Unrecognized checkpoint format at {cp_path}")

        # If wrapped, handle possible prefix differences
        if isinstance(model, GastricCancerRNNWrapper):
            try:
                model.rnn_model.load_state_dict(state_dict)
            except Exception:
                model.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)

    model.to(device)
    return model, input_mode

