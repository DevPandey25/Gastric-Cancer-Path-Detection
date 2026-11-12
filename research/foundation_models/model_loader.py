"""Foundation Model Architecture Loader & Model Factory.

Instantiates frozen visual encoders for histopathology feature extraction.
Supports DINOv2 (ViT-S/14, ViT-B/14) and Phikon adapter interfaces with
strict weight freezing and device placement.
"""

from __future__ import annotations

import logging
from typing import Literal
import torch
from torch import nn

logger = logging.getLogger(__name__)

SUPPORTED_FOUNDATION_MODELS = {
    "dinov2_vits14": {
        "architecture": "Vision Transformer Small (ViT-S/14)",
        "embedding_dim": 384,
        "parameters": "22.05M",
        "patch_size": 14,
        "license": "Apache 2.0",
        "source": "torch.hub: facebookresearch/dinov2",
    },
    "dinov2_vitb14": {
        "architecture": "Vision Transformer Base (ViT-B/14)",
        "embedding_dim": 768,
        "parameters": "86.6M",
        "patch_size": 14,
        "license": "Apache 2.0",
        "source": "torch.hub: facebookresearch/dinov2",
    },
    "phikon": {
        "architecture": "Vision Transformer Base (iBOT ViT-B/16)",
        "embedding_dim": 768,
        "parameters": "85.8M",
        "patch_size": 16,
        "license": "Owkin Non-Commercial License",
        "source": "Hugging Face: owkin/phikon",
    },
}


class FrozenEncoderWrapper(nn.Module):
    """Wrapper ensuring frozen inference and uniform forward output for vision encoders."""

    def __init__(self, encoder: nn.Module, embedding_dim: int, model_name: str) -> None:
        super().__init__()
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.model_name = model_name

        # Enforce parameter freeze
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts [B, embedding_dim] representation from input batch."""
        with torch.no_grad():
            output = self.encoder(x)
            # If model returns an object with last_hidden_state or pooler_output (Hugging Face style)
            if hasattr(output, "last_hidden_state"):
                # Take [CLS] token at index 0
                cls_token = output.last_hidden_state[:, 0]
                return cls_token
            elif isinstance(output, torch.Tensor):
                return output
            elif isinstance(output, dict) and "x_norm_clstoken" in output:
                return output["x_norm_clstoken"]
            else:
                raise TypeError(f"Unrecognized encoder output format: {type(output)}")


def load_foundation_encoder(
    model_name: str = "dinov2_vits14",
    device: str = "cpu",
    pretrained: bool = True,
) -> tuple[nn.Module, int]:
    """Loads a frozen foundation encoder, returns (model, embedding_dim).

    Args:
        model_name: Name of model ('dinov2_vits14', 'dinov2_vitb14', 'phikon')
        device: Device to place the model on ('cpu' or 'cuda')
        pretrained: Whether to load official pretrained weights
    """
    model_name_clean = model_name.lower().replace("-", "_")

    if model_name_clean in ("dinov2_vits14", "dinov2_small"):
        logger.info("Loading DINOv2 ViT-S/14 via torch.hub on %s...", device)
        # Use torch.hub with local cache fallback
        encoder = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vits14",
            pretrained=pretrained,
        )
        embedding_dim = 384
        wrapped_model = FrozenEncoderWrapper(encoder, embedding_dim, "dinov2_vits14")

    elif model_name_clean in ("dinov2_vitb14", "dinov2_base"):
        logger.info("Loading DINOv2 ViT-B/14 via torch.hub on %s...", device)
        encoder = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitb14",
            pretrained=pretrained,
        )
        embedding_dim = 768
        wrapped_model = FrozenEncoderWrapper(encoder, embedding_dim, "dinov2_vitb14")

    elif model_name_clean == "phikon":
        logger.info("Loading Phikon (owkin/phikon) via transformers...")
        try:
            from transformers import AutoModel
            encoder = AutoModel.from_pretrained("owkin/phikon")
            embedding_dim = 768
            wrapped_model = FrozenEncoderWrapper(encoder, embedding_dim, "phikon")
        except ImportError as exc:
            raise ImportError(
                "Phikon requires the 'transformers' library. Install via: pip install transformers"
            ) from exc

    else:
        supported = list(SUPPORTED_FOUNDATION_MODELS.keys())
        raise ValueError(f"Unsupported foundation model '{model_name}'. Supported models: {supported}")

    wrapped_model.to(device)
    wrapped_model.eval()
    return wrapped_model, embedding_dim

