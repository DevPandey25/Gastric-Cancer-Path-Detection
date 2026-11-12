"""Deterministic Pathology Tile Preprocessing for Foundation Models.

Converts RGBA PNG tiles losslessly to RGB, resizes to target spatial resolution,
and applies channel z-score normalization for vision transformer backbones.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Union
import numpy as np
import torch
from PIL import Image

PathLike = Union[str, Path]


class DeterministicPathologyPreprocessor:
    """Preprocesses histopathology biopsy patches deterministically for frozen encoders."""

    def __init__(
        self,
        image_size: int = 224,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.image_size = image_size
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def preprocess_image(self, image: Image.Image | PathLike | bytes) -> torch.Tensor:
        """Preprocesses a single image into a normalized [3, H, W] tensor."""
        if isinstance(image, (str, Path)):
            with open(image, "rb") as f:
                img_bytes = f.read()
            with Image.open(io.BytesIO(img_bytes)) as raw_img:
                pil_img = raw_img.convert("RGB")
        elif isinstance(image, bytes):
            with Image.open(io.BytesIO(image)) as raw_img:
                pil_img = raw_img.convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image)}")

        # Deterministic resize to canonical spatial size
        if pil_img.size != (self.image_size, self.image_size):
            pil_img = pil_img.resize(
                (self.image_size, self.image_size),
                resample=Image.Resampling.BILINEAR,
            )

        # Convert to numpy array [H, W, C] in range [0, 1]
        arr = np.asarray(pil_img, dtype=np.float32) / 255.0
        # Transpose to [C, H, W] tensor
        tensor = torch.from_numpy(arr).permute(2, 0, 1)

        # Apply z-score normalization: (x - mean) / std
        normalized = (tensor - self.mean) / self.std
        return normalized

    def preprocess_batch(self, images: list[Image.Image | PathLike | bytes]) -> torch.Tensor:
        """Preprocesses a list of images into a batched [B, 3, H, W] tensor."""
        tensors = [self.preprocess_image(img) for img in images]
        return torch.stack(tensors, dim=0)

