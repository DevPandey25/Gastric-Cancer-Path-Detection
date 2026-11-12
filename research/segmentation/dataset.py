"""Mask-Aware Histopathology Segmentation Dataset and DataLoader.

Supports paired loading of 224x224 RGB tiles and pixel-level class masks,
synchronous geometric transformations, and graceful handling of missing annotations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger("tissue_segmentation.dataset")

# Standard pathology normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class HistopathologySegmentationDataset(Dataset):
    """PyTorch Dataset for paired histopathology image and segmentation mask loading."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        image_size: Tuple[int, int] = (224, 224),
        transform: Optional[Callable] = None,
        normalize: bool = True,
    ) -> None:
        """
        Args:
            samples: List of dictionaries with keys:
                     'image_path': Path to the RGB image.
                     'mask_path': Optional Path to the ground-truth mask (or None).
                     'image_id': Optional string identifier.
                     'class_name': Optional tile-level class.
            image_size: Target (height, width) tuple.
            transform: Optional custom augmentation callable (image, mask) -> (image, mask).
            normalize: Whether to apply standard ImageNet mean/std normalization.
        """
        self.samples = samples
        self.height, self.width = image_size
        self.transform = transform
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image_path = Path(sample["image_path"])

        if not image_path.exists():
            raise FileNotFoundError(f"Histopathology image not found: {image_path}")

        # Load RGB Image
        pil_img = Image.open(image_path).convert("RGB")
        if pil_img.size != (self.width, self.height):
            pil_img = pil_img.resize((self.width, self.height), Image.Resampling.BILINEAR)
        img_np = np.array(pil_img, dtype=np.float32) / 255.0  # (H, W, 3)

        # Load Mask if available
        mask_path = sample.get("mask_path")
        has_mask = False
        mask_np = np.zeros((self.height, self.width), dtype=np.int64)

        if mask_path is not None:
            m_p = Path(mask_path)
            if m_p.exists():
                pil_mask = Image.open(m_p)
                if pil_mask.size != (self.width, self.height):
                    pil_mask = pil_mask.resize((self.width, self.height), Image.Resampling.NEAREST)
                mask_np = np.array(pil_mask, dtype=np.int64)
                has_mask = True

        # Synchronous transforms if provided
        if self.transform is not None:
            img_np, mask_np = self.transform(img_np, mask_np)

        # Normalization
        if self.normalize:
            img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD

        # Convert to PyTorch tensors
        # Image: (C, H, W) float32
        tensor_img = torch.from_numpy(img_np).permute(2, 0, 1).contiguous().float()
        # Mask: (H, W) long
        tensor_mask = torch.from_numpy(mask_np).contiguous().long()

        return {
            "image": tensor_img,
            "mask": tensor_mask,
            "has_mask": has_mask,
            "image_id": sample.get("image_id", image_path.name),
            "image_path": str(image_path),
            "mask_path": str(mask_path) if mask_path else "",
        }

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Union[str, Path],
        dataset_root: Union[str, Path],
        masks_dir: Optional[Union[str, Path]] = None,
        split_filter: Optional[str] = None,
        image_size: Tuple[int, int] = (224, 224),
    ) -> HistopathologySegmentationDataset:
        """Constructs dataset from a CSV manifest, checking for paired mask files."""
        m_path = Path(manifest_path)
        if not m_path.exists():
            raise FileNotFoundError(f"Manifest not found: {m_path}")

        df = pd.read_csv(m_path)
        if split_filter and "split" in df.columns:
            df = df[df["split"] == split_filter].reset_index(drop=True)

        d_root = Path(dataset_root)
        m_dir = Path(masks_dir) if masks_dir else None

        samples = []
        for _, row in df.iterrows():
            rel_p = row["relative_path"]
            img_p = d_root / rel_p

            mask_p = None
            if m_dir:
                # Check for matching mask file (e.g. ADI_1118_mask.png or ADI/ADI_1118.png)
                candidate = m_dir / rel_p
                if candidate.exists():
                    mask_p = candidate
                else:
                    alt_candidate = m_dir / (Path(rel_p).stem + "_mask.png")
                    if alt_candidate.exists():
                        mask_p = alt_candidate

            samples.append({
                "image_path": img_p,
                "mask_path": mask_p,
                "image_id": row.get("image_id", Path(rel_p).name),
                "class_name": row.get("class_name", ""),
            })

        logger.info(
            "Constructed segmentation dataset with %d samples (masks found: %d)",
            len(samples),
            sum(1 for s in samples if s["mask_path"] is not None),
        )
        return cls(samples, image_size=image_size)


def create_segmentation_dataloader(
    dataset: HistopathologySegmentationDataset,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Builds a standard DataLoader for segmentation."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )

