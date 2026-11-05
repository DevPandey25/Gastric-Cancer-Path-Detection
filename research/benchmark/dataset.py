"""Manifest-based Histopathology Dataset for GastricPath AI.

Loads images strictly from verified split manifests, performs lossless RGBA to RGB conversion,
supports 2D ConvNet and 16-patch sequence representations, and enforces deterministic
validation/test evaluation and conservative pathology-preserving training augmentation.
"""

from __future__ import annotations

import csv
import io
import random
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from app.ml.preprocessing import image_to_patch_sequence

CANONICAL_CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CANONICAL_CLASSES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CANONICAL_CLASSES)}


class PathologyAugmentation:
    """Conservative pathology-preserving augmentations.

    Applies spatial symmetries (H-flip, V-flip, 90-degree rotations) that are invariant
    in histopathology biopsy slides, and very subtle stain color jitter.
    """

    def __init__(
        self,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rot90_prob: float = 0.5,
        color_jitter: bool = True,
    ) -> None:
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rot90_prob = rot90_prob
        self.color_jitter = color_jitter

    def __call__(self, img: Image.Image) -> Image.Image:
        # Spatial symmetries (histology sections have no canonical gravitational up/down)
        if random.random() < self.hflip_prob:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < self.vflip_prob:
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if random.random() < self.rot90_prob:
            rot_choice = random.choice([Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270])
            img = img.transpose(rot_choice)

        # Subtle stain variation (simulates minor H&E bath differences without altering cellular boundaries)
        if self.color_jitter:
            if random.random() < 0.3:
                enh_b = ImageEnhance.Brightness(img)
                img = enh_b.enhance(random.uniform(0.96, 1.04))
            if random.random() < 0.3:
                enh_c = ImageEnhance.Contrast(img)
                img = enh_c.enhance(random.uniform(0.96, 1.04))
            if random.random() < 0.3:
                enh_s = ImageEnhance.Color(img)
                img = enh_s.enhance(random.uniform(0.96, 1.04))

        return img


class ManifestHistopathologyDataset(Dataset):
    """Manifest-backed PyTorch dataset supporting both 2D ConvNets and Sequence models."""

    def __init__(
        self,
        manifest_path: Path | str,
        dataset_root: Path | str,
        mode: Literal["2d", "sequence"] = "2d",
        is_train: bool = False,
        image_size: int = 224,
        patch_size: int = 56,
        sequence_length: int = 16,
        limit: int | None = None,
        custom_transform: Callable[[Image.Image], Image.Image] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.dataset_root = Path(dataset_root).resolve()
        self.mode = mode
        self.is_train = is_train
        self.image_size = image_size
        self.patch_size = patch_size
        self.sequence_length = sequence_length

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")

        # Read manifest rows
        self.rows: list[dict[str, str]] = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append(row)

        if limit is not None and limit > 0:
            by_class: dict[str, list[dict[str, str]]] = {}
            for r in self.rows:
                c = r.get("class_name", "UNKNOWN")
                if c not in by_class:
                    by_class[c] = []
                by_class[c].append(r)
            per_class_limit = max(1, (limit + len(by_class) - 1) // max(1, len(by_class)))
            sampled_rows: list[dict[str, str]] = []
            for c_list in by_class.values():
                sampled_rows.extend(c_list[:per_class_limit])
            self.rows = sampled_rows[:limit]

        # Augmentation is strictly disabled for validation and test splits
        if self.is_train:
            self.augmentor = custom_transform if custom_transform else PathologyAugmentation()
        else:
            self.augmentor = None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.rows[index]
        rel_path = row["relative_path"]
        class_name = row["class_name"]

        img_full_path = self.dataset_root / rel_path
        if not img_full_path.exists():
            # In case dataset_root points directly to all_image or root
            alt_path = self.dataset_root / "all_image" / rel_path
            if alt_path.exists():
                img_full_path = alt_path
            else:
                raise FileNotFoundError(f"Histopathology image file missing: {img_full_path}")

        # Read bytes and convert RGBA to RGB losslessly
        file_bytes = img_full_path.read_bytes()
        with Image.open(io.BytesIO(file_bytes)) as raw_img:
            img = raw_img.convert("RGB")

        # Resize to standard canonical size if necessary
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)

        # Apply augmentation strictly if training
        if self.is_train and self.augmentor:
            img = self.augmentor(img)

        # Convert to tensor based on architecture requirements
        if self.mode == "sequence":
            # [16, 3, 56, 56]
            tensor_data = image_to_patch_sequence(img, self.patch_size, self.sequence_length)
        else:
            # 2D ConvNet tensor: [3, 224, 224] in [0, 1] range
            arr = np.asarray(img, dtype=np.float32) / 255.0
            # [H, W, C] -> [C, H, W]
            tensor_data = torch.from_numpy(arr).permute(2, 0, 1)

        label_idx = CLASS_TO_IDX.get(class_name, int(row.get("class_index", -1)))
        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        return tensor_data, label_tensor, rel_path

