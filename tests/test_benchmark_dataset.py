"""Tests for Benchmark Dataset and Preprocessing."""

import csv
from pathlib import Path
from PIL import Image
import torch
import numpy as np

from research.benchmark.dataset import (
    ManifestHistopathologyDataset,
    PathologyAugmentation,
    CANONICAL_CLASSES,
)


def _create_synthetic_rgba_image(path: Path, color=(180, 120, 160, 255)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", (224, 224), color=color)
    img.save(path)


def test_manifest_dataset_loading_and_conversion(tmp_path: Path):
    dataset_dir = tmp_path / "mock_dataset"
    manifest_csv = tmp_path / "test_manifest.csv"

    # Create 3 synthetic RGBA images
    img1 = dataset_dir / "TUM" / "TUM_1.png"
    img2 = dataset_dir / "NOR" / "NOR_1.png"
    img3 = dataset_dir / "STR" / "STR_1.png"
    _create_synthetic_rgba_image(img1)
    _create_synthetic_rgba_image(img2)
    _create_synthetic_rgba_image(img3)

    # Write manifest CSV
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["relative_path", "class_name", "class_index"])
        writer.writerow(["TUM/TUM_1.png", "TUM", "7"])
        writer.writerow(["NOR/NOR_1.png", "NOR", "5"])
        writer.writerow(["STR/STR_1.png", "STR", "6"])

    # 1. Test 2D mode
    ds_2d = ManifestHistopathologyDataset(
        manifest_path=manifest_csv,
        dataset_root=dataset_dir,
        mode="2d",
        is_train=False,
    )
    assert len(ds_2d) == 3
    tensor_2d, label_2d, rel_path = ds_2d[0]
    assert isinstance(tensor_2d, torch.Tensor)
    assert tensor_2d.shape == (3, 224, 224)  # 3 channels (converted from RGBA)
    assert label_2d.item() == 7
    assert rel_path == "TUM/TUM_1.png"

    # 2. Test Sequence mode
    ds_seq = ManifestHistopathologyDataset(
        manifest_path=manifest_csv,
        dataset_root=dataset_dir,
        mode="sequence",
        is_train=False,
    )
    tensor_seq, label_seq, _ = ds_seq[1]
    assert tensor_seq.shape == (16, 3, 56, 56)
    assert label_seq.item() == 5


def test_deterministic_eval_preprocessing(tmp_path: Path):
    dataset_dir = tmp_path / "mock_dataset"
    manifest_csv = tmp_path / "test_manifest.csv"
    img_path = dataset_dir / "ADI" / "ADI_1.png"
    _create_synthetic_rgba_image(img_path)

    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["relative_path", "class_name", "class_index"])
        writer.writerow(["ADI/ADI_1.png", "ADI", "0"])

    ds_eval = ManifestHistopathologyDataset(
        manifest_path=manifest_csv,
        dataset_root=dataset_dir,
        mode="2d",
        is_train=False,
    )
    t1, _, _ = ds_eval[0]
    t2, _, _ = ds_eval[0]
    # Invariant: Eval preprocessing must be 100% bitwise deterministic
    assert torch.equal(t1, t2)


def test_pathology_augmentation():
    img = Image.new("RGB", (64, 64), color=(200, 100, 150))
    # Put an asymmetric mark
    img.putpixel((0, 0), (0, 0, 0))
    augmentor = PathologyAugmentation(hflip_prob=1.0, vflip_prob=0.0, rot90_prob=0.0, color_jitter=False)
    aug_img = augmentor(img)
    # H-flip moves pixel (0,0) to (63, 0)
    assert aug_img.getpixel((63, 0)) == (0, 0, 0)

