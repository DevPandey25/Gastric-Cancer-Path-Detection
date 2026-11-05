"""Tests for Perceptual Leakage and Near-Duplicate Detection."""

from pathlib import Path
from PIL import Image, ImageEnhance
import numpy as np

from research.data.leakage_detector import (
    NearDuplicateDetector,
    compute_dhash,
    compute_ahash,
    hamming_distance,
)


def _create_patterned_image(path: Path, pattern_type: str = "gradient") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if pattern_type == "gradient":
        arr = np.linspace(0, 255, 64 * 64, dtype=np.uint8).reshape((64, 64))
        img = Image.fromarray(arr, mode="L").convert("RGB")
    elif pattern_type == "checkerboard":
        arr = np.indices((64, 64)).sum(axis=0) % 2 * 255
        img = Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
    elif pattern_type == "vertical":
        arr = np.tile(np.arange(64, dtype=np.uint8) * 4, (64, 1))
        img = Image.fromarray(arr, mode="L").convert("RGB")
    else:
        img = Image.new("RGB", (64, 64), color=(100, 100, 100))
    img.save(path)


def test_dhash_computation_and_distance(tmp_path: Path):
    img_path1 = tmp_path / "img1.png"
    # Create textured image
    np.random.seed(42)
    tex_arr = (np.random.rand(64, 64) * 200).astype(np.uint8)
    Image.fromarray(tex_arr, mode="L").convert("RGB").save(img_path1)

    # Identical image
    h1 = compute_dhash(img_path1)
    h2 = compute_dhash(img_path1)
    assert h1 == h2
    assert hamming_distance(h1, h2) == 0

    # Slightly perturbed image (subtle brightness change)
    with Image.open(img_path1) as img:
        enhancer = ImageEnhance.Brightness(img)
        perturbed = enhancer.enhance(1.02)
        img_path2 = tmp_path / "img2.png"
        perturbed.save(img_path2)

    h_perturbed = compute_dhash(img_path2)
    dist = hamming_distance(h1, h_perturbed)
    assert dist <= 3  # Near-duplicate threshold

    # Completely different image pattern
    img_path3 = tmp_path / "img3.png"
    _create_patterned_image(img_path3, "checkerboard")
    h3 = compute_dhash(img_path3)
    dist_diff = hamming_distance(h1, h3)
    assert dist_diff > 10


def test_near_duplicate_clustering(tmp_path: Path):
    dataset_dir = tmp_path / "cluster_dataset"
    _create_patterned_image(dataset_dir / "TUM" / "orig.png", "vertical")

    # Save near duplicate with minor contrast adjustment
    with Image.open(dataset_dir / "TUM" / "orig.png") as img:
        enhancer = ImageEnhance.Contrast(img)
        near_dup = enhancer.enhance(1.02)
        (dataset_dir / "TUM" / "near_dup.png").parent.mkdir(parents=True, exist_ok=True)
        near_dup.save(dataset_dir / "TUM" / "near_dup.png")

    # Distinct image
    _create_patterned_image(dataset_dir / "TUM" / "distinct.png", "checkerboard")

    detector = NearDuplicateDetector(dataset_dir=dataset_dir, hamming_threshold=3)
    report, _ = detector.find_near_duplicates()

    assert report.total_images_evaluated == 3
    assert report.total_clusters == 1
    cluster = report.clusters[0]
    assert cluster.size == 2
    assert "TUM/orig.png" in cluster.paths
    assert "TUM/near_dup.png" in cluster.paths
    assert "TUM/distinct.png" not in cluster.paths
    assert not cluster.is_cross_class
    assert "SCIENTIFIC INTEGRITY LIMITATION" in report.scientific_limitation_disclosure


def test_near_duplicate_cross_class_leakage(tmp_path: Path):
    dataset_dir = tmp_path / "leakage_dataset"
    _create_patterned_image(dataset_dir / "TUM" / "tum_patch.png", "gradient")

    with Image.open(dataset_dir / "TUM" / "tum_patch.png") as img:
        enhancer = ImageEnhance.Brightness(img)
        leakage_patch = enhancer.enhance(1.01)
        (dataset_dir / "STR").mkdir(parents=True, exist_ok=True)
        leakage_patch.save(dataset_dir / "STR" / "str_patch.png")

    detector = NearDuplicateDetector(dataset_dir=dataset_dir, hamming_threshold=3)
    report, _ = detector.find_near_duplicates()

    assert report.cross_class_clusters_count == 1
    assert len(report.cross_class_conflicts) == 1
    conflict = report.cross_class_conflicts[0]
    assert set(conflict["conflicting_classes"]) == {"TUM", "STR"}
