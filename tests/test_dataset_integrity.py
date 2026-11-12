"""Tests for Dataset Integrity Scanner."""

from pathlib import Path
from PIL import Image
import numpy as np

from research.data.integrity_scanner import DatasetIntegrityScanner


def _create_synthetic_image(path: Path, size: tuple[int, int] = (224, 224), mode: str = "RGB", color: tuple[int, int, int] = (180, 100, 150)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "RGB":
        img = Image.new("RGB", size, color=color)
    elif mode == "L":
        img = Image.new("L", size, color=color[0])
    else:
        img = Image.new(mode, size)
    img.save(path)


def test_integrity_scanner_valid_dataset(tmp_path: Path):
    classes = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]
    dataset_dir = tmp_path / "mock_dataset"

    # Create 2 valid images per class
    for c in classes:
        for i in range(2):
            img_path = dataset_dir / c / f"{c}_{i}.png"
            _create_synthetic_image(img_path, size=(224, 224), mode="RGB", color=(100 + i * 20, 120, 140))

    scanner = DatasetIntegrityScanner(dataset_dir=dataset_dir, expected_classes=classes)
    report, records = scanner.scan(compute_rgb_stats=True)

    assert report.total_files_scanned == 16
    assert report.total_valid == 16
    assert report.total_corrupt == 0
    assert report.total_invalid_dimension == 0
    assert report.total_invalid_channels == 0
    assert len(report.missing_classes) == 0
    assert report.class_counts["TUM"] == 2
    assert "224x224" in report.dimensions_distribution
    assert report.dimensions_distribution["224x224"] == 16
    assert report.rgb_intensity_stats["channel_r"]["mean"] > 0
    assert len(records) == 16

    # Test JSON and CSV export
    json_out = tmp_path / "report.json"
    csv_out = tmp_path / "records.csv"
    report.to_json(json_out)
    report.to_csv(csv_out, records)
    assert json_out.exists() and json_out.stat().st_size > 0
    assert csv_out.exists() and csv_out.stat().st_size > 0


def test_integrity_scanner_corrupt_files(tmp_path: Path):
    classes = ["TUM", "NOR"]
    dataset_dir = tmp_path / "corrupt_dataset"

    # Valid image
    _create_synthetic_image(dataset_dir / "TUM" / "valid.png", size=(224, 224))

    # Empty 0-byte file
    empty_file = dataset_dir / "TUM" / "empty.png"
    empty_file.parent.mkdir(parents=True, exist_ok=True)
    empty_file.write_bytes(b"")

    # Corrupted header / random text file
    junk_file = dataset_dir / "NOR" / "corrupted.png"
    junk_file.parent.mkdir(parents=True, exist_ok=True)
    junk_file.write_bytes(b"NOT_A_VALID_PNG_HEADER_DATA_STREAM")

    scanner = DatasetIntegrityScanner(dataset_dir=dataset_dir, expected_classes=classes)
    report, records = scanner.scan()

    assert report.total_files_scanned == 3
    assert report.total_valid == 1
    assert report.total_corrupt == 2
    assert len(report.corrupt_files) == 2


def test_integrity_scanner_dimension_and_channel_anomalies(tmp_path: Path):
    classes = ["TUM", "STR"]
    dataset_dir = tmp_path / "anomaly_dataset"

    # 1 valid 224x224 RGB
    _create_synthetic_image(dataset_dir / "TUM" / "valid.png", size=(224, 224), mode="RGB")
    # 1 invalid size 100x100 RGB
    _create_synthetic_image(dataset_dir / "TUM" / "wrong_dim.png", size=(100, 100), mode="RGB")
    # 1 grayscale image (1 channel)
    _create_synthetic_image(dataset_dir / "STR" / "grayscale.png", size=(224, 224), mode="L")

    scanner = DatasetIntegrityScanner(dataset_dir=dataset_dir, expected_classes=classes)
    report, records = scanner.scan()

    assert report.total_files_scanned == 3
    assert report.total_valid == 1
    assert report.total_invalid_dimension == 1
    assert report.total_invalid_channels == 1

