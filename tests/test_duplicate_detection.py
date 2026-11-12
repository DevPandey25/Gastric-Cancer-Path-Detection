"""Tests for Exact Duplicate Detector."""

from pathlib import Path
from PIL import Image

from research.data.duplicate_detector import ExactDuplicateDetector, compute_sha256


def _create_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (64, 64), color=color)
    img.save(path)


def test_exact_duplicate_intra_class(tmp_path: Path):
    dataset_dir = tmp_path / "dup_dataset"

    # Distinct image
    _create_image(dataset_dir / "TUM" / "unique.png", color=(255, 0, 0))

    # Identical image duplicated inside TUM
    _create_image(dataset_dir / "TUM" / "dup_orig.png", color=(0, 255, 0))
    # Copy exact bytes
    orig_bytes = (dataset_dir / "TUM" / "dup_orig.png").read_bytes()
    (dataset_dir / "TUM" / "dup_copy.png").write_bytes(orig_bytes)

    detector = ExactDuplicateDetector(dataset_dir=dataset_dir)
    report, path_to_hash = detector.find_duplicates()

    assert report.total_files_audited == 3
    assert report.total_unique_hashes == 2
    assert report.total_duplicate_files == 2
    assert report.total_duplicate_groups == 1
    assert report.cross_class_duplicate_groups == 0
    assert report.intra_class_duplicate_groups == 1
    assert len(report.cross_class_conflicts) == 0


def test_exact_duplicate_cross_class_conflict(tmp_path: Path):
    dataset_dir = tmp_path / "conflict_dataset"

    # Create image in TUM
    _create_image(dataset_dir / "TUM" / "tum_lesion.png", color=(128, 64, 32))
    content = (dataset_dir / "TUM" / "tum_lesion.png").read_bytes()

    # Place exact same bytes in NOR (Normal Tissue) - Critical Conflict!
    (dataset_dir / "NOR").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "NOR" / "nor_sample.png").write_bytes(content)

    detector = ExactDuplicateDetector(dataset_dir=dataset_dir)
    report, _ = detector.find_duplicates()

    assert report.total_files_audited == 2
    assert report.total_unique_hashes == 1
    assert report.cross_class_duplicate_groups == 1
    assert len(report.cross_class_conflicts) == 1
    conflict = report.cross_class_conflicts[0]
    assert set(conflict["conflicting_classes"]) == {"TUM", "NOR"}


def test_duplicate_report_export(tmp_path: Path):
    dataset_dir = tmp_path / "export_dataset"
    _create_image(dataset_dir / "TUM" / "img1.png", color=(50, 50, 50))
    b = (dataset_dir / "TUM" / "img1.png").read_bytes()
    (dataset_dir / "TUM" / "img2.png").write_bytes(b)

    detector = ExactDuplicateDetector(dataset_dir=dataset_dir)
    report, _ = detector.find_duplicates()

    json_path = tmp_path / "dup_report.json"
    csv_path = tmp_path / "dup_report.csv"
    report.to_json(json_path)
    report.to_csv(csv_path)

    assert json_path.exists() and json_path.stat().st_size > 0
    assert csv_path.exists() and csv_path.stat().st_size > 0

