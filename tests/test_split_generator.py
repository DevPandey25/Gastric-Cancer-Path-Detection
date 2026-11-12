"""Tests for Dataset Manifest and Reproducible Split Generator."""

from pathlib import Path
from PIL import Image

from research.data.manifest_generator import DatasetManifestGenerator, ManifestRecord
from research.data.split_generator import ReproducibleSplitGenerator


def _create_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (224, 224), color=color)
    img.save(path)


def test_manifest_generator(tmp_path: Path):
    classes = ["ADI", "TUM"]
    dataset_dir = tmp_path / "manifest_test_dir"

    for c in classes:
        for i in range(3):
            _create_image(dataset_dir / c / f"{c}_{i}.png", color=(50 * i, 100, 150))

    generator = DatasetManifestGenerator(dataset_dir=dataset_dir, class_names=classes)
    records = generator.generate_manifest()

    assert len(records) == 6
    # Verify deterministic alphabetical ordering by relative path
    paths = [r.relative_path for r in records]
    assert paths == sorted(paths)
    assert records[0].width == 224
    assert records[0].channels == 3
    assert len(records[0].sha256) == 64
    assert len(records[0].dhash) == 16
    assert records[0].audit_status == "valid"

    # CSV save and reload
    csv_file = tmp_path / "manifest.csv"
    sha_csv = generator.save_csv(records, csv_file)
    assert len(sha_csv) == 64

    reloaded = generator.load_csv(csv_file)
    assert len(reloaded) == 6
    assert reloaded[0].relative_path == records[0].relative_path
    assert reloaded[0].sha256 == records[0].sha256


def test_split_generator_reproducibility_and_exclusivity(tmp_path: Path):
    classes = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]
    records: list[ManifestRecord] = []

    # Create 80 synthetic records (10 per class)
    for c_idx, c in enumerate(classes):
        for i in range(10):
            records.append(
                ManifestRecord(
                    relative_path=f"{c}/{c}_{i}.png",
                    class_name=c,
                    class_index=c_idx,
                    width=224,
                    height=224,
                    channels=3,
                    file_size_bytes=1000 + i,
                    sha256=f"hash_{c}_{i}",
                    dhash=f"dhash_{c}_{i}",
                    dataset_id="GCHTID",
                    audit_status="valid",
                )
            )

    splits_dir1 = tmp_path / "splits_run1"
    splitter = ReproducibleSplitGenerator(
        seed=42,
        split_ratios=(0.70, 0.15, 0.15),
        splits_dir=splits_dir1,
    )
    summary1, train1, val1, test1 = splitter.generate_splits(records, manifest_source_hash="test_manifest_hash")

    # Mutual exclusivity
    train_paths = set(r.relative_path for r in train1)
    val_paths = set(r.relative_path for r in val1)
    test_paths = set(r.relative_path for r in test1)

    assert len(train_paths & val_paths) == 0
    assert len(train_paths & test_paths) == 0
    assert len(val_paths & test_paths) == 0
    assert len(train1) + len(val1) + len(test1) == 80

    # Class stratification check: each class represented in train, val, and test
    for c in classes:
        assert summary1.class_distribution[c]["train_count"] > 0
        assert summary1.class_distribution[c]["val_count"] > 0
        assert summary1.class_distribution[c]["test_count"] > 0

    # Reproducibility run with same seed yields identical hashes and assignments
    splits_dir2 = tmp_path / "splits_run2"
    splitter2 = ReproducibleSplitGenerator(
        seed=42,
        split_ratios=(0.70, 0.15, 0.15),
        splits_dir=splits_dir2,
    )
    summary2, train2, val2, test2 = splitter2.generate_splits(records, manifest_source_hash="test_manifest_hash")

    assert [r.relative_path for r in train1] == [r.relative_path for r in train2]
    assert [r.relative_path for r in val1] == [r.relative_path for r in val2]
    assert [r.relative_path for r in test1] == [r.relative_path for r in test2]
    assert summary1.split_manifest_hashes == summary2.split_manifest_hashes
    assert "SCIENTIFIC INTEGRITY DISCLOSURE" in summary1.scientific_disclosure


def test_split_leakage_duplicate_grouping(tmp_path: Path):
    # Create records with identical SHA-256 (exact duplicate pairs)
    records = [
        ManifestRecord("TUM/t1.png", "TUM", 7, 224, 224, 3, 100, "dup_sha_1", "d1", "GCHTID", "valid"),
        ManifestRecord("TUM/t2.png", "TUM", 7, 224, 224, 3, 100, "dup_sha_1", "d1", "GCHTID", "valid"),  # Duplicate of t1
        ManifestRecord("TUM/t3.png", "TUM", 7, 224, 224, 3, 100, "unique_sha_2", "d2", "GCHTID", "valid"),
        ManifestRecord("NOR/n1.png", "NOR", 5, 224, 224, 3, 100, "dup_sha_3", "d3", "GCHTID", "valid"),
        ManifestRecord("NOR/n2.png", "NOR", 5, 224, 224, 3, 100, "dup_sha_3", "d3", "GCHTID", "valid"),  # Duplicate of n1
    ]

    splits_dir = tmp_path / "leak_test_splits"
    splitter = ReproducibleSplitGenerator(seed=42, splits_dir=splits_dir)
    summary, train, val, test = splitter.generate_splits(records)

    train_set = set(r.relative_path for r in train)
    val_set = set(r.relative_path for r in val)
    test_set = set(r.relative_path for r in test)

    # Invariant: t1 and t2 MUST be in the exact same partition
    for part in (train_set, val_set, test_set):
        if "TUM/t1.png" in part:
            assert "TUM/t2.png" in part, "Exact duplicate TUM/t2.png leaked into a different split partition!"
        if "NOR/n1.png" in part:
            assert "NOR/n2.png" in part, "Exact duplicate NOR/n2.png leaked into a different split partition!"

