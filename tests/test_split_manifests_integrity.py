"""Integrity and Mutual Exclusivity Verification for Generated Dataset Splits."""

import csv
from pathlib import Path

from research.config import load_research_config
from research.data.duplicate_detector import compute_sha256


def test_generated_manifests_integrity_and_exclusivity():
    cfg = load_research_config()
    splits_dir = cfg.paths.splits_dir
    manifests_dir = cfg.paths.workspace_root / "research" / "manifests"

    manifest_all_path = manifests_dir / "manifest_all.csv"
    train_path = splits_dir / "manifest_train.csv"
    val_path = splits_dir / "manifest_val.csv"
    test_path = splits_dir / "manifest_test.csv"
    summary_path = splits_dir / "split_summary.json"

    assert manifest_all_path.exists(), "manifest_all.csv missing"
    assert train_path.exists(), "manifest_train.csv missing"
    assert val_path.exists(), "manifest_val.csv missing"
    assert test_path.exists(), "manifest_test.csv missing"
    assert summary_path.exists(), "split_summary.json missing"

    def read_csv(path: Path) -> list[dict[str, str]]:
        with open(path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    all_rows = read_csv(manifest_all_path)
    train_rows = read_csv(train_path)
    val_rows = read_csv(val_path)
    test_rows = read_csv(test_path)

    # 1. Total counts
    assert len(all_rows) == 31096
    assert len(train_rows) == 21767
    assert len(val_rows) == 4664
    assert len(test_rows) == 4665
    assert len(train_rows) + len(val_rows) + len(test_rows) == 31096

    # 2. Path mutual exclusivity
    train_paths = set(r["relative_path"] for r in train_rows)
    val_paths = set(r["relative_path"] for r in val_rows)
    test_paths = set(r["relative_path"] for r in test_rows)

    assert len(train_paths & val_paths) == 0, "Train and Val paths overlap!"
    assert len(train_paths & test_paths) == 0, "Train and Test paths overlap!"
    assert len(val_paths & test_paths) == 0, "Val and Test paths overlap!"

    # 3. Duplicate leakage prevention: No SHA-256 hash in Train can appear in Val or Test
    train_shas = set(r["sha256"] for r in train_rows if r["sha256"])
    val_shas = set(r["sha256"] for r in val_rows if r["sha256"])
    test_shas = set(r["sha256"] for r in test_rows if r["sha256"])

    assert len(train_shas & val_shas) == 0, "Duplicate SHA-256 leaked between Train and Val!"
    assert len(train_shas & test_shas) == 0, "Duplicate SHA-256 leaked between Train and Test!"
    assert len(val_shas & test_shas) == 0, "Duplicate SHA-256 leaked between Val and Test!"

    # 4. Checksums match split_summary.json
    import json
    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    assert summary_data["split_manifest_hashes"]["manifest_train.csv"] == compute_sha256(train_path)
    assert summary_data["split_manifest_hashes"]["manifest_val.csv"] == compute_sha256(val_path)
    assert summary_data["split_manifest_hashes"]["manifest_test.csv"] == compute_sha256(test_path)
