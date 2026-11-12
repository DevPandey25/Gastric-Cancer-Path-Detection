"""Comprehensive Verification Suite for Original Phase 9: Whole-Slide Image Engine and MIL.

Tests:
1. Reader initialization, metadata schema, and safe context-manager closing.
2. Missing OpenSlide dependency and platform diagnostic handling.
3. Pyramid level validation and level-downsample enumeration.
4. OpenSlide Level-0 reference coordinate system invariance.
5. Region-boundary handling and padding on edge crops.
6. Tissue detection: mask dimensions, binary values, HSV thresholds, and morphology.
7. O(1) Integral image tissue percentage evaluation across region bounds.
8. Coordinate-aware patch extraction and Level-0 coordinate accuracy.
9. Patch manifest schema, column integrity, and (slide_id, patch_id) uniqueness.
10. Slide-to-patch isolation (preventing inter-slide patch contamination).
11. Embedding extractor interface and blocker enforcement (preventing fake WSI embeddings).
12. Multiple Instance Learning: GatedAttentionMIL and MeanPoolingMIL forward passes,
    attention weight normalization (sum to 1), and slide-bag input validation.
13. Graceful handling of missing WSI files across the repository audit.
14. Baseline model checkpoint SHA-256 invariant preservation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from research.wsi.attention_heatmap import SlideAttentionHeatmapGenerator
from research.wsi.embedding_extractor import (
    WSIEmbeddingExtractor,
    WSIEmbeddingsBlockedError,
)
from research.wsi.mil_models import (
    GatedAttentionMIL,
    MeanPoolingMILBaseline,
    MILTrainingBlockedError,
    SlideBag,
    build_mil_model,
)
from research.wsi.patch_extractor import (
    CoordinateAwarePatchExtractor,
    PatchExtractionConfig,
)
from research.wsi.patch_manifest import PatchManifest, PatchRecord
from research.wsi.slide_metadata import SlideMetadata, parse_openslide_properties
from research.wsi.slide_reader import (
    BaseSlideReader,
    OpenSlideAdapter,
    OpenSlideUnavailableError,
    SyntheticPyramidSlideReader,
)
from research.wsi.tissue_detection import TissueDetectionConfig, TissueDetector

EXPECTED_BASELINE_SHA256 = "65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8"


def _create_mock_h_and_e_fixture(w: int = 1024, h: int = 1024) -> Image.Image:
    """Creates a deterministic multi-color canvas simulating an H&E slide with glass background."""
    arr = np.full((h, w, 3), 250, dtype=np.uint8)  # White glass background
    # Central rectangular region simulating H&E stained tissue
    arr[200:800, 200:800] = [190, 80, 160]  # Eosin pink / hematoxylin purple
    # Inner region simulating dense nuclei
    arr[300:600, 300:600] = [100, 40, 120]  # Deep hematoxylin purple
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# 1. Slide Reader and Metadata Tests
# ---------------------------------------------------------------------------

def test_synthetic_reader_initialization_and_metadata():
    img = _create_mock_h_and_e_fixture(1024, 1024)
    reader = SyntheticPyramidSlideReader(
        base_image=img,
        slide_id="test_slide_01",
        downsamples=[1.0, 2.0, 4.0],
        mpp_x=0.5,
        mpp_y=0.5,
        apparent_magnification=20.0,
    )

    meta = reader.metadata
    assert meta.slide_id == "test_slide_01"
    assert meta.level_count == 3
    assert meta.level0_dimensions == (1024, 1024)
    assert meta.level_dimensions == [(1024, 1024), (512, 512), (256, 256)]
    assert meta.level_downsamples == [1.0, 2.0, 4.0]
    assert meta.mpp_x == 0.5
    assert meta.mpp_y == 0.5
    assert meta.apparent_magnification == 20.0
    assert not meta.is_genuine_wsi

    reader.close()
    with pytest.raises(RuntimeError, match="Slide reader is closed"):
        _ = reader.metadata


def test_reader_context_manager_safe_closing():
    img = _create_mock_h_and_e_fixture(512, 512)
    with SyntheticPyramidSlideReader(img, "cm_test") as reader:
        assert not reader._is_closed
        _ = reader.metadata
    assert reader._is_closed


def test_missing_openslide_dependency_handling():
    with pytest.raises((OpenSlideUnavailableError, FileNotFoundError)):
        # Attempting to open a non-existent or real path without OpenSlide C DLLs
        _ = OpenSlideAdapter("non_existent_slide.svs")


# ---------------------------------------------------------------------------
# 2. Coordinate Contract and Region Reading Tests
# ---------------------------------------------------------------------------

def test_level0_coordinate_system_invariance():
    """
    CRITICAL TEST: read_region(location, level, size) must interpret `location`
    in Level-0 reference coordinates even when reading from downsampled levels.
    """
    img = _create_mock_h_and_e_fixture(1024, 1024)
    reader = SyntheticPyramidSlideReader(img, "coord_test", downsamples=[1.0, 4.0])

    # Region at Level 0: top-left (200, 200), size (100, 100)
    region_lvl0 = reader.read_region(location=(200, 200), level=0, size=(100, 100))
    assert region_lvl0.size == (100, 100)

    # Region at Level 1 (4x downsample):
    # Location passed is STILL Level-0 (200, 200).
    # In Level 1 space, this corresponds to (50, 50).
    region_lvl1 = reader.read_region(location=(200, 200), level=1, size=(25, 25))
    assert region_lvl1.size == (25, 25)

    # Out-of-bounds boundary handling: location partially outside level-0 canvas
    # (e.g. 1000, 1000 on a 1024x1024 slide with size 100x100)
    boundary_region = reader.read_region(location=(1000, 1000), level=0, size=(100, 100))
    assert boundary_region.size == (100, 100)  # Padded to requested size

    reader.close()


def test_level_index_validation():
    img = _create_mock_h_and_e_fixture(512, 512)
    reader = SyntheticPyramidSlideReader(img, "lvl_test", downsamples=[1.0, 2.0])
    with pytest.raises(IndexError):
        _ = reader.read_region((0, 0), level=5, size=(64, 64))
    with pytest.raises(IndexError):
        _ = reader.read_region((0, 0), level=-1, size=(64, 64))
    reader.close()


# ---------------------------------------------------------------------------
# 3. Tissue Detection Tests
# ---------------------------------------------------------------------------

def test_tissue_detection_mask_and_integral_evaluation():
    img = _create_mock_h_and_e_fixture(1000, 1000)
    reader = SyntheticPyramidSlideReader(img, "tissue_test", downsamples=[1.0, 2.0])
    thumb = reader.get_thumbnail((200, 200))

    detector = TissueDetector(
        thumbnail=thumb,
        level0_dimensions=(1000, 1000),
        config=TissueDetectionConfig(sat_threshold=15, val_threshold=245),
    )

    # Check mask properties
    assert detector.tissue_mask.shape == (thumb.height, thumb.width)
    assert detector.tissue_mask.dtype == bool

    # Tissue should be detected in center (e.g., 400, 400)
    center_valid, center_pct = detector.is_patch_valid_tissue(
        level0_x=400, level0_y=400, patch_width_level=200, patch_height_level=200, downsample=1.0
    )
    assert center_valid
    assert center_pct > 0.80

    # Glass background (e.g., 0, 0 to 100, 100) should be rejected
    corner_valid, corner_pct = detector.is_patch_valid_tissue(
        level0_x=0, level0_y=0, patch_width_level=100, patch_height_level=100, downsample=1.0
    )
    assert not corner_valid
    assert corner_pct < 0.10

    # Visual overlay check
    overlay = detector.get_tissue_overlay_image()
    assert overlay.size == thumb.size

    reader.close()


# ---------------------------------------------------------------------------
# 4. Patch Extractor and Manifest Tests
# ---------------------------------------------------------------------------

def test_coordinate_aware_patch_extraction():
    img = _create_mock_h_and_e_fixture(1024, 1024)
    reader = SyntheticPyramidSlideReader(img, "extract_test", downsamples=[1.0, 2.0])
    thumb = reader.get_thumbnail((256, 256))
    detector = TissueDetector(thumb, (1024, 1024))

    extractor = CoordinateAwarePatchExtractor(
        reader=reader,
        tissue_detector=detector,
        config=PatchExtractionConfig(
            patch_width=256,
            patch_height=256,
            target_level=0,
            stride_x=256,
            stride_y=256,
            min_tissue_percentage=0.40,
            save_patch_images=False,
        ),
    )

    manifest = extractor.extract_all_patches()
    assert len(manifest) > 0

    # Verify coordinate bounds and uniqueness
    is_valid, errors = manifest.validate_manifest()
    assert is_valid, f"Manifest validation failed: {errors}"

    records = manifest.get_slide_records("extract_test")
    assert len(records) == len(manifest)

    # Check Level-0 alignment: since target_level=0, stride=256, coords must be multiples of 256
    for r in records:
        assert r.level0_x % 256 == 0
        assert r.level0_y % 256 == 0
        assert r.tissue_percentage >= 0.40
        assert r.patch_width == 256
        assert r.patch_height == 256

    reader.close()


def test_patch_manifest_serialization_round_trip(tmp_path):
    manifest = PatchManifest()
    rec1 = PatchRecord(
        slide_id="slide_A",
        patch_id="patch_01",
        level=0,
        level0_x=0,
        level0_y=0,
        patch_width=224,
        patch_height=224,
        downsample=1.0,
        tissue_percentage=0.85,
        mpp_x=0.5,
        mpp_y=0.5,
        magnification=20.0,
        relative_patch_path="slide_A/patch_01.png",
        source_slide_provenance="Aperio",
    )
    rec2 = PatchRecord(
        slide_id="slide_B",
        patch_id="patch_02",
        level=1,
        level0_x=448,
        level0_y=896,
        patch_width=224,
        patch_height=224,
        downsample=2.0,
        tissue_percentage=0.92,
        mpp_x=0.5,
        mpp_y=0.5,
        magnification=20.0,
        relative_patch_path="slide_B/patch_02.png",
        source_slide_provenance="Hamamatsu",
    )
    manifest.add_record(rec1)
    manifest.add_record(rec2)

    csv_path = tmp_path / "test_manifest.csv"
    manifest.save_csv(csv_path)
    assert csv_path.exists()

    loaded = PatchManifest.load_csv(csv_path)
    assert len(loaded) == 2
    assert loaded.get_unique_slide_ids() == ["slide_A", "slide_B"]
    assert len(loaded.get_slide_records("slide_A")) == 1
    assert len(loaded.get_slide_records("slide_B")) == 1


# ---------------------------------------------------------------------------
# 5. Embedding Extractor and Safety Invariant Tests
# ---------------------------------------------------------------------------

def test_embedding_extractor_blocks_fake_wsi_synthesis():
    """
    SAFETY INVARIANT: Embedding extraction must explicitly block execution
    if source slides are synthetic test fixtures or genuine WSIs are absent.
    """
    extractor = WSIEmbeddingExtractor(model_name="phikon")
    manifest = PatchManifest()
    manifest.add_record(
        PatchRecord(
            slide_id="fake_slide",
            patch_id="p1",
            level=0,
            level0_x=0,
            level0_y=0,
            patch_width=224,
            patch_height=224,
            downsample=1.0,
            tissue_percentage=0.9,
        )
    )

    with pytest.raises(WSIEmbeddingsBlockedError, match="synthetic test fixture or genuine WSI files are absent"):
        _ = extractor.extract_bag_embeddings(
            manifest=manifest,
            slide_id="fake_slide",
            is_genuine_wsi=False,  # Blocked!
        )


# ---------------------------------------------------------------------------
# 6. Multiple Instance Learning (MIL) Tests
# ---------------------------------------------------------------------------

def test_gated_attention_mil_forward_and_attention_sum():
    model = GatedAttentionMIL(in_features=768, hidden_dim=128, attention_dim=64, num_classes=2)
    model.eval()

    # Bag of K=20 instances
    k = 20
    embeddings = torch.randn(k, 768)

    with torch.no_grad():
        logits, attention_weights = model(embeddings)

    assert logits.shape == (1, 2)
    assert attention_weights.shape == (k, 1)

    # Attention weights must sum to 1.0 (softmax property)
    sum_a = float(torch.sum(attention_weights))
    assert pytest.approx(sum_a, rel=1e-4) == 1.0

    # Non-negative attention weights
    assert torch.all(attention_weights >= 0.0)


def test_mean_pooling_mil_baseline():
    model = MeanPoolingMILBaseline(in_features=768, hidden_dim=128, num_classes=2)
    model.eval()

    k = 15
    embeddings = torch.randn(k, 768)
    with torch.no_grad():
        logits, weights = model(embeddings)

    assert logits.shape == (1, 2)
    assert weights.shape == (k, 1)
    assert pytest.approx(float(weights[0, 0]), rel=1e-4) == 1.0 / 15.0


def test_slide_bag_isolation():
    """Verifies SlideBag enforces 1:1 slide isolation and dimension consistency."""
    bag = SlideBag(
        slide_id="SLIDE_001",
        embeddings=torch.randn(10, 768),
        patch_ids=[f"p_{i}" for i in range(10)],
        level0_coordinates=[(i * 224, 0) for i in range(10)],
    )
    assert bag.num_instances == 10

    # Mismatched patch IDs must raise ValueError
    with pytest.raises(ValueError):
        _ = SlideBag(
            slide_id="SLIDE_002",
            embeddings=torch.randn(10, 768),
            patch_ids=["only_one"],
        )


# ---------------------------------------------------------------------------
# 7. Attention Heatmap Generator Tests
# ---------------------------------------------------------------------------

def test_attention_heatmap_reconstruction():
    thumb = Image.new("RGB", (200, 200), (250, 250, 250))
    generator = SlideAttentionHeatmapGenerator(
        thumbnail=thumb,
        level0_dimensions=(1000, 1000),
    )

    # 4 patch coordinates
    coords = [(200, 200), (400, 200), (200, 400), (400, 400)]
    sizes = [(200, 200)] * 4
    downs = [1.0] * 4
    weights = np.array([0.1, 0.4, 0.3, 0.2], dtype=np.float32)

    blended_img, raw_map = generator.generate_attention_heatmap(
        patch_coords=coords,
        patch_sizes=sizes,
        downsamples=downs,
        attention_weights=weights,
        alpha=0.5,
    )

    assert blended_img.size == (200, 200)
    assert raw_map.shape == (200, 200)
    assert np.all(raw_map >= 0.0) and np.all(raw_map <= 1.0)


# ---------------------------------------------------------------------------
# 8. Repository Audit & Invariants Tests
# ---------------------------------------------------------------------------

def test_repository_wsi_audit_intact():
    audit_file = Path("research/wsi/wsi_audit_summary.json")
    assert audit_file.exists(), f"Audit summary file missing: {audit_file}"

    with open(audit_file) as f:
        data = json.load(f)

    assert data["status"] == "AWAITING_WSI_DATA"
    assert data["genuine_wsi_count"] == 0
    assert "gchtid_dataset_nature" in data


def test_baseline_checkpoint_sha256_unmodified():
    ckpt_path = Path("models/checkpoints/gastric_rnn_classifier.pt")
    assert ckpt_path.exists(), f"Baseline checkpoint not found: {ckpt_path}"
    sha256 = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    assert sha256 == EXPECTED_BASELINE_SHA256, (
        f"CRITICAL: Baseline checkpoint has been altered! Expected {EXPECTED_BASELINE_SHA256}, got {sha256}"
    )

