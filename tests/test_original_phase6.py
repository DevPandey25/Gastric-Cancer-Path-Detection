"""Test suite for Original Phase 6: Tissue Segmentation Pipeline and Baseline Architecture.

Validates:
1. 9-Class Segmentation Ontology: Background (0) + 8 canonical GCHTID classes (ADI..TUM).
2. QuPath GeoJSON ingestion pipeline (parsing, coordinate validation, bounds checks, rasterization).
3. 2D U-Net baseline architecture (9 output channels, forward pass shapes, parameter counts, mask prediction).
4. Loss functions (DiceLoss, CombinedDiceCrossEntropyLoss).
5. Evaluation metrics (Dice coefficient, IoU/Jaccard, empty/disjoint edge cases).
6. Dataset & DataLoader (paired loading, missing-mask fallback handling).
7. Visualization & Overlay generation (alpha blending, untrained state placeholder).
8. Data audit & Repository invariants (zero masks in GCHTID confirmed, baseline checkpoint SHA-256).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np
import pytest
import torch

from research.segmentation.annotations import (
    CANONICAL_SEGMENTATION_TAXONOMY,
    DEFAULT_CLASS_MAPPING,
    DEFAULT_OVERLAP_PRIORITY,
    QuPathAnnotationParser,
    generate_synthetic_geojson_fixture,
    rasterize_qupath_geojson,
)
from research.segmentation.dataset import (
    HistopathologySegmentationDataset,
    create_segmentation_dataloader,
)
from research.segmentation.inference import (
    DEFAULT_CLASS_COLORS,
    DEFAULT_CLASS_NAMES,
    overlay_mask_on_image,
    render_segmentation_comparison,
)
from research.segmentation.losses import (
    DiceLoss,
    CombinedDiceCrossEntropyLoss,
)
from research.segmentation.metrics import (
    compute_dice_coefficient,
    compute_iou_jaccard,
    evaluate_segmentation_batch,
)
from research.segmentation.unet import (
    UNet,
    create_unet_baseline,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
SEGMENTATION_DIR = WORKSPACE_ROOT / "research" / "segmentation"
BASELINE_CKPT = WORKSPACE_ROOT / "models" / "checkpoints" / "gastric_rnn_classifier.pt"
BASELINE_SHA256 = "65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8"

EXPECTED_CANONICAL_CLASSES = {
    0: "Background",
    1: "ADI",
    2: "DEB",
    3: "LYM",
    4: "MUC",
    5: "MUS",
    6: "NOR",
    7: "STR",
    8: "TUM",
}


class TestQuPathAnnotationParser:
    """Tests QuPath GeoJSON parsing, coordinate checks, and rasterization."""

    def test_all_canonical_classes_supported(self):
        parser = QuPathAnnotationParser(image_size=(224, 224))
        # Ensure Background (0) and all 8 tissue classes are mapped
        for c_idx, code in EXPECTED_CANONICAL_CLASSES.items():
            assert parser.resolve_class_index(code) == c_idx
            assert parser.resolve_class_index(code.lower()) == c_idx

        # Ensure explicit synonym mappings
        assert parser.resolve_class_index("Adipose") == 1
        assert parser.resolve_class_index("Necrosis") == 2
        assert parser.resolve_class_index("Lymphocytes") == 3
        assert parser.resolve_class_index("Inflammatory") == 3
        assert parser.resolve_class_index("Mucus") == 4
        assert parser.resolve_class_index("Smooth Muscle") == 5
        assert parser.resolve_class_index("Normal Mucosa") == 6
        assert parser.resolve_class_index("NORM") == 6
        assert parser.resolve_class_index("Stroma") == 7
        assert parser.resolve_class_index("Tumor") == 8
        assert parser.resolve_class_index("Adenocarcinoma") == 8

    def test_synthetic_geojson_generation(self):
        geojson = generate_synthetic_geojson_fixture()
        assert geojson["type"] == "FeatureCollection"
        assert len(geojson["features"]) == 3
        classes = [f["properties"]["classification"]["name"] for f in geojson["features"]]
        assert "ADI" in classes
        assert "TUM" in classes
        assert "STR" in classes

    def test_coordinate_validation_valid_and_invalid(self):
        parser = QuPathAnnotationParser(image_size=(224, 224))

        # Valid polygon (exterior ring of 4 points + closed loop)
        valid_coords = [[[10.0, 10.0], [50.0, 10.0], [50.0, 50.0], [10.0, 50.0], [10.0, 10.0]]]
        is_valid, pts, issues = parser.validate_polygon_coordinates(valid_coords)
        assert is_valid is True
        assert pts is not None
        assert pts.shape == (5, 2)
        assert len(issues) == 0

        # Out-of-bounds coordinates
        oob_coords = [[[10.0, 10.0], [300.0, 10.0], [300.0, 50.0], [10.0, 10.0]]]
        is_valid, pts, issues = parser.validate_polygon_coordinates(oob_coords)
        assert is_valid is True  # Valid geometry but records out-of-bounds issue
        assert any("exceed image bounds" in issue for issue in issues)

        # Degenerate coordinates (fewer than 3 vertices)
        degenerate = [[[10.0, 10.0], [20.0, 20.0]]]
        is_valid, pts, issues = parser.validate_polygon_coordinates(degenerate)
        assert is_valid is False
        assert pts is None

    def test_rasterization_output_properties(self):
        geojson = generate_synthetic_geojson_fixture()
        mask, summary = rasterize_qupath_geojson(geojson, target_size=(224, 224))

        assert isinstance(mask, np.ndarray)
        assert mask.shape == (224, 224)
        assert mask.dtype == np.uint8
        assert summary["total_features_evaluated"] == 3
        assert summary["successfully_rasterized"] == 3
        assert summary["skipped_or_rejected"] == 0

        # Verify class pixel counts: ADI (1), STR (7), TUM (8), and Background (0)
        assert np.sum(mask == 1) > 0  # ADI
        assert np.sum(mask == 7) > 0  # STR
        assert np.sum(mask == 8) > 0  # TUM
        assert np.sum(mask == 0) > 0  # Background

    def test_overlap_priority_resolution(self):
        # Create overlapping boxes: Box A (Adipose) and Box B (Tumor) overlapping
        overlap_features = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[20, 20], [80, 20], [80, 80], [20, 80], [20, 20]]],
                },
                "properties": {"classification": {"name": "ADI"}},
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[40, 40], [100, 40], [100, 100], [40, 100], [40, 40]]],
                },
                "properties": {"classification": {"name": "TUM"}},
            },
        ]
        # Priority: ADI first, TUM second -> TUM should overwrite in overlap region (40-80, 40-80)
        parser = QuPathAnnotationParser(
            image_size=(224, 224),
            overlap_priority=["Background", "ADI", "TUM"],
        )
        mask, _ = parser.rasterize_annotations(overlap_features)
        # Point (50, 50) is in both boxes -> should have Tumor class (8)
        assert mask[50, 50] == 8
        # Point (30, 30) is only in ADI box -> should have ADI class (1)
        assert mask[30, 30] == 1


class TestUNetArchitecture:
    """Tests 2D U-Net baseline architecture instantiation and forward pass."""

    def test_unet_initialization_and_parameters(self):
        model = create_unet_baseline(in_channels=3, num_classes=9, features=[32, 64, 128, 256])
        n_params = model.count_parameters()
        assert n_params == 7_763_305, f"Expected 7,763,305 parameters, got {n_params}"

    def test_unet_forward_pass_shape(self):
        model = create_unet_baseline(in_channels=3, num_classes=9, features=[32, 64, 128, 256])
        model.eval()

        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (2, 9, 224, 224)
        assert not torch.isnan(output).any()

    def test_unet_predict_mask(self):
        model = create_unet_baseline(in_channels=3, num_classes=9, features=[32, 64, 128, 256])
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        preds, probs = model.predict_mask(dummy_input)

        assert isinstance(preds, torch.Tensor)
        assert preds.shape == (1, 224, 224)
        assert preds.dtype == torch.long
        assert preds.min().item() >= 0
        assert preds.max().item() < 9

        assert isinstance(probs, torch.Tensor)
        assert probs.shape == (1, 9, 224, 224)
        assert torch.allclose(probs.sum(dim=1), torch.ones(1, 224, 224), atol=1e-5)


class TestSegmentationLossesAndMetrics:
    """Tests multi-class Dice and Combined Cross-Entropy loss & evaluation metrics."""

    def test_dice_loss(self):
        dice_loss_fn = DiceLoss(smooth=1e-5)
        logits = torch.randn(2, 9, 32, 32)
        targets = torch.randint(0, 9, (2, 32, 32), dtype=torch.long)

        loss = dice_loss_fn(logits, targets)
        assert loss.ndim == 0
        assert 0.0 <= loss.item() <= 1.0

    def test_combined_loss(self):
        combined_loss_fn = CombinedDiceCrossEntropyLoss(
            weight_ce=0.5, weight_dice=0.5
        )
        logits = torch.randn(2, 9, 32, 32)
        targets = torch.randint(0, 9, (2, 32, 32), dtype=torch.long)

        total_loss = combined_loss_fn(logits, targets)
        assert total_loss.ndim == 0
        assert total_loss.item() > 0.0

    def test_dice_and_iou_exact_match(self):
        m1 = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        m2 = np.array([[1, 1], [0, 0]], dtype=np.uint8)

        dice = compute_dice_coefficient(m1, m2, class_id=1)
        iou = compute_iou_jaccard(m1, m2, class_id=1)
        assert dice == pytest.approx(1.0, 1e-6)
        assert iou == pytest.approx(1.0, 1e-6)

    def test_dice_and_iou_disjoint(self):
        m1 = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        m2 = np.array([[0, 0], [1, 1]], dtype=np.uint8)

        dice = compute_dice_coefficient(m1, m2, class_id=1)
        iou = compute_iou_jaccard(m1, m2, class_id=1)
        assert dice == pytest.approx(0.0, 1e-6)
        assert iou == pytest.approx(0.0, 1e-6)

    def test_dice_and_iou_empty_masks(self):
        # Both empty for class 1 -> 1.0 (correctly identified absence)
        m1 = np.zeros((4, 4), dtype=np.uint8)
        m2 = np.zeros((4, 4), dtype=np.uint8)

        dice = compute_dice_coefficient(m1, m2, class_id=1)
        iou = compute_iou_jaccard(m1, m2, class_id=1)
        assert dice == 1.0
        assert iou == 1.0

        # One empty, one non-empty -> 0.0
        m3 = np.zeros((4, 4), dtype=np.uint8)
        m3[0, 0] = 1
        assert compute_dice_coefficient(m1, m3, class_id=1) == 0.0
        assert compute_iou_jaccard(m1, m3, class_id=1) == 0.0

    def test_batch_metrics_evaluation(self):
        preds = np.random.randint(0, 9, size=(3, 32, 32), dtype=np.uint8)
        targets = np.random.randint(0, 9, size=(3, 32, 32), dtype=np.uint8)
        class_names = {f"Class_{i}": i for i in range(9)}

        metrics = evaluate_segmentation_batch(
            preds, targets, class_names=class_names, include_background=True
        )
        assert "mean_dice" in metrics
        assert "mean_iou" in metrics
        assert "per_class_dice" in metrics
        assert "per_class_iou" in metrics
        assert 0.0 <= metrics["mean_dice"] <= 1.0
        assert 0.0 <= metrics["mean_iou"] <= 1.0


class TestSegmentationDataset:
    """Tests segmentation dataset and missing-mask fallback behavior."""

    def test_missing_mask_handling(self, tmp_path):
        # Create a dummy image
        img_path = tmp_path / "test_tile.png"
        dummy_img = np.full((224, 224, 3), 180, dtype=np.uint8)
        cv2.imwrite(str(img_path), dummy_img)

        # Dataset without mask path
        dataset = HistopathologySegmentationDataset(
            samples=[{"image_path": str(img_path)}],
        )
        assert len(dataset) == 1
        sample = dataset[0]

        assert "image" in sample
        assert "mask" in sample
        assert "has_mask" in sample
        assert sample["has_mask"] is False
        assert sample["image"].shape == (3, 224, 224)
        assert sample["mask"].shape == (224, 224)
        # When has_mask is False, mask should be all zeros (background)
        assert torch.all(sample["mask"] == 0)


class TestSegmentationVisualization:
    """Tests overlay generation, transparency, and reporting figures."""

    def test_overlay_mask_on_image(self):
        img = np.full((224, 224, 3), 200, dtype=np.uint8)
        mask = np.zeros((224, 224), dtype=np.uint8)
        mask[50:150, 50:150] = 8  # Tumor (8)

        blended = overlay_mask_on_image(img, mask, alpha=0.5)
        assert blended.shape == (224, 224, 3)
        assert blended.dtype == np.uint8
        # Outside mask should be unchanged
        assert np.array_equal(blended[0, 0], [200, 200, 200])
        # Inside mask should be blended
        assert not np.array_equal(blended[100, 100], [200, 200, 200])

    def test_example_overlay_figure_exists(self):
        p = SEGMENTATION_DIR / "segmentation_overlay_example.png"
        assert p.exists(), f"Missing example overlay: {p}"
        assert p.stat().st_size > 50_000, f"Overlay image too small: {p.stat().st_size}"


class TestOriginalPhase6AuditAndInvariants:
    """Validates segmentation data audit findings and baseline integrity."""

    def test_data_audit_confirms_zero_masks(self):
        cfg_path = SEGMENTATION_DIR / "config.json"
        assert cfg_path.exists(), f"Missing config: {cfg_path}"
        with open(cfg_path) as f:
            cfg = json.load(f)

        assert cfg["phase"] == "Original Phase 6"
        assert cfg["implementation_status"] == "BLOCKED_BY_DATA_UNTRAINED"
        assert cfg["classes"]["num_classes"] == 9
        assert cfg["classes"]["mapping"]["ADI"] == 1
        assert cfg["classes"]["mapping"]["LYM"] == 3
        assert cfg["classes"]["mapping"]["TUM"] == 8
        assert cfg["model"]["num_classes"] == 9
        assert cfg["model"]["total_parameters"] == 7763305
        assert cfg["dataset_audit"]["pixel_segmentation_masks_found"] == 0
        assert cfg["dataset_audit"]["qupath_geojson_annotations_found"] == 0
        assert cfg["dataset_audit"]["xml_contour_annotations_found"] == 0

    def test_baseline_checkpoint_sha256_unmodified(self):
        assert BASELINE_CKPT.exists(), f"Baseline checkpoint not found at {BASELINE_CKPT}"
        hasher = hashlib.sha256()
        with open(BASELINE_CKPT, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        actual_hash = hasher.hexdigest()
        assert actual_hash == BASELINE_SHA256, (
            f"Baseline checkpoint was modified! Expected {BASELINE_SHA256}, got {actual_hash}"
        )
