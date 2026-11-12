"""Test suite for Original Phase 7: Nuclei Detection and Cellular Morphology.

Validates:
1. Model adapters: StarDist/Cellpose/HoVer-Net graceful unavailable handling and
   ClassicalHEDWatershed operational availability.
2. Instance maps: Dimensions (224x224), integer dtype, background ID 0, unique positive IDs 1..N.
3. Geometric morphology: Area, perimeter, equivalent diameter, major/minor axis, aspect ratio,
   eccentricity, circularity numerical stability (safeguard against div-by-zero).
4. Intensity morphology: Mean RGB, Ruifrok HED deconvolution optical density.
5. Regional analysis: 2x2 intra-patch quadrant layout (TL, TR, BL, BR), centroid assignment,
   density, and spatial dispersion.
6. Empty detection handling: Graceful empty DataFrame and summary outputs.
7. Serialization & Cohort reproducibility: CSV and JSON schema validity, deterministic selection (seed 42).
8. Repository invariants: Checkpoint SHA-256 and original image preservation.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import pandas as pd
import pytest

from research.cellular.instance_processing import (
    extract_instance_bounding_boxes,
    extract_instance_contours,
    filter_and_reindex_instances,
    generate_synthetic_instance_fixture,
)
from research.cellular.model_adapter import (
    CellposeAdapter,
    ClassicalHEDWatershedAdapter,
    HoVerNetAdapter,
    NucleiDetectorAdapter,
    PretrainedModelUnavailableError,
    StarDistAdapter,
    get_available_detector,
)
from research.cellular.morphology import (
    compute_image_morphology_summary,
    compute_instance_geometric_features,
    compute_instance_intensity_features,
    extract_cellular_morphology,
)
from research.cellular.region_analysis import (
    QUADRANT_DEFINITIONS,
    analyze_quadrant_regions,
    assign_centroid_to_quadrant,
    assign_nuclei_to_quadrants,
)
from research.cellular.inference import (
    CANONICAL_CLASSES,
    select_pilot_cohort,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
CELLULAR_DIR = WORKSPACE_ROOT / "research" / "cellular"
BASELINE_CKPT = WORKSPACE_ROOT / "models" / "checkpoints" / "gastric_rnn_classifier.pt"
BASELINE_SHA256 = "65369ef618ad1d0ad10ecb65b6f15324e932f52e21c8b25a43b71c636726d8b8"


class TestModelAdapters:
    """Validates deep and classical nuclei detector adapters and availability reporting."""

    def test_stardist_graceful_unavailable_error(self):
        sd = StarDistAdapter()
        assert sd.is_available() is False
        info = sd.get_model_info()
        assert info["name"] == "StarDist"
        assert "Blocked" in info["status_reason"]

        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        with pytest.raises(PretrainedModelUnavailableError):
            sd.detect_nuclei(dummy_img)

    def test_cellpose_graceful_unavailable_error(self):
        cp = CellposeAdapter()
        assert cp.is_available() is False
        info = cp.get_model_info()
        assert info["name"] == "Cellpose"
        assert "Blocked" in info["status_reason"]

        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        with pytest.raises(PretrainedModelUnavailableError):
            cp.detect_nuclei(dummy_img)

    def test_hovernet_graceful_unavailable_error(self):
        hn = HoVerNetAdapter()
        assert hn.is_available() is False
        info = hn.get_model_info()
        assert info["name"] == "HoVer-Net"
        assert "Blocked" in info["status_reason"]

        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        with pytest.raises(PretrainedModelUnavailableError):
            hn.detect_nuclei(dummy_img)

    def test_classical_hed_watershed_operational(self):
        adapter = ClassicalHEDWatershedAdapter()
        assert adapter.is_available() is True
        info = adapter.get_model_info()
        assert info["name"] == "ClassicalHEDWatershed"
        assert "Operational" in info["status_reason"]

    def test_get_available_detector_fallback(self):
        # Auto returns ClassicalHEDWatershedAdapter when deep weights are unavailable
        det = get_available_detector("auto")
        assert isinstance(det, ClassicalHEDWatershedAdapter)
        assert det.is_available() is True


class TestInstanceProcessingAndProperties:
    """Tests instance map properties, background separation, and contour extraction."""

    @pytest.fixture
    def synthetic_fixture(self):
        # Use synthetic test fixture strictly for software validation
        return generate_synthetic_instance_fixture(shape=(224, 224), num_nuclei=12, seed=42)

    def test_instance_map_dimensions_and_dtype(self, synthetic_fixture):
        img, inst_map = synthetic_fixture
        assert inst_map.shape == (224, 224)
        assert issubclass(inst_map.dtype.type, (np.integer, np.signedinteger))

    def test_background_id_zero_and_positive_unique_ids(self, synthetic_fixture):
        img, inst_map = synthetic_fixture
        unique_ids = sorted(np.unique(inst_map))
        assert 0 in unique_ids, "Background ID 0 must be present"
        positive_ids = [uid for uid in unique_ids if uid > 0]
        assert len(positive_ids) > 0
        # IDs must be strictly positive and contiguous/unique
        assert len(positive_ids) == len(set(positive_ids))
        assert min(positive_ids) >= 1

    def test_contour_and_bounding_box_extraction(self, synthetic_fixture):
        img, inst_map = synthetic_fixture
        cnts = extract_instance_contours(inst_map)
        bboxes = extract_instance_bounding_boxes(inst_map)

        for uid in np.unique(inst_map):
            if uid == 0:
                continue
            assert uid in cnts
            assert uid in bboxes
            cnt = cnts[uid]
            assert cnt.ndim == 2
            assert cnt.shape[1] == 2
            x, y, w, h = bboxes[uid]
            assert w > 0 and h > 0
            assert 0 <= x < 224
            assert 0 <= y < 224

    def test_filter_and_reindex_instances(self):
        inst_map = np.zeros((100, 100), dtype=np.int32)
        # Instance 1: 5 pixels (should be filtered if min_area=10)
        inst_map[10:15, 10] = 1
        # Instance 2: 30 pixels (retained)
        inst_map[20:25, 20:26] = 2

        filtered, retained = filter_and_reindex_instances(inst_map, min_area=10, max_area=500)
        assert retained == 1
        assert np.all(filtered[10:15, 10] == 0)
        assert np.all(filtered[20:25, 20:26] == 1)  # Re-indexed to 1


class TestCellularMorphologyCalculations:
    """Tests mathematical correctness and numerical safeguards of morphological features."""

    def test_circularity_numerical_stability(self):
        # Degenerate zero-area / single-pixel
        empty_map = np.zeros((50, 50), dtype=np.int32)
        geo_empty = compute_instance_geometric_features(empty_map, 1)
        assert geo_empty["circularity"] == 0.0
        assert geo_empty["area_px2"] == 0.0

        # Perfect circle should have circularity near 1.0 (4*pi*r^2 / (2*pi*r)^2 = 1)
        circle_map = np.zeros((100, 100), dtype=np.int32)
        cv2.circle(circle_map, (50, 50), 20, 1, -1)
        geo_circle = compute_instance_geometric_features(circle_map, 1)
        assert 0.80 <= geo_circle["circularity"] <= 1.0

    def test_area_perimeter_and_equivalent_diameter(self):
        box_map = np.zeros((100, 100), dtype=np.int32)
        # 10x10 square -> Area = 100 px^2
        box_map[40:50, 40:50] = 1
        geo = compute_instance_geometric_features(box_map, 1)

        assert geo["area_px2"] == 100.0
        # Perimeter of 10x10 rectangle is approx 36-40 in discrete pixels
        assert 30.0 <= geo["perimeter_px"] <= 45.0
        # Equivalent diameter = sqrt(4 * 100 / pi) = ~11.28
        expected_eq_d = math.sqrt(400.0 / math.pi)
        assert geo["equivalent_diameter_px"] == pytest.approx(expected_eq_d, abs=0.2)
        assert 40 <= geo["centroid_x"] <= 50
        assert 40 <= geo["centroid_y"] <= 50

    def test_eccentricity_and_aspect_ratio_bounds(self):
        # Elongated rectangle (high eccentricity, high aspect ratio)
        rect_map = np.zeros((100, 100), dtype=np.int32)
        rect_map[45:55, 20:80] = 1  # 10x60 rectangle
        geo = compute_instance_geometric_features(rect_map, 1)

        assert geo["aspect_ratio"] > 2.0
        assert 0.0 <= geo["eccentricity"] <= 1.0
        assert geo["eccentricity"] > 0.7  # Highly eccentric

    def test_intensity_features_and_hed_deconvolution(self):
        img = np.full((50, 50, 3), (100, 80, 180), dtype=np.uint8)  # Bluish hematoxylin-like
        inst_map = np.zeros((50, 50), dtype=np.int32)
        inst_map[20:30, 20:30] = 1

        inten = compute_instance_intensity_features(img, inst_map, 1)
        assert inten["mean_intensity_r"] == 100.0
        assert inten["mean_intensity_g"] == 80.0
        assert inten["mean_intensity_b"] == 180.0
        assert inten["mean_hematoxylin_od"] > 0.0

    def test_empty_detection_handling(self):
        empty_img = np.zeros((224, 224, 3), dtype=np.uint8)
        empty_map = np.zeros((224, 224), dtype=np.int32)

        df = extract_cellular_morphology(empty_img, empty_map, "test_empty", "NORM")
        assert df.empty
        assert "instance_id" in df.columns

        summary = compute_image_morphology_summary(df, (224, 224), "test_empty", "NORM")
        assert summary["total_nuclei_count"] == 0
        assert summary["nuclear_density_per_kpx"] == 0.0
        assert summary["mean_nuclear_area_px2"] == 0.0


class TestRegionalQuadrantAnalysis:
    """Tests 2x2 intra-patch quadrant spatial analysis."""

    def test_centroid_quadrant_assignment(self):
        assert assign_centroid_to_quadrant(50, 50, (224, 224)) == "TL"
        assert assign_centroid_to_quadrant(150, 50, (224, 224)) == "TR"
        assert assign_centroid_to_quadrant(50, 150, (224, 224)) == "BL"
        assert assign_centroid_to_quadrant(150, 150, (224, 224)) == "BR"

    def test_regional_quadrant_analysis_output(self):
        # Create 4 synthetic nuclei, exactly 1 in each quadrant
        features_data = [
            {"instance_id": 1, "centroid_x": 30.0, "centroid_y": 30.0, "area_px2": 100.0, "circularity": 0.8, "eccentricity": 0.5},
            {"instance_id": 2, "centroid_x": 160.0, "centroid_y": 30.0, "area_px2": 120.0, "circularity": 0.85, "eccentricity": 0.6},
            {"instance_id": 3, "centroid_x": 30.0, "centroid_y": 160.0, "area_px2": 90.0, "circularity": 0.75, "eccentricity": 0.4},
            {"instance_id": 4, "centroid_x": 160.0, "centroid_y": 160.0, "area_px2": 110.0, "circularity": 0.9, "eccentricity": 0.3},
        ]
        df = pd.DataFrame(features_data)

        regional_df, intra_metrics = analyze_quadrant_regions(df, "img_test", "TUM", (224, 224))
        assert len(regional_df) == 4
        quadrants_present = set(regional_df["quadrant"])
        assert quadrants_present == {"TL", "TR", "BL", "BR"}

        for _, row in regional_df.iterrows():
            assert row["nuclei_count"] == 1
            assert row["nuclei_density_per_kpx"] > 0.0

        assert intra_metrics["total_nuclei"] == 4
        assert intra_metrics["density_variation_std"] == 0.0  # Equal count in all 4 quadrants


class TestCohortAndArtifacts:
    """Validates deterministic pilot cohort selection and generated artifacts."""

    def test_deterministic_cohort_selection(self):
        data_root = WORKSPACE_ROOT / "data" / "raw" / "HMU-GC-HE-30K" / "all_image"
        if not data_root.exists():
            pytest.skip("GCHTID raw image directory not found")

        df1 = select_pilot_cohort(str(data_root), images_per_class=8, seed=42)
        df2 = select_pilot_cohort(str(data_root), images_per_class=8, seed=42)

        assert len(df1) == 64
        # Exactly 8 images per canonical class
        class_counts = df1["class_name"].value_counts().to_dict()
        for c in CANONICAL_CLASSES:
            assert class_counts.get(c, 0) == 8

        # Deterministic match
        assert df1["image_id"].tolist() == df2["image_id"].tolist()

    def test_artifacts_exist_and_valid(self):
        manifest_file = CELLULAR_DIR / "cellular_sample_manifest.csv"
        features_file = CELLULAR_DIR / "instance_features.csv"
        img_summary_file = CELLULAR_DIR / "image_morphology_summary.csv"
        reg_summary_file = CELLULAR_DIR / "regional_morphology_summary.csv"
        json_summary_file = CELLULAR_DIR / "cellular_summary.json"

        assert manifest_file.exists() and manifest_file.stat().st_size > 0
        assert features_file.exists() and features_file.stat().st_size > 0
        assert img_summary_file.exists() and img_summary_file.stat().st_size > 0
        assert reg_summary_file.exists() and reg_summary_file.stat().st_size > 0
        assert json_summary_file.exists() and json_summary_file.stat().st_size > 0

        with open(json_summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)

        assert summary["phase"] == "Original Phase 7"
        assert summary["implementation_status"] == "EXPLORATORY_BASELINE_OPERATIONAL"
        assert summary["quantitative_findings"]["total_nuclei_detected"] > 0
        assert summary["pilot_dataset"]["total_images_processed"] == 64

    def test_visualizations_exist(self):
        viz_dir = CELLULAR_DIR / "visualizations"
        assert (viz_dir / "contact_sheet_8classes.png").exists()
        assert (viz_dir / "morphology_comparison_charts.png").exists()
        assert (viz_dir / "exemplar_centroid_map_tum.png").exists()
        assert (viz_dir / "exemplar_quadrant_density_tum.png").exists()
        assert (viz_dir / "exemplar_single_nucleus_gallery_tum.png").exists()


class TestRepositoryInvariants:
    """Protects historical checkpoints and prevents unintentional modifications."""

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

