"""QuPath GeoJSON Annotation Ingestion and Rasterization Pipeline.

Parses QuPath-exported GeoJSON FeatureCollections, validates coordinates against image bounds,
resolves multi-class overlaps, and rasterizes polygon/multipolygon geometries into pixel masks.

Explicitly implements the 9-class histopathology segmentation ontology:
Class 0: Background / Lumen
Class 1: ADI (Adipose tissue)
Class 2: DEB (Debris / necrosis)
Class 3: LYM (Lymphocytes / inflammatory infiltrate)
Class 4: MUC (Extracellular mucus)
Class 5: MUS (Smooth muscle)
Class 6: NOR (Normal gastric mucosa)
Class 7: STR (Cancer-associated stroma)
Class 8: TUM (Tumor adenocarcinoma epithelium)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("tissue_segmentation.annotations")

# Detailed Canonical 9-Class Segmentation Taxonomy
CANONICAL_SEGMENTATION_TAXONOMY: Dict[int, Dict[str, Any]] = {
    0: {
        "code": "BG",
        "name": "Background",
        "description": "Background, lumen, glass, optical void non-tissue space",
        "synonyms": ["Background", "BG", "Lumen", "Glass", "Void", "Non-tissue"],
    },
    1: {
        "code": "ADI",
        "name": "Adipose",
        "description": "Adipose tissue, adipocytes, lipid vacuoles",
        "synonyms": ["ADI", "Adipose", "Adipose Tissue", "Fat", "Adipocytes"],
    },
    2: {
        "code": "DEB",
        "name": "Debris",
        "description": "Necrosis, cellular debris, karyorrhectic fragments",
        "synonyms": ["DEB", "Debris", "Necrosis", "Cellular Debris", "Necrotic"],
    },
    3: {
        "code": "LYM",
        "name": "Lymphocytes",
        "description": "Lymphocytes, lymphoid aggregates, inflammatory infiltrate",
        "synonyms": ["LYM", "Lymphocytes", "Lymphoid Infiltrate", "Inflammatory", "Inflammatory Infiltrate", "Immune Infiltrate", "INF"],
    },
    4: {
        "code": "MUC",
        "name": "Mucus",
        "description": "Extracellular mucus, mucin pools, goblet cell secretion",
        "synonyms": ["MUC", "Mucus", "Extracellular Mucus", "Mucin", "Mucin Pool"],
    },
    5: {
        "code": "MUS",
        "name": "Muscle",
        "description": "Smooth muscle, muscularis propria, muscularis mucosae",
        "synonyms": ["MUS", "Muscle", "Smooth Muscle", "Muscularis Propria", "Muscularis Mucosae"],
    },
    6: {
        "code": "NOR",
        "name": "Normal",
        "description": "Normal gastric mucosa, benign gastric epithelium and non-neoplastic glands",
        "synonyms": ["NOR", "NORM", "Normal", "Normal Mucosa", "Benign Mucosa", "Benign Glands"],
    },
    7: {
        "code": "STR",
        "name": "Stroma",
        "description": "Cancer-associated stroma, desmoplastic stroma, reactive fibrous tissue",
        "synonyms": ["STR", "Stroma", "Cancer-Associated Stroma", "Desmoplasia", "Desmoplastic Stroma", "Fibrous Stroma"],
    },
    8: {
        "code": "TUM",
        "name": "Tumor",
        "description": "Malignant gastric adenocarcinoma, neoplastic epithelium, poorly/moderately differentiated carcinoma",
        "synonyms": ["TUM", "Tumor", "Tumour", "Adenocarcinoma", "Carcinoma", "Malignant Epithelium", "Gastric Adenocarcinoma"],
    },
}

# Exhaustive mapping from string synonyms to integer class IDs
DEFAULT_CLASS_MAPPING: Dict[str, int] = {
    # 0: Background
    "Background": 0,
    "BG": 0,
    "Lumen": 0,
    "Glass": 0,
    "Void": 0,
    # 1: ADI
    "ADI": 1,
    "Adipose": 1,
    "Adipose Tissue": 1,
    "Fat": 1,
    # 2: DEB
    "DEB": 2,
    "Debris": 2,
    "Necrosis": 2,
    "Cellular Debris": 2,
    # 3: LYM (Lymphocytes / Inflammatory Infiltrate)
    "LYM": 3,
    "Lymphocytes": 3,
    "Lymphoid Infiltrate": 3,
    "Inflammatory": 3,
    "Inflammatory Infiltrate": 3,
    "Immune Infiltrate": 3,
    # 4: MUC
    "MUC": 4,
    "Mucus": 4,
    "Extracellular Mucus": 4,
    "Mucin": 4,
    "Mucin Pool": 4,
    # 5: MUS
    "MUS": 5,
    "Muscle": 5,
    "Smooth Muscle": 5,
    "Muscularis Propria": 5,
    # 6: NOR
    "NOR": 6,
    "NORM": 6,
    "Normal": 6,
    "Normal Mucosa": 6,
    "Benign Mucosa": 6,
    "Benign Glands": 6,
    # 7: STR
    "STR": 7,
    "Stroma": 7,
    "Cancer-Associated Stroma": 7,
    "Desmoplasia": 7,
    "Desmoplastic Stroma": 7,
    # 8: TUM
    "TUM": 8,
    "Tumor": 8,
    "Tumour": 8,
    "Carcinoma": 8,
    "Adenocarcinoma": 8,
    "Malignant Epithelium": 8,
}

# Distinct visualization colors (RGB) for each class index
CLASS_COLORS_RGB: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),         # Background (Transparent/Black)
    1: (255, 230, 150),   # ADI (Adipose - Soft Cream / Light Yellow)
    2: (128, 128, 128),   # DEB (Debris - Medium Grey)
    3: (255, 215, 0),     # LYM (Lymphocytes / Inflammatory - Gold/Yellow)
    4: (255, 140, 0),     # MUC (Mucus - Orange)
    5: (147, 112, 219),   # MUS (Muscle - Medium Purple)
    6: (30, 144, 255),    # NOR (Normal Mucosa - Dodger Blue)
    7: (46, 139, 87),     # STR (Stroma - Sea Green)
    8: (220, 20, 60),     # TUM (Tumor - Crimson Red)
}

# Default Overlap Priority: Later classes overwrite earlier classes
DEFAULT_OVERLAP_PRIORITY: List[str] = [
    "Background",
    "ADI",
    "NOR",
    "MUS",
    "DEB",
    "STR",
    "MUC",
    "LYM",
    "TUM",
]


class QuPathAnnotationParser:
    """Parses and validates QuPath GeoJSON annotations for histopathology tiles."""

    def __init__(
        self,
        class_mapping: Optional[Dict[str, int]] = None,
        image_size: Tuple[int, int] = (224, 224),
        overlap_priority: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            class_mapping: Dict mapping string class names to integer labels.
            image_size: Target (height, width) of the corresponding tile.
            overlap_priority: Order in which classes are rasterized (later overwrites earlier).
        """
        self.class_mapping = class_mapping or DEFAULT_CLASS_MAPPING
        self.height, self.width = image_size
        self.overlap_priority = overlap_priority or DEFAULT_OVERLAP_PRIORITY

    def resolve_class_index(self, name: str) -> Optional[int]:
        """Resolves raw class name string (case-insensitive) to canonical integer class ID."""
        if not name:
            return None
        cleaned = str(name).strip()
        if cleaned in self.class_mapping:
            return self.class_mapping[cleaned]
        for k, v in self.class_mapping.items():
            if cleaned.lower() == k.lower():
                return v
        return None

    def parse_geojson(self, geojson_path: Union[str, Path]) -> Dict[str, Any]:
        """Loads and verifies a QuPath GeoJSON file."""
        p = Path(geojson_path)
        if not p.exists():
            raise FileNotFoundError(f"GeoJSON file not found: {p}")

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid GeoJSON structure in {p}: root must be a JSON object")

        data_type = data.get("type", "")
        if data_type == "FeatureCollection":
            features = data.get("features", [])
        elif data_type == "Feature":
            features = [data]
        else:
            raise ValueError(f"Unsupported GeoJSON type '{data_type}' in {p}")

        logger.info("Loaded GeoJSON %s with %d features", p.name, len(features))
        return {"source_path": str(p), "features": features}

    def extract_classification_name(self, properties: Dict[str, Any]) -> str:
        """Extracts class name from QuPath feature properties dictionary."""
        classification = properties.get("classification")
        if isinstance(classification, dict):
            name = classification.get("name", "Unknown")
        elif isinstance(classification, str):
            name = classification
        else:
            name = properties.get("name", "Unknown")
        return str(name).strip()

    def validate_polygon_coordinates(
        self, coordinates: List[Any]
    ) -> Tuple[bool, Optional[np.ndarray], List[str]]:
        """
        Validates polygon coordinate rings.
        Coordinates in GeoJSON are [ [ [x1, y1], [x2, y2], ... ] ] (exterior ring + holes).
        Returns:
            is_valid: bool
            np_coords: np.ndarray of shape (N, 2) in integer pixel coordinates (x, y)
            issues: list of warning/error descriptions
        """
        issues = []
        if not coordinates or not isinstance(coordinates, list):
            return False, None, ["Coordinates must be a non-empty list"]

        # Exterior ring is first element
        exterior = coordinates[0]
        if not isinstance(exterior, list) or len(exterior) < 3:
            return False, None, [f"Exterior ring must contain at least 3 vertices, got {len(exterior)}"]

        # Extract numeric pairs
        pts = []
        for idx, pt in enumerate(exterior):
            if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                issues.append(f"Vertex {idx} is not an [x, y] pair: {pt}")
                continue
            x, y = pt[0], pt[1]
            if not np.isfinite(x) or not np.isfinite(y):
                issues.append(f"Vertex {idx} contains non-finite values: ({x}, {y})")
                continue
            pts.append([float(x), float(y)])

        if len(pts) < 3:
            return False, None, ["Fewer than 3 valid numeric vertices: " + "; ".join(issues)]

        arr = np.array(pts, dtype=np.float32)

        # Coordinate bounds check for 224x224 tile
        min_x, min_y = np.min(arr, axis=0)
        max_x, max_y = np.max(arr, axis=0)

        if min_x < 0 or min_y < 0:
            issues.append(f"Negative coordinates detected: min_x={min_x:.1f}, min_y={min_y:.1f}")
        if max_x > self.width or max_y > self.height:
            issues.append(
                f"Coordinates exceed image bounds ({self.width}x{self.height}): "
                f"max_x={max_x:.1f}, max_y={max_y:.1f}"
            )

        # Round to integer pixel grid
        int_pts = np.round(arr).astype(np.int32)
        return True, int_pts, issues

    def rasterize_annotations(
        self,
        features: List[Dict[str, Any]],
        strict_bounds: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Rasterizes a list of GeoJSON features into a 2D integer class mask (H, W).
        Returns:
            mask: np.ndarray of shape (H, W), dtype=np.uint8
            metadata: summary dictionary detailing parsed, skipped, and out-of-bounds annotations
        """
        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        parsed_count = 0
        skipped_count = 0
        feature_audit: List[Dict[str, Any]] = []

        # Sort features by overlap priority so higher priority overwrites lower
        def get_priority(feat: Dict[str, Any]) -> int:
            cls_name = self.extract_classification_name(feat.get("properties", {}))
            cls_idx = self.resolve_class_index(cls_name)
            if cls_idx is not None:
                for p_idx, p_name in enumerate(self.overlap_priority):
                    p_cls_idx = self.resolve_class_index(p_name)
                    if p_cls_idx == cls_idx or p_name.lower() == cls_name.lower():
                        return p_idx
            return -1

        sorted_features = sorted(features, key=get_priority)

        for feat in sorted_features:
            geom = feat.get("geometry", {})
            props = feat.get("properties", {})
            geom_type = geom.get("type", "")
            cls_name = self.extract_classification_name(props)
            cls_idx = self.resolve_class_index(cls_name)

            if cls_idx is None:
                skipped_count += 1
                feature_audit.append({
                    "id": feat.get("id", "unknown"),
                    "status": "skipped",
                    "reason": f"Class '{cls_name}' not in declared class mapping",
                })
                continue

            coords = geom.get("coordinates", [])

            # Handle Polygon vs MultiPolygon
            polygon_list = []
            if geom_type == "Polygon":
                polygon_list.append(coords)
            elif geom_type == "MultiPolygon":
                polygon_list.extend(coords)
            else:
                skipped_count += 1
                feature_audit.append({
                    "id": feat.get("id", "unknown"),
                    "status": "skipped",
                    "reason": f"Unsupported geometry type: '{geom_type}'",
                })
                continue

            feature_valid = True
            for poly_coords in polygon_list:
                valid, int_pts, issues = self.validate_polygon_coordinates(poly_coords)
                if not valid:
                    feature_valid = False
                    skipped_count += 1
                    feature_audit.append({
                        "id": feat.get("id", "unknown"),
                        "status": "invalid_geometry",
                        "issues": issues,
                    })
                    break

                if issues and strict_bounds:
                    feature_valid = False
                    skipped_count += 1
                    feature_audit.append({
                        "id": feat.get("id", "unknown"),
                        "status": "out_of_bounds_rejected",
                        "issues": issues,
                    })
                    break

                # Clip coordinates to canvas boundaries
                clipped_pts = np.clip(int_pts, [0, 0], [self.width - 1, self.height - 1])
                # Rasterize filled polygon onto mask
                cv2.fillPoly(mask, [clipped_pts], color=int(cls_idx))

            if feature_valid:
                parsed_count += 1
                feature_audit.append({
                    "id": feat.get("id", "unknown"),
                    "status": "rasterized",
                    "class_name": cls_name,
                    "class_index": cls_idx,
                })

        # Count pixels per canonical class
        class_pixel_counts: Dict[str, int] = {}
        for c_idx, c_info in CANONICAL_SEGMENTATION_TAXONOMY.items():
            class_pixel_counts[c_info["code"]] = int(np.sum(mask == c_idx))

        total_pixels = self.height * self.width
        summary = {
            "total_features_evaluated": len(features),
            "successfully_rasterized": parsed_count,
            "skipped_or_rejected": skipped_count,
            "mask_shape": [self.height, self.width],
            "class_pixel_counts": class_pixel_counts,
            "class_area_fractions": {
                c: round(cnt / total_pixels, 4) for c, cnt in class_pixel_counts.items()
            },
            "feature_audit": feature_audit,
        }
        return mask, summary

    @staticmethod
    def create_synthetic_test_feature_collection(
        image_id: str = "TUM/TUM_1001.png",
    ) -> Dict[str, Any]:
        """
        Creates a verified synthetic QuPath GeoJSON FeatureCollection.
        NOTE: This is strictly a software unit-testing fixture to verify coordinate validation
        and rasterization pipelines. It is NEVER used as research data or training annotations.
        """
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": "annot-test-adi-01",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[10.0, 10.0], [45.0, 10.0], [45.0, 45.0], [10.0, 45.0], [10.0, 10.0]]
                        ],
                    },
                    "properties": {
                        "classification": {"name": "ADI", "color": [255, 230, 150]},
                        "image_id": image_id,
                    },
                },
                {
                    "type": "Feature",
                    "id": "annot-test-tumor-01",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[50.0, 50.0], [150.0, 50.0], [150.0, 150.0], [50.0, 150.0], [50.0, 50.0]]
                        ],
                    },
                    "properties": {
                        "classification": {"name": "TUM", "color": [220, 20, 60]},
                        "image_id": image_id,
                    },
                },
                {
                    "type": "Feature",
                    "id": "annot-test-stroma-01",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[160.0, 50.0], [210.0, 50.0], [210.0, 100.0], [160.0, 100.0], [160.0, 50.0]]
                        ],
                    },
                    "properties": {
                        "classification": {"name": "STR", "color": [46, 139, 87]},
                        "image_id": image_id,
                    },
                },
            ],
        }

    @staticmethod
    def get_qupath_export_instructions() -> str:
        """Returns clear, actionable instructions for exporting annotations from QuPath."""
        return (
            "=== QuPath Annotation Export Workflow for GastricPath AI ===\n"
            "1. In QuPath, open the 224x224 histopathology tile (or define a 224x224 ROI on a WSI).\n"
            "2. Use the Brush or Wand tool to outline tissue compartments.\n"
            "3. Assign classes matching our canonical 8-tissue taxonomy (plus Background):\n"
            "   - ADI (Adipose), DEB (Debris/Necrosis), LYM (Lymphocytes/Inflammatory),\n"
            "   - MUC (Mucus), MUS (Muscle), NOR (Normal Mucosa), STR (Stroma), TUM (Tumor).\n"
            "4. Go to File -> Export -> Export GeoJSON.\n"
            "5. Select 'FeatureCollection' format with 'Export annotations' checked.\n"
            "6. Name the exported file with the corresponding tile image ID (e.g., TUM_1001.geojson)\n"
            "   and place it in `data/annotations/qupath/`.\n"
            "7. The GastricPath AI QuPathAnnotationParser will automatically validate and rasterize it."
        )


def generate_synthetic_geojson_fixture(image_id: str = "TUM/TUM_1001.png") -> Dict[str, Any]:
    """Convenience functional wrapper for creating synthetic GeoJSON software test fixture."""
    return QuPathAnnotationParser.create_synthetic_test_feature_collection(image_id=image_id)


def rasterize_qupath_geojson(
    geojson_data: Dict[str, Any],
    target_size: Tuple[int, int] = (224, 224),
    class_mapping: Optional[Dict[str, int]] = None,
    strict_bounds: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convenience functional wrapper to parse and rasterize GeoJSON data dict directly.
    """
    parser = QuPathAnnotationParser(class_mapping=class_mapping, image_size=target_size)
    features = geojson_data.get("features", [])
    return parser.rasterize_annotations(features, strict_bounds=strict_bounds)
