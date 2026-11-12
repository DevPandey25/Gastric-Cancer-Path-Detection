"""Whole-Slide Image (WSI) Processing and Multiple Instance Learning (MIL) Engine.

Original Phase 9 research package:
- Slide reading and OpenSlide adapter with Level-0 reference coordinate invariance.
- Structured slide metadata extraction and physical resolution preservation.
- Lightweight thumbnail-based tissue detection and O(1) integral region evaluation.
- Coordinate-aware patch tiling, boundary handling, and PatchManifest persistence.
- Foundation model embedding integration interface.
- Multiple Instance Learning architectures (GatedAttentionMIL, MeanPoolingMILBaseline).
- Slide-level attention heatmap reconstruction.
"""

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
    SlideReadError,
    SyntheticPyramidSlideReader,
    open_slide,
)
from research.wsi.tissue_detection import TissueDetectionConfig, TissueDetector

__all__ = [
    "SlideMetadata",
    "parse_openslide_properties",
    "BaseSlideReader",
    "OpenSlideAdapter",
    "SyntheticPyramidSlideReader",
    "OpenSlideUnavailableError",
    "SlideReadError",
    "open_slide",
    "TissueDetector",
    "TissueDetectionConfig",
    "CoordinateAwarePatchExtractor",
    "PatchExtractionConfig",
    "PatchRecord",
    "PatchManifest",
    "WSIEmbeddingExtractor",
    "WSIEmbeddingsBlockedError",
    "GatedAttentionMIL",
    "MeanPoolingMILBaseline",
    "SlideBag",
    "MILTrainingBlockedError",
    "build_mil_model",
    "SlideAttentionHeatmapGenerator",
]

