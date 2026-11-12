"""GastricPath AI — Cellular Morphology and Nuclei Detection Research Module.

Provides model adapters, instance segmentation post-processing, mathematical
cellular morphology extraction, intra-patch 2x2 regional spatial analysis,
and publication visualization tools.
"""

from __future__ import annotations

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
    extract_cellular_morphology,
    compute_image_morphology_summary,
)
from research.cellular.region_analysis import (
    analyze_quadrant_regions,
    assign_nuclei_to_quadrants,
)
from research.cellular.visualization import (
    render_instance_overlay,
    render_centroid_map,
    render_quadrant_density_map,
)

__all__ = [
    "NucleiDetectorAdapter",
    "StarDistAdapter",
    "CellposeAdapter",
    "HoVerNetAdapter",
    "ClassicalHEDWatershedAdapter",
    "PretrainedModelUnavailableError",
    "get_available_detector",
    "extract_cellular_morphology",
    "compute_image_morphology_summary",
    "analyze_quadrant_regions",
    "assign_nuclei_to_quadrants",
    "render_instance_overlay",
    "render_centroid_map",
    "render_quadrant_density_map",
]

