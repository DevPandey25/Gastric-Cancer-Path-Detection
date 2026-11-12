"""GastricPath AI - Dataset Integrity, Leakage Detection & Manifest Module.

Provides reproducible dataset scanning, SHA-256 duplicate detection,
perceptual near-duplicate clustering, manifest generation, and deterministic
stratified splitting for histopathology research.
"""

from research.data.integrity_scanner import (
    DatasetIntegrityScanner,
    DatasetIntegrityReport,
    ImageIntegrityRecord,
)
from research.data.duplicate_detector import (
    ExactDuplicateDetector,
    ExactDuplicateReport,
    DuplicateGroup,
)
from research.data.leakage_detector import (
    NearDuplicateDetector,
    NearDuplicateReport,
    NearDuplicateCluster,
    compute_dhash,
    compute_ahash,
    hamming_distance,
)
from research.data.manifest_generator import (
    DatasetManifestGenerator,
    ManifestRecord,
)
from research.data.split_generator import (
    ReproducibleSplitGenerator,
    SplitSummary,
)

__all__ = [
    "DatasetIntegrityScanner",
    "DatasetIntegrityReport",
    "ImageIntegrityRecord",
    "ExactDuplicateDetector",
    "ExactDuplicateReport",
    "DuplicateGroup",
    "NearDuplicateDetector",
    "NearDuplicateReport",
    "NearDuplicateCluster",
    "compute_dhash",
    "compute_ahash",
    "hamming_distance",
    "DatasetManifestGenerator",
    "ManifestRecord",
    "ReproducibleSplitGenerator",
    "SplitSummary",
]

