"""
Histopathology Image Ingestion and Metadata Extraction.
GastricPath AI — Phase 3: Document and Image Ingestion.

Preserves genuine image metadata without fabricating optical magnification,
pixel spacing, physical dimensions, or scale bars.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from PIL import Image

from research.clinical_reporting.security.file_validation import validate_image_file


@dataclass
class ImageEvidenceRecord:
    sample_id: str
    original_filename: str
    file_path: str
    sha256_hash: str
    file_format: str
    width: int
    height: int
    bit_depth: int
    stain_type: str = "Hematoxylin & Eosin (H&E)"
    scanner_manufacturer: str = "Unknown / Unrecorded in File Header"
    optical_magnification: Optional[str] = None  # None indicates unknown; never fabricate 20x
    pixel_spacing_um: Optional[float] = None     # None indicates unknown; never fabricate 0.5 um/px
    dataset_source: str = "Local Upload / HMU-GC-HE-30K"
    known_class: Optional[str] = None
    verification_status: str = "Raw Image Verified"
    notes: Optional[str] = None

    def display_optical_spec(self) -> str:
        """Renders verified optical specs or explicit unknown indicators."""
        mag = self.optical_magnification if self.optical_magnification else "[Uncalibrated / Unknown]"
        res = f"{self.pixel_spacing_um} um/px" if self.pixel_spacing_um else "[Uncalibrated / Unknown]"
        return f"Magnification: {mag} | Resolution: {res}"


def ingest_histology_image(
    image_path: Path,
    sample_id: Optional[str] = None,
    dataset_source: str = "Local Clinical / Research Upload",
    known_class: Optional[str] = None
) -> ImageEvidenceRecord:
    """Ingests an H&E image, computing cryptographic hash and extracting genuine metadata."""
    validate_image_file(image_path)
    
    with open(image_path, "rb") as f:
        file_bytes = f.read()
        sha256_hash = hashlib.sha256(file_bytes).hexdigest()
        
    with Image.open(image_path) as img:
        width, height = img.size
        img_format = img.format or image_path.suffix.upper().replace(".", "")
        mode = img.mode
        bit_depth = 8 * len(mode) if mode else 24

    if not sample_id:
        sample_id = f"IMG-{sha256_hash[:8].upper()}"

    return ImageEvidenceRecord(
        sample_id=sample_id,
        original_filename=image_path.name,
        file_path=str(image_path.resolve()),
        sha256_hash=sha256_hash,
        file_format=img_format,
        width=width,
        height=height,
        bit_depth=bit_depth,
        dataset_source=dataset_source,
        known_class=known_class,
        notes="Authentic image metadata verified without synthetic optical magnification."
    )

