"""
Security, File Validation, and Path Traversal Protection.
GastricPath AI — Phase 3: Document and Image Ingestion.
"""

from pathlib import Path
import re
from typing import Optional

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
ALLOWED_DOCUMENT_EXTENSIONS = {".json", ".csv", ".txt", ".pdf"}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB


class SecurityValidationError(Exception):
    pass


def sanitize_filename(filename: str) -> str:
    """Removes path traversal characters and illegal symbols."""
    # Strip any directory separators
    clean = Path(filename).name
    # Keep only safe alphanumeric, hyphen, underscore, and dot
    clean = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", clean)
    if not clean or clean.startswith("."):
        clean = f"safe_file_{clean}"
    return clean


def validate_file_path(base_dir: Path, target_path: Path) -> Path:
    """Ensures target_path resolves strictly within base_dir, preventing directory traversal."""
    resolved_base = base_dir.resolve()
    resolved_target = target_path.resolve()
    try:
        resolved_target.relative_to(resolved_base)
    except ValueError:
        raise SecurityValidationError(f"Path traversal detected: {target_path} is outside {base_dir}")
    return resolved_target


def validate_image_file(file_path: Path) -> None:
    """Validates file extension and size for incoming images."""
    if file_path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        raise SecurityValidationError(f"Unsupported image extension: {file_path.suffix}. Allowed: {ALLOWED_IMAGE_EXTENSIONS}")
    if file_path.stat().st_size > MAX_FILE_SIZE_BYTES:
        raise SecurityValidationError(f"File size exceeds maximum limit of {MAX_FILE_SIZE_BYTES} bytes.")

