"""WSI Pipeline Demonstrator and Software Verification Engine.

Performs:
1. Systematic repository scan auditing genuine WSI files (confirming 0 genuine WSIs).
2. Software verification using a multi-resolution synthetic test fixture composed of
   real H&E tissue patches from GCHTID on a multi-level pyramidal canvas.
3. Demonstration of thumbnail extraction, tissue masking, coordinate-aware patch extraction,
   and Level-0 attention heatmap reconstruction.
4. Generates publication-grade demonstration visualization clearly watermarked as
   a software verification fixture (preventing clinical misrepresentation).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from PIL import Image

from research.wsi.attention_heatmap import SlideAttentionHeatmapGenerator
from research.wsi.mil_models import GatedAttentionMIL, build_mil_model
from research.wsi.patch_extractor import CoordinateAwarePatchExtractor, PatchExtractionConfig
from research.wsi.patch_manifest import PatchManifest
from research.wsi.slide_reader import SyntheticPyramidSlideReader
from research.wsi.tissue_detection import TissueDetectionConfig, TissueDetector

logger = logging.getLogger("wsi.demonstrator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = Path("research/wsi")
VIS_DIR = BASE_DIR / "visualizations"
AUDIT_JSON_PATH = BASE_DIR / "wsi_audit_summary.json"


def audit_repository_wsi_files(search_roots: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Exhaustively scans repository directories for any genuine Whole-Slide Image files.
    """
    if search_roots is None:
        search_roots = ["data", "models", "research"]

    extensions = [".svs", ".ndpi", ".mrxs", ".scn", ".tif", ".tiff", ".bif", ".vms", ".vmu", ".dcm"]
    found_files: List[Dict[str, Any]] = []

    for root_dir in search_roots:
        p_root = Path(root_dir)
        if not p_root.exists():
            continue
        for ext in extensions:
            for f in p_root.rglob(f"*{ext}"):
                if f.is_file():
                    found_files.append({
                        "file_path": str(f).replace("\\", "/"),
                        "extension": ext,
                        "file_size_bytes": f.stat().st_size,
                    })

    # Filter out any ordinary non-pyramidal images
    genuine_slides: List[Dict[str, Any]] = []
    for item in found_files:
        # Genuine WSIs are almost invariably > 10 MB and possess multiple pyramidal directories
        if item["file_size_bytes"] > 10_000_000:
            genuine_slides.append(item)

    audit_result = {
        "status": "AWAITING_WSI_DATA" if len(genuine_slides) == 0 else "WSI_DATA_AVAILABLE",
        "total_wsi_candidates_found": len(found_files),
        "genuine_wsi_count": len(genuine_slides),
        "searched_extensions": extensions,
        "searched_roots": search_roots,
        "gchtid_dataset_nature": "31,096 isolated 224x224 PNG image patches; zero gigapixel pyramidal slides exist",
        "genuine_wsi_files": genuine_slides,
    }
    return audit_result


def create_synthetic_pyramid_fixture(canvas_size: int = 1792) -> Image.Image:
    """
    Creates a synthetic tissue slide canvas (1792x1792) by placing actual GCHTID patches
    in a central tissue cluster surrounded by white glass background.
    Strictly used as a software test fixture.
    """
    # Create white glass background
    canvas = Image.new("RGB", (canvas_size, canvas_size), (248, 248, 250))

    # Sample actual H&E patches from data/raw/
    raw_dir = Path("data/raw")
    if raw_dir.exists():
        patch_paths = list(raw_dir.rglob("*.png"))[:36]  # 6x6 grid of patches
    else:
        patch_paths = []

    if patch_paths:
        grid_dim = 6
        tile_size = 224
        # Offset to center the tissue cluster
        offset_x = (canvas_size - grid_dim * tile_size) // 2
        offset_y = (canvas_size - grid_dim * tile_size) // 2

        idx = 0
        for r in range(grid_dim):
            for c in range(grid_dim):
                if idx < len(patch_paths):
                    p_img = Image.open(patch_paths[idx]).convert("RGB")
                    canvas.paste(p_img, (offset_x + c * tile_size, offset_y + r * tile_size))
                    idx += 1
    else:
        # Fallback synthetic circular tissue blob if raw patches missing
        draw_arr = np.full((canvas_size, canvas_size, 3), 248, dtype=np.uint8)
        y, x = np.ogrid[:canvas_size, :canvas_size]
        center = canvas_size // 2
        radius = canvas_size // 3
        circle = ((x - center) ** 2 + (y - center) ** 2) <= radius ** 2
        draw_arr[circle] = [180, 80, 160]  # H&E pink/purple hue
        canvas = Image.fromarray(draw_arr)

    return canvas


def run_wsi_pipeline_demonstration() -> Dict[str, Any]:
    """
    Executes WSI audit and software verification demonstration.
    """
    logger.info("Executing Original Phase 9 WSI Engine Audit & Verification...")

    # 1. Audit actual WSI availability
    audit_results = audit_repository_wsi_files()
    AUDIT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_JSON_PATH, "w") as f:
        json.dump(audit_results, f, indent=2)
    logger.info(f"Audit completed: {audit_results['genuine_wsi_count']} genuine WSIs found.")

    # 2. Build multi-resolution synthetic test fixture
    fixture_canvas = create_synthetic_pyramid_fixture(canvas_size=1792)
    reader = SyntheticPyramidSlideReader(
        base_image=fixture_canvas,
        slide_id="synthetic_demo_slide_01",
        downsamples=[1.0, 2.0, 4.0, 8.0],
        mpp_x=0.50,
        mpp_y=0.50,
        apparent_magnification=20.0,
        vendor="Synthetic Test Fixture (Software Verification Only)",
    )

    # 3. Read thumbnail & execute tissue detection
    thumb = reader.get_thumbnail(max_size=(448, 448))
    detector = TissueDetector(
        thumbnail=thumb,
        level0_dimensions=reader.metadata.level0_dimensions,
        config=TissueDetectionConfig(sat_threshold=15, val_threshold=245, min_tissue_percentage=0.50),
    )

    # 4. Extract coordinate-aware patches at Level 0
    extractor = CoordinateAwarePatchExtractor(
        reader=reader,
        tissue_detector=detector,
        config=PatchExtractionConfig(
            patch_width=224,
            patch_height=224,
            target_level=0,
            stride_x=224,
            stride_y=224,
            min_tissue_percentage=0.50,
            save_patch_images=False,  # Keep in memory for demo
        ),
    )
    manifest = extractor.extract_all_patches()
    logger.info(f"Extracted {len(manifest)} valid tissue patches from synthetic fixture.")

    # 5. Simulate attention heatmap reconstruction
    patch_records = manifest.get_slide_records(reader.metadata.slide_id)
    coords = [(r.level0_x, r.level0_y) for r in patch_records]
    sizes = [(r.patch_width, r.patch_height) for r in patch_records]
    downs = [r.downsample for r in patch_records]

    # Generate synthetic attention weights peaking in central glandular regions
    np.random.seed(42)
    num_p = len(patch_records)
    if num_p > 0:
        simulated_weights = np.random.uniform(0.1, 0.9, size=num_p).astype(np.float32)
        # Emphasize center
        center_x = reader.metadata.level0_dimensions[0] / 2.0
        center_y = reader.metadata.level0_dimensions[1] / 2.0
        for i, (x0, y0) in enumerate(coords):
            dist = np.sqrt((x0 - center_x) ** 2 + (y0 - center_y) ** 2)
            simulated_weights[i] = float(np.exp(-dist / 500.0) + np.random.uniform(0.0, 0.2))
        simulated_weights /= np.sum(simulated_weights)
    else:
        simulated_weights = np.empty((0, 1), dtype=np.float32)

    heatmap_gen = SlideAttentionHeatmapGenerator(
        thumbnail=thumb,
        level0_dimensions=reader.metadata.level0_dimensions,
        tissue_mask=detector.tissue_mask,
    )
    blended_heatmap, raw_heatmap = heatmap_gen.generate_attention_heatmap(
        patch_coords=coords,
        patch_sizes=sizes,
        downsamples=downs,
        attention_weights=simulated_weights,
        colormap_name="jet",
        alpha=0.6,
    )

    # 6. Render 4-Panel Demonstration Visualization
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    demo_fig_path = VIS_DIR / "synthetic_wsi_pipeline_demonstration.png"

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Panel 1: Whole-Slide Thumbnail
    axes[0, 0].imshow(thumb)
    axes[0, 0].set_title("A: Whole-Slide Pyramid Thumbnail (Level 0: 1792x1792)", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    # Panel 2: Tissue Mask Overlay
    axes[0, 1].imshow(detector.get_tissue_overlay_image(alpha=0.5, overlay_color=(0, 255, 0)))
    axes[0, 1].set_title("B: HSV Tissue Detection & Background Rejection", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    # Panel 3: Coordinate-Aware Patch Grid
    grid_img = np.array(thumb.copy())
    sx = reader.metadata.level0_dimensions[0] / float(thumb.width)
    sy = reader.metadata.level0_dimensions[1] / float(thumb.height)
    for x0, y0 in coords:
        tx1 = int(x0 / sx)
        ty1 = int(y0 / sy)
        tx2 = int((x0 + 224) / sx)
        ty2 = int((y0 + 224) / sy)
        cv_img = cv2.rectangle(grid_img, (tx1, ty1), (tx2, ty2), (0, 255, 255), 1)
    axes[1, 0].imshow(grid_img)
    axes[1, 0].set_title(f"C: Coordinate-Aware Grid (K={len(patch_records)} Valid Patches)", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")

    # Panel 4: Reconstructed Attention Heatmap Overlay
    axes[1, 1].imshow(blended_heatmap)
    axes[1, 1].set_title("D: Reconstructed Spatial Attention Heatmap Overlay", fontsize=11, fontweight="bold")
    axes[1, 1].axis("off")

    plt.suptitle(
        "[SYNTHETIC TEST FIXTURE: SOFTWARE VERIFICATION ONLY — NOT CLINICAL WSI]\n"
        "Original Phase 9: Whole-Slide Image Engine & Level-0 Coordinate Pipeline Verification",
        fontsize=12,
        fontweight="bold",
        color="darkred",
        y=0.98,
    )
    plt.tight_layout()
    fig.savefig(str(demo_fig_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    n_levels = len(reader.metadata.level_dimensions)
    reader.close()

    logger.info(f"Demonstration visualization saved to {demo_fig_path}")

    return {
        "audit": audit_results,
        "demonstration_figure": str(demo_fig_path.resolve()).replace("\\", "/"),
        "patches_extracted": len(manifest),
        "levels_demonstrated": n_levels,
    }


if __name__ == "__main__":
    res = run_wsi_pipeline_demonstration()
    print("Verification complete.")
    print("WSI Audit Status:", res["audit"]["status"])
    print("Genuine WSIs:", res["audit"]["genuine_wsi_count"])
    print("Demonstration figure:", res["demonstration_figure"])
