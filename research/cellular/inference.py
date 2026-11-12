"""Pilot Nuclei Inference and Cellular Morphology Pipeline Coordinator.

Executes sequential, deterministic CPU inference across the 64-image GCHTID pilot:
- 8 images from each of the 8 canonical classes (seed 42).
- Extracts instance-level and image-level morphology descriptors.
- Computes 2x2 intra-patch quadrant spatial analysis.
- Generates contact sheets, individual nucleus galleries, and morphology comparison charts.
- Exports tabular CSVs and structured JSON summaries.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import psutil

from research.cellular.model_adapter import (
    ClassicalHEDWatershedAdapter,
    NucleiDetectorAdapter,
    get_available_detector,
)
from research.cellular.morphology import (
    compute_image_morphology_summary,
    extract_cellular_morphology,
)
from research.cellular.region_analysis import analyze_quadrant_regions
from research.cellular.visualization import (
    render_centroid_map,
    render_contact_sheet,
    render_instance_overlay,
    render_morphology_comparison_charts,
    render_quadrant_density_map,
    render_single_nucleus_gallery,
)

logger = logging.getLogger("cellular.inference")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CANONICAL_CLASSES = ["ADI", "DEB", "LYM", "MUC", "MUS", "NOR", "STR", "TUM"]


def select_pilot_cohort(
    data_root: str = "data/raw/HMU-GC-HE-30K/all_image",
    images_per_class: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Selects a deterministic, class-balanced cohort of GCHTID tiles.

    Args:
        data_root: Directory containing canonical class subfolders.
        images_per_class: Number of samples per class (default: 8).
        seed: Random seed for reproducibility (default: 42).

    Returns:
        pd.DataFrame: Manifest with columns ['image_id', 'class_name', 'image_path', 'pilot_index'].
    """
    root_path = Path(data_root)
    if not root_path.exists():
        raise FileNotFoundError(f"GCHTID data directory not found at: {root_path}")

    records = []
    pilot_idx = 0

    for c in CANONICAL_CLASSES:
        c_dir = root_path / c
        if not c_dir.exists():
            raise FileNotFoundError(f"Class directory not found: {c_dir}")

        image_files = sorted([
            f.name for f in c_dir.iterdir()
            if f.suffix.lower() in [".png", ".jpg", ".jpeg"]
        ])
        if len(image_files) < images_per_class:
            raise ValueError(f"Insufficient images in {c_dir}: found {len(image_files)}, need {images_per_class}")

        rng = random.Random(seed)
        selected_files = rng.sample(image_files, images_per_class)

        for fname in selected_files:
            rel_path = f"{c}/{fname}"
            full_path = str((c_dir / fname).resolve()).replace("\\", "/")
            records.append({
                "pilot_index": pilot_idx,
                "image_id": rel_path,
                "class_name": c,
                "filename": fname,
                "image_path": full_path,
            })
            pilot_idx += 1

    manifest_df = pd.DataFrame(records)
    logger.info("Selected %d pilot images across %d classes (seed %d)", len(manifest_df), len(CANONICAL_CLASSES), seed)
    return manifest_df


def run_pilot_inference(
    pilot_df: pd.DataFrame,
    output_dir: str = "research/cellular",
    detector: Optional[NucleiDetectorAdapter] = None,
    save_masks: bool = True,
    smoke_test_only: bool = False,
) -> Dict[str, Any]:
    """
    Runs sequential CPU nuclei instance detection and morphology feature extraction.

    Args:
        pilot_df: Cohort DataFrame from select_pilot_cohort.
        output_dir: Destination directory for artifacts.
        detector: Instantiated NucleiDetectorAdapter (defaults to ClassicalHEDWatershedAdapter).
        save_masks: Whether to persist uint16/int32 instance-label maps.
        smoke_test_only: If True, executes only the first 4 samples.

    Returns:
        Dict of execution summary and high-level statistics.
    """
    out_path = Path(output_dir)
    masks_dir = out_path / "instance_maps"
    viz_dir = out_path / "visualizations"
    masks_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    if detector is None:
        detector = get_available_detector()

    model_info = detector.get_model_info()
    logger.info("Using nuclei detector: %s (%s)", model_info["name"], model_info["model_variant"])

    process_df = pilot_df.head(4) if smoke_test_only else pilot_df

    all_instance_dfs: List[pd.DataFrame] = []
    image_summaries: List[Dict[str, Any]] = []
    regional_dfs: List[pd.DataFrame] = []

    # Exemplars for contact sheet (1 per class)
    exemplars_by_class: Dict[str, Dict[str, Any]] = {}

    start_time = time.time()
    mem_start = psutil.Process().memory_info().rss / (1024 * 1024)

    logger.info("Beginning sequential inference on %d images...", len(process_df))

    for idx, row in process_df.iterrows():
        t0 = time.time()
        img_id = row["image_id"]
        c_name = row["class_name"]
        p = Path(row["image_path"])

        # Load RGB image
        pil_img = Image.open(p).convert("RGB")
        img_np = np.array(pil_img, dtype=np.uint8)

        # Detect nuclei
        instance_map, meta = detector.detect_nuclei(img_np)
        n_detected = meta.get("detected_nuclei_count", 0)

        # Extract morphology
        inst_df = extract_cellular_morphology(
            img_np, instance_map, image_id=img_id, class_name=c_name
        )
        if not inst_df.empty:
            all_instance_dfs.append(inst_df)

        # Compute image summary
        img_summary = compute_image_morphology_summary(
            inst_df, image_shape=img_np.shape[:2], image_id=img_id, class_name=c_name
        )
        img_summary["elapsed_sec"] = round(time.time() - t0, 3)
        image_summaries.append(img_summary)

        # Compute 2x2 regional analysis
        reg_df, _ = analyze_quadrant_regions(
            inst_df, image_id=img_id, class_name=c_name, image_size=img_np.shape[:2]
        )
        regional_dfs.append(reg_df)

        # Save instance map if requested
        if save_masks:
            safe_name = img_id.replace("/", "_").replace(".png", "")
            mask_file = masks_dir / f"{safe_name}_instances.png"
            # Save as 16-bit PNG preserving exact instance IDs up to 65,535
            cv2.imwrite(str(mask_file), instance_map.astype(np.uint16))

        # Store first exemplar per class for contact sheet
        if c_name not in exemplars_by_class:
            exemplars_by_class[c_name] = {
                "class_name": c_name,
                "image": img_np,
                "instance_map": instance_map,
                "nuclei_count": n_detected,
                "features_df": inst_df,
                "regional_df": reg_df,
            }

        if (idx + 1) % 8 == 0 or (idx + 1) == len(process_df):
            logger.info("Processed %d / %d images (current: %s, nuclei=%d)", idx + 1, len(process_df), img_id, n_detected)

    total_time = round(time.time() - start_time, 2)
    mem_end = psutil.Process().memory_info().rss / (1024 * 1024)

    # Concatenate all tables
    master_instance_df = pd.concat(all_instance_dfs, ignore_index=True) if all_instance_dfs else pd.DataFrame()
    master_image_df = pd.DataFrame(image_summaries)
    master_regional_df = pd.concat(regional_dfs, ignore_index=True) if regional_dfs else pd.DataFrame()

    # Save CSV outputs
    manifest_out = out_path / "cellular_sample_manifest.csv"
    process_df.to_csv(manifest_out, index=False)

    inst_out = out_path / "instance_features.csv"
    master_instance_df.to_csv(inst_out, index=False)

    img_out = out_path / "image_morphology_summary.csv"
    master_image_df.to_csv(img_out, index=False)

    reg_out = out_path / "regional_morphology_summary.csv"
    master_regional_df.to_csv(reg_out, index=False)

    # Generate Publication Visualizations
    logger.info("Generating publication-quality visualization figures...")

    # 1. 8-class contact sheet
    contact_list = [exemplars_by_class[c] for c in CANONICAL_CLASSES if c in exemplars_by_class]
    contact_sheet_path = str(viz_dir / "contact_sheet_8classes.png")
    render_contact_sheet(contact_list, contact_sheet_path)

    # 2. Class-level morphology comparison charts
    charts_path = str(viz_dir / "morphology_comparison_charts.png")
    render_morphology_comparison_charts(master_image_df, charts_path)

    # 3. Deep-dive exemplar visualizations on a representative TUM tile
    if "TUM" in exemplars_by_class:
        tum_ex = exemplars_by_class["TUM"]
        render_centroid_map(
            tum_ex["image"], tum_ex["features_df"],
            save_path=str(viz_dir / "exemplar_centroid_map_tum.png"),
            title="TUM Exemplar — Nuclear Centroids & 2×2 Quadrant Layout"
        )
        render_quadrant_density_map(
            tum_ex["image"], tum_ex["regional_df"],
            save_path=str(viz_dir / "exemplar_quadrant_density_tum.png"),
            title="TUM Exemplar — Intra-Patch Quadrant Density Analysis"
        )
        render_single_nucleus_gallery(
            tum_ex["image"], tum_ex["features_df"], tum_ex["instance_map"],
            save_path=str(viz_dir / "exemplar_single_nucleus_gallery_tum.png")
        )

    # Compute high-level quantitative summary
    total_detected = len(master_instance_df)
    mean_nuclei_per_tile = round(float(master_image_df["total_nuclei_count"].mean()), 2)
    median_nuclei_per_tile = round(float(master_image_df["total_nuclei_count"].median()), 2)
    mean_nuclear_area = round(float(master_instance_df["area_px2"].mean()), 2) if total_detected > 0 else 0.0
    median_nuclear_area = round(float(master_instance_df["area_px2"].median()), 2) if total_detected > 0 else 0.0

    # Class-level aggregates
    class_aggregates = {}
    for c in CANONICAL_CLASSES:
        c_sub = master_image_df[master_image_df["class_name"] == c]
        if not c_sub.empty:
            class_aggregates[c] = {
                "sample_count": len(c_sub),
                "mean_nuclei_count": round(float(c_sub["total_nuclei_count"].mean()), 2),
                "std_nuclei_count": round(float(c_sub["total_nuclei_count"].std()), 2),
                "mean_nuclear_area_px2": round(float(c_sub["mean_nuclear_area_px2"].mean()), 2),
                "mean_circularity": round(float(c_sub["mean_circularity"].mean()), 3),
                "mean_eccentricity": round(float(c_sub["mean_eccentricity"].mean()), 3),
                "mean_density_per_kpx": round(float(c_sub["nuclear_density_per_kpx"].mean()), 4),
            }

    summary_payload = {
        "phase": "Original Phase 7",
        "phase_name": "Nuclei Detection and Cellular Morphology",
        "implementation_status": "EXPLORATORY_BASELINE_OPERATIONAL",
        "deep_models_status": "BLOCKED_BY_PRETRAINED_WEIGHTS_AND_DEPENDENCIES",
        "selected_detector": model_info,
        "pilot_dataset": {
            "name": "HMU-GC-HE-30K (GCHTID)",
            "total_images_processed": len(process_df),
            "classes_evaluated": len(class_aggregates),
            "images_per_class": int(len(process_df) / len(class_aggregates)),
            "sampling_seed": 42,
        },
        "quantitative_findings": {
            "total_nuclei_detected": total_detected,
            "mean_nuclei_per_tile": mean_nuclei_per_tile,
            "median_nuclei_per_tile": median_nuclei_per_tile,
            "mean_nuclear_area_px2": mean_nuclear_area,
            "median_nuclear_area_px2": median_nuclear_area,
            "class_level_breakdown": class_aggregates,
        },
        "telemetry": {
            "total_runtime_seconds": total_time,
            "mean_seconds_per_image": round(total_time / len(process_df), 3),
            "rss_memory_start_mb": round(mem_start, 2),
            "rss_memory_end_mb": round(mem_end, 2),
        },
        "scientific_disclaimer": (
            "All nuclear instance detections and morphology measurements were derived "
            "using an automated, unvalidated color-deconvolved watershed baseline. "
            "Without pathologist-verified manual nuclear ground truth annotations, "
            "these values represent computational image descriptors and must NOT be "
            "interpreted as verified biological differences between tissue classes."
        ),
    }

    summary_out = out_path / "cellular_summary.json"
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    logger.info("Phase 7 pilot execution completed in %.2fs (Total nuclei: %d)", total_time, total_detected)
    return summary_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GastricPath AI — Cellular Morphology Pipeline")
    parser.add_argument("--smoke-test", action="store_true", help="Run 4-image smoke test only")
    args = parser.parse_args()

    pilot_manifest = select_pilot_cohort(images_per_class=8, seed=42)
    summary = run_pilot_inference(pilot_manifest, smoke_test_only=args.smoke_test)
    print("\n=== Phase 7 Pilot Summary ===")
    print(json.dumps(summary["quantitative_findings"], indent=2))

