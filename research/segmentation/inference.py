"""
Inference and Visualization Module for Gastric Histopathology Segmentation.

Provides overlay utilities, color mappings, and side-by-side comparative
visualization for H&E images, ground truth annotations, and model predictions.
Transparently communicates when predictions are unavailable due to data blockers.
"""

from typing import Dict, Optional, Tuple, Union
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from research.segmentation.annotations import (
    DEFAULT_CLASS_MAPPING,
    generate_synthetic_geojson_fixture,
    rasterize_qupath_geojson,
)

# Standard histopathology segmentation color palette (RGB)
DEFAULT_CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (240, 240, 240),  # Background / Lumen (Off-white/Light grey)
    1: (255, 230, 150),  # ADI - Adipose Tissue (Soft Cream / Light Yellow)
    2: (128, 128, 128),  # DEB - Debris / Necrosis (Medium Grey)
    3: (255, 215, 0),    # LYM - Lymphocytes / Inflammatory (Gold/Yellow)
    4: (255, 140, 0),    # MUC - Extracellular Mucus (Dark Orange)
    5: (147, 112, 219),  # MUS - Smooth Muscle (Medium Purple)
    6: (30, 144, 255),   # NOR - Normal Gastric Mucosa (Dodger Blue)
    7: (46, 139, 87),    # STR - Stroma / Desmoplasia (Sea Green)
    8: (220, 20, 60),    # TUM - Tumor Adenocarcinoma (Crimson Red)
}

DEFAULT_CLASS_NAMES: Dict[int, str] = {
    0: "Background / Lumen",
    1: "Adipose Tissue (ADI)",
    2: "Debris / Necrosis (DEB)",
    3: "Lymphocytes / Inflammatory (LYM)",
    4: "Extracellular Mucus (MUC)",
    5: "Smooth Muscle (MUS)",
    6: "Normal Gastric Mucosa (NOR)",
    7: "Cancer-Associated Stroma (STR)",
    8: "Tumor Epithelium (TUM)",
}


def overlay_mask_on_image(
    image: Union[np.ndarray, Image.Image],
    mask: np.ndarray,
    alpha: float = 0.45,
    class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    ignore_class: int = 0,
) -> np.ndarray:
    """
    Overlay a categorical segmentation mask onto an H&E RGB image using alpha blending.

    Args:
        image: RGB image as uint8 numpy array [H, W, 3] or PIL Image.
        mask: Categorical 2D integer mask array [H, W].
        alpha: Transparency factor in [0.0, 1.0] for the mask overlay.
        class_colors: Mapping from class integer ID to (R, G, B) tuple.
        ignore_class: Class ID that should remain completely transparent (default: 0).

    Returns:
        np.ndarray: Blended uint8 RGB image of shape [H, W, 3].
    """
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"), dtype=np.uint8)
    elif not isinstance(image, np.ndarray):
        raise TypeError(f"Expected image as np.ndarray or PIL.Image, got {type(image)}")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Image must have shape (H, W, 3), got {image.shape}")

    if mask.ndim != 2:
        raise ValueError(f"Mask must have shape (H, W), got {mask.shape}")

    if image.shape[:2] != mask.shape:
        raise ValueError(
            f"Shape mismatch: Image shape {image.shape[:2]} vs Mask shape {mask.shape}"
        )

    colors = class_colors or DEFAULT_CLASS_COLORS
    blended = image.astype(np.float32).copy()

    # Apply color per unique class present in mask (skipping ignore_class)
    present_classes = np.unique(mask)
    for cls_id in present_classes:
        if cls_id == ignore_class:
            continue
        color_rgb = np.array(colors.get(int(cls_id), (128, 128, 128)), dtype=np.float32)
        cls_pixels = (mask == cls_id)
        blended[cls_pixels] = (1.0 - alpha) * blended[cls_pixels] + alpha * color_rgb

    return np.clip(blended, 0, 255).astype(np.uint8)


def render_segmentation_comparison(
    image: Union[np.ndarray, Image.Image],
    gt_mask: Optional[np.ndarray] = None,
    pred_mask: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    class_names: Optional[Dict[int, str]] = None,
    class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
    alpha: float = 0.45,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Render a clean side-by-side comparative panel for H&E patch, Ground Truth, and Model Prediction.

    If pred_mask is None, clearly displays an 'Untrained Model / Awaiting Pathologist Annotations'
    placeholder banner rather than hallucinating predictions.

    Args:
        image: RGB image [H, W, 3].
        gt_mask: Optional Ground Truth mask [H, W].
        pred_mask: Optional predicted mask [H, W].
        title: Overall super-title for the figure.
        class_names: Dict mapping class IDs to human-readable names.
        class_colors: Dict mapping class IDs to RGB tuples.
        alpha: Blending transparency.
        save_path: Optional file path to save the rendered figure.

    Returns:
        plt.Figure: The rendered Matplotlib Figure object.
    """
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"), dtype=np.uint8)

    names = class_names or DEFAULT_CLASS_NAMES
    colors = class_colors or DEFAULT_CLASS_COLORS

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)

    # Panel 1: Original H&E Patch
    axes[0].imshow(image)
    axes[0].set_title("Input H&E Tile (224×224)", fontsize=12, fontweight="bold", pad=8)
    axes[0].axis("off")

    # Panel 2: Ground Truth Annotation / Overlay
    if gt_mask is not None:
        gt_overlay = overlay_mask_on_image(image, gt_mask, alpha=alpha, class_colors=colors)
        axes[1].imshow(gt_overlay)
        axes[1].set_title("Ground Truth Annotation (Overlay)", fontsize=12, fontweight="bold", pad=8)
    else:
        axes[1].imshow(np.ones_like(image) * 245)
        axes[1].text(
            0.5, 0.5,
            "No Ground Truth Mask\nAvailable in Dataset\n(Classification Only)",
            ha="center", va="center", fontsize=11, color="#555555",
            transform=axes[1].transAxes,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#F0F0F0", edgecolor="#AAAAAA")
        )
        axes[1].set_title("Ground Truth (Unavailable)", fontsize=12, fontweight="bold", pad=8)
    axes[1].axis("off")

    # Panel 3: Model Prediction or Untrained Status Banner
    if pred_mask is not None:
        pred_overlay = overlay_mask_on_image(image, pred_mask, alpha=alpha, class_colors=colors)
        axes[2].imshow(pred_overlay)
        axes[2].set_title("U-Net Prediction (Overlay)", fontsize=12, fontweight="bold", pad=8)
    else:
        # Transparent display of data blocker / untrained state
        axes[2].imshow(np.ones_like(image) * 240)
        axes[2].text(
            0.5, 0.5,
            "Model Status: UNTRAINED\n\n"
            "Pixel Annotations Pending\n"
            "Architecture Ready (2D U-Net)\n"
            "Awaiting Pathologist GeoJSON",
            ha="center", va="center", fontsize=10, fontweight="semibold", color="#8B0000",
            transform=axes[2].transAxes,
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#FFF0F0", edgecolor="#D9534F", linewidth=1.5)
        )
        axes[2].set_title("U-Net Prediction (Awaiting Data)", fontsize=12, fontweight="bold", pad=8)
    axes[2].axis("off")

    # Create Legend Elements for classes present in GT or Prediction
    present_classes = set()
    if gt_mask is not None:
        present_classes.update(np.unique(gt_mask))
    if pred_mask is not None:
        present_classes.update(np.unique(pred_mask))

    legend_patches = []
    for cls_id in sorted(present_classes):
        if cls_id == 0:
            continue
        c_rgb = [x / 255.0 for x in colors.get(cls_id, (128, 128, 128))]
        label = names.get(cls_id, f"Class {cls_id}")
        patch = mpatches.Patch(color=c_rgb, label=label)
        legend_patches.append(patch)

    if legend_patches:
        fig.legend(
            handles=legend_patches,
            loc="lower center",
            ncol=min(len(legend_patches), 4),
            bbox_to_anchor=(0.5, -0.05),
            frameon=True,
            facecolor="#FAFAFA",
            edgecolor="#CCCCCC",
            fontsize=10,
        )

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def run_segmentation_demo(
    image_path: Optional[str] = None,
    output_path: str = "research/segmentation/segmentation_overlay_example.png",
) -> str:
    """
    Run an end-to-end demonstration of the segmentation pipeline:
    Loads a reference H&E patch, rasterizes synthetic QuPath GeoJSON annotations,
    and produces a publication-quality comparison figure illustrating the
    untrained baseline state awaiting clinical annotations.

    Args:
        image_path: Path to reference image, or None to load from dataset.
        output_path: Output PNG path.

    Returns:
        str: Absolute path of saved comparison figure.
    """
    # Attempt to locate reference H&E image in dataset
    target_img = None
    if image_path and os.path.exists(image_path):
        target_img = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    else:
        candidate_paths = [
            "data/raw/HMU-GC-HE-30K/all_image/ADI/ADI_1033.png",
            "data/raw/HMU-GC-HE-30K/all_image/NORM/NORM_1001.png",
            "data/raw/HMU-GC-HE-30K/all_image/TUM/TUM_1001.png",
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                target_img = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
                break

    if target_img is None:
        # Create synthetic H&E background if no raw image found
        h, w = 224, 224
        target_img = np.full((h, w, 3), (230, 200, 215), dtype=np.uint8)  # Eosin pinkish
        # Add slight texture
        noise = np.random.RandomState(42).randint(-15, 15, (h, w, 3))
        target_img = np.clip(target_img.astype(np.int32) + noise, 0, 255).astype(np.uint8)

    # Generate synthetic QuPath GeoJSON fixture representing multi-tissue histology
    synthetic_geojson = generate_synthetic_geojson_fixture()
    gt_mask, summary = rasterize_qupath_geojson(
        synthetic_geojson,
        target_size=(224, 224),
        class_mapping=DEFAULT_CLASS_MAPPING,
    )

    fig = render_segmentation_comparison(
        image=target_img,
        gt_mask=gt_mask,
        pred_mask=None,  # Untrained baseline
        title="GastricPath AI — Original Phase 6: Tissue Segmentation Pipeline Verification",
        save_path=output_path,
    )
    plt.close(fig)

    abs_output = os.path.abspath(output_path)
    print(f"[Phase 6] Segmentation demo comparison saved to: {abs_output}")
    return abs_output


if __name__ == "__main__":
    run_segmentation_demo()

