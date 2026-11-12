"""Evaluation Metrics for Tissue Segmentation: Dice Similarity Coefficient and IoU.

Implements standard medical image segmentation benchmarks with robust edge-case handling.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


def compute_dice_coefficient(
    pred_mask: Union[np.ndarray, torch.Tensor],
    gt_mask: Union[np.ndarray, torch.Tensor],
    class_id: int,
    epsilon: float = 1e-6,
) -> float:
    """
    Computes Dice Similarity Coefficient (F1) for a specific class.
    Args:
        pred_mask: 2D array/tensor of integer predicted class labels.
        gt_mask: 2D array/tensor of integer ground-truth class labels.
        class_id: Class integer index to evaluate.
    Returns:
        dice: Float in [0.0, 1.0].
    """
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.detach().cpu().numpy()

    p = (pred_mask == class_id).astype(np.float32)
    g = (gt_mask == class_id).astype(np.float32)

    sum_p = np.sum(p)
    sum_g = np.sum(g)

    # Edge case: both ground-truth and prediction have 0 pixels for this class
    if sum_p == 0 and sum_g == 0:
        return 1.0  # Perfect agreement on class absence
    if sum_p == 0 or sum_g == 0:
        return 0.0

    intersection = np.sum(p * g)
    dice = (2.0 * intersection) / (sum_p + sum_g + epsilon)
    return float(np.clip(dice, 0.0, 1.0))


def compute_iou_jaccard(
    pred_mask: Union[np.ndarray, torch.Tensor],
    gt_mask: Union[np.ndarray, torch.Tensor],
    class_id: int,
    epsilon: float = 1e-6,
) -> float:
    """
    Computes Intersection over Union (Jaccard Index) for a specific class.
    """
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.detach().cpu().numpy()

    p = (pred_mask == class_id).astype(np.float32)
    g = (gt_mask == class_id).astype(np.float32)

    sum_p = np.sum(p)
    sum_g = np.sum(g)

    if sum_p == 0 and sum_g == 0:
        return 1.0
    if sum_p == 0 or sum_g == 0:
        return 0.0

    intersection = np.sum(p * g)
    union = sum_p + sum_g - intersection
    iou = intersection / (union + epsilon)
    return float(np.clip(iou, 0.0, 1.0))


def evaluate_segmentation_batch(
    pred_masks: Union[np.ndarray, torch.Tensor],
    gt_masks: Union[np.ndarray, torch.Tensor],
    class_names: Optional[Dict[str, int]] = None,
    include_background: bool = False,
) -> Dict[str, Any]:
    """
    Evaluates a batch of masks, reporting per-class Dice, IoU, Mean Dice, and Mean IoU.
    Args:
        pred_masks: (B, H, W) or (H, W) array of predicted class integers.
        gt_masks: (B, H, W) or (H, W) array of ground-truth class integers.
        class_names: Dict mapping class names to integer indices.
        include_background: Whether to include class 0 (Background) in the macro mean.
    """
    if isinstance(pred_masks, torch.Tensor):
        pred_masks = pred_masks.detach().cpu().numpy()
    if isinstance(gt_masks, torch.Tensor):
        gt_masks = gt_masks.detach().cpu().numpy()

    if pred_masks.ndim == 2:
        pred_masks = np.expand_dims(pred_masks, axis=0)
    if gt_masks.ndim == 2:
        gt_masks = np.expand_dims(gt_masks, axis=0)

    if class_names is None:
        unique_classes = sorted(list(set(np.unique(pred_masks)) | set(np.unique(gt_masks))))
        class_names = {f"Class_{c}": c for c in unique_classes}

    per_class_dice: Dict[str, float] = {}
    per_class_iou: Dict[str, float] = {}

    eval_dice_list = []
    eval_iou_list = []

    for name, c_idx in class_names.items():
        batch_dices = []
        batch_ious = []
        for i in range(pred_masks.shape[0]):
            d = compute_dice_coefficient(pred_masks[i], gt_masks[i], class_id=c_idx)
            u = compute_iou_jaccard(pred_masks[i], gt_masks[i], class_id=c_idx)
            batch_dices.append(d)
            batch_ious.append(u)

        mean_d = float(np.mean(batch_dices))
        mean_u = float(np.mean(batch_ious))
        per_class_dice[name] = round(mean_d, 4)
        per_class_iou[name] = round(mean_u, 4)

        if c_idx != 0 or include_background:
            eval_dice_list.append(mean_d)
            eval_iou_list.append(mean_u)

    m_dice = float(np.mean(eval_dice_list)) if eval_dice_list else 0.0
    m_iou = float(np.mean(eval_iou_list)) if eval_iou_list else 0.0

    return {
        "mean_dice": round(m_dice, 4),
        "mean_iou": round(m_iou, 4),
        "per_class_dice": per_class_dice,
        "per_class_iou": per_class_iou,
        "num_evaluated_samples": pred_masks.shape[0],
    }

