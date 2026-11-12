"""Model Evaluator for GastricPath AI.

Performs forward inference over validation or test DataLoaders, collects softmax probabilities,
and computes overall and per-class computational pathology metrics.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from research.benchmark.dataset import CANONICAL_CLASSES
from research.benchmark.metrics import compute_comprehensive_metrics

logger = logging.getLogger(__name__)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | str = "cpu",
    class_names: list[str] | None = None,
    print_progress: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluates a model over a DataLoader, returning comprehensive metrics and detailed prediction rows."""
    if class_names is None:
        class_names = CANONICAL_CLASSES

    model.eval()
    model.to(device)

    all_y_true: list[int] = []
    all_y_prob: list[list[float]] = []
    prediction_records: list[dict[str, Any]] = []

    total_batches = len(dataloader)
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            inputs, targets, paths = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            t_list = targets.cpu().tolist()
            p_list = preds.tolist()

            all_y_true.extend(t_list)
            all_y_prob.extend(probs.tolist())

            for i in range(len(paths)):
                rel_path = paths[i]
                gt_idx = t_list[i]
                pred_idx = p_list[i]
                is_correct = bool(gt_idx == pred_idx)
                conf = float(probs[i, pred_idx])

                rec = {
                    "relative_path": rel_path,
                    "ground_truth_label": class_names[gt_idx],
                    "ground_truth_index": gt_idx,
                    "predicted_label": class_names[pred_idx],
                    "predicted_index": pred_idx,
                    "is_correct": is_correct,
                    "confidence": round(conf, 4),
                }
                # Add per-class probability columns
                for c_i, c_n in enumerate(class_names):
                    rec[f"prob_{c_n}"] = round(float(probs[i, c_i]), 4)

                prediction_records.append(rec)

            if print_progress and ((batch_idx + 1) % 25 == 0 or (batch_idx + 1) == total_batches):
                pct = 100.0 * (batch_idx + 1) / total_batches
                print(f"  [Progress] Processed batch {batch_idx + 1}/{total_batches} ({pct:.1f}%) — {len(all_y_true)} images evaluated")

    metrics_result = compute_comprehensive_metrics(all_y_true, all_y_prob, class_names=class_names)
    return metrics_result, prediction_records
