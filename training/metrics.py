"""Segmentation metrics and moving-average tracking."""

from typing import Dict, List, Optional, Union
from medpy.metric.binary import hd95 as medpy_hd95

import numpy as np
import torch
import torch.nn.functional as F


def pseudo_dice_batch(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-5,
) -> float:
    if isinstance(logits, (list, tuple)):
        logits = logits[0]

    with torch.no_grad():
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        dice_values = []
        for c in range(1, num_classes):
            pred_c = (preds == c).float()
            tgt_c  = (targets == c).float()
            inter  = (pred_c * tgt_c).sum()
            union  = pred_c.sum() + tgt_c.sum()
            dice_values.append(((2.0 * inter + smooth) / (union + smooth)).item())

        return float(np.mean(dice_values)) if dice_values else 0.0


def compute_dice(pred: np.ndarray, target: np.ndarray, num_classes: int, smooth: float = 1e-5) -> float:
    dice_values = []
    for c in range(1, num_classes):
        pred_c = (pred == c)
        tgt_c  = (target == c)
        inter  = (pred_c & tgt_c).sum()
        union  = pred_c.sum() + tgt_c.sum()
        if union == 0:
            dice_values.append(float("nan"))
        else:
            dice_values.append((2.0 * inter + smooth) / (union + smooth))
    valid = [v for v in dice_values if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def compute_iou(pred: np.ndarray, target: np.ndarray, num_classes: int, smooth: float = 1e-5) -> float:
    iou_values = []
    for c in range(1, num_classes):
        pred_c = (pred == c)
        tgt_c  = (target == c)
        inter  = (pred_c & tgt_c).sum()
        union  = (pred_c | tgt_c).sum()
        if union == 0:
            iou_values.append(float("nan"))
        else:
            iou_values.append((inter + smooth) / (union + smooth))
    valid = [v for v in iou_values if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def compute_hd95(pred: np.ndarray, target: np.ndarray, num_classes: int) -> float:

    hd95_values = []
    for c in range(1, num_classes):
        pred_c = (pred == c)
        tgt_c  = (target == c)
        if pred_c.sum() == 0 or tgt_c.sum() == 0:
            hd95_values.append(float("nan"))
        else:
            hd95_values.append(medpy_hd95(pred_c, tgt_c))

    valid = [v for v in hd95_values if not np.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def aggregate_metrics(
    all_preds: List[np.ndarray],
    all_targets: List[np.ndarray],
    num_classes: int,
    compute_hd: bool = False,
) -> Dict[str, float]:
    dice_list, iou_list, hd95_list = [], [], []
    for pred, tgt in zip(all_preds, all_targets):
        dice_list.append(compute_dice(pred, tgt, num_classes))
        iou_list.append(compute_iou(pred, tgt, num_classes))
        if compute_hd:
            hd95_list.append(compute_hd95(pred, tgt, num_classes))

    result = {
        "dice": float(np.nanmean(dice_list)),
        "iou": float(np.nanmean(iou_list)),
    }
    if compute_hd:
        result["hd95"] = float(np.nanmean(hd95_list))
    return result


class EMATracker:

    def __init__(self, alpha: float = 0.9):
        self.alpha = alpha
        self._value: Optional[float] = None

    def update(self, value: float) -> float:
        if self._value is None:
            self._value = value
        else:
            self._value = self.alpha * self._value + (1.0 - self.alpha) * value
        return self._value

    @property
    def value(self) -> Optional[float]:
        return self._value

    def reset(self):
        self._value = None
