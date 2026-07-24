"""Loss functions for segmentation and CAM supervision."""

from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):

    def __init__(self, smooth: float = 1e-5, apply_softmax: bool = True):
        super().__init__()
        self.smooth = smooth
        self.apply_softmax = apply_softmax

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        num_classes = logits.shape[1]

        if self.apply_softmax:
            probs = torch.softmax(logits, dim=1)
        else:
            probs = logits

        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        probs_flat = probs.view(probs.shape[0], num_classes, -1)
        targets_flat = targets_one_hot.view(targets_one_hot.shape[0], num_classes, -1)
        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)
        dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_fg = dice_per_class[:, 1:].mean()
        return 1.0 - dice_fg


class DiceCELoss(nn.Module):

    def __init__(
        self,
        ce_weight: float = 0.5,
        smooth: float = 1e-5,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.dice_weight = 1.0 - ce_weight
        self.ce_weight = ce_weight
        self.dice = SoftDiceLoss(smooth=smooth)
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        return self.dice_weight * self.dice(logits, targets) + self.ce_weight * self.ce(logits, targets)


class DeepSupervisionLoss(nn.Module):

    def __init__(self, base_loss: nn.Module, weights: List[float]):
        super().__init__()
        self.base_loss = base_loss
        total = sum(weights)
        self.weights = [w / total for w in weights]

    def forward(
        self,
        outputs: Union[torch.Tensor, List[torch.Tensor]],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(outputs, (list, tuple)):
            return self.base_loss(outputs, targets)

        loss = 0.0
        for i, pred in enumerate(outputs):
            w = self.weights[i] if i < len(self.weights) else self.weights[-1]
            if pred.shape[-2:] != targets.shape[-2:]:
                tgt = _resize_targets(targets, pred.shape[-2:])
            else:
                tgt = targets
            loss = loss + w * self.base_loss(pred, tgt)
        return loss


class SoftBinaryDiceLoss(nn.Module):

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, cam: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if cam.dim() == 3:
            cam = cam.unsqueeze(1)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        cam = cam.float().clamp(0.0, 1.0)
        targets = targets.float().clamp(0.0, 1.0)

        dims = tuple(range(2, cam.dim()))
        intersection = (cam * targets).sum(dim=dims)
        union = cam.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


def mask_to_foreground(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    return (mask > 0).float().unsqueeze(1)


def normalize_cam(cam: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if cam.dim() == 3:
        cam = cam.unsqueeze(1)
    cam = cam.float()
    dims = tuple(range(2, cam.dim()))
    cam_min = cam.amin(dim=dims, keepdim=True)
    cam_max = cam.amax(dim=dims, keepdim=True)
    return (cam - cam_min) / (cam_max - cam_min + eps)


def resize_cam(cam: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if cam.dim() == 3:
        cam = cam.unsqueeze(1)
    return F.interpolate(cam, size=size, mode="bilinear", align_corners=False)


def _resize_targets(targets: torch.Tensor, size: tuple) -> torch.Tensor:
    return F.interpolate(
        targets.unsqueeze(1).float(),
        size=size,
        mode="nearest",
    ).squeeze(1).long()


def build_loss(cfg: dict) -> nn.Module:
    ce_weight = cfg["training"].get("loss_ce_weight", 0.5)
    base_loss = DiceCELoss(ce_weight=ce_weight)

    if cfg["model"].get("deep_supervision", False):
        weights = cfg["model"].get("ds_weights", [1.0, 0.5, 0.25, 0.125])
        return DeepSupervisionLoss(base_loss, weights)

    return base_loss
