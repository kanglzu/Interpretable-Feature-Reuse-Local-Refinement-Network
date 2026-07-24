"""CAM heatmap generation and mask-supervision loss."""

import re
from typing import Optional

import torch
import torch.nn as nn

from training.loss import (
    SoftBinaryDiceLoss,
    mask_to_foreground,
    normalize_cam,
    resize_cam,
)


def resolve_layer(model: nn.Module, layer_path: str) -> nn.Module:
    obj = model
    parts = [part for part in re.split(r"[\.\[\]]", layer_path) if part]
    for part in parts:
        if re.fullmatch(r"-?\d+", part):
            obj = obj[int(part)]  # type: ignore[index]
        else:
            obj = getattr(obj, part)
    return obj


def _to_nchw(t: torch.Tensor) -> torch.Tensor:
    if t.dim() != 4:
        return t
    _, d1, d2, d3 = t.shape
    if d1 == d2 and d1 != d3:
        return t.permute(0, 3, 1, 2).contiguous()
    return t


class CAMLoss(nn.Module):
    """Align CAM heatmaps with ground-truth masks."""

    def __init__(self, smooth: float = 1e-6, foreground_channel: int = 1):
        super().__init__()
        self.eps = float(smooth)
        self.foreground_channel = int(foreground_channel)
        self.mask_loss = SoftBinaryDiceLoss(smooth=self.eps)

    def grad_cam_heatmap(
        self,
        activation: Optional[torch.Tensor],
        outputs,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        if activation is None:
            raise RuntimeError("CAM loss is enabled but no target-layer activation was captured.")

        score = self._foreground_logit(outputs).sum()
        grads = self._activation_grads(score, activation)

        activation = _to_nchw(activation.float())
        grads = _to_nchw(grads.float())
        if activation.dim() != 4 or activation.shape != grads.shape:
            raise RuntimeError(
                "CAM activation and gradient must be matching 4D tensors after "
                f"NCHW conversion, got activation={tuple(activation.shape)} and "
                f"grads={tuple(grads.shape)}."
            )

        weights = grads.mean(dim=(2, 3), keepdim=True)
        weights = weights.detach()
        heatmap = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        heatmap = resize_cam(heatmap, masks.shape[-2:])
        return normalize_cam(heatmap, eps=self.eps)

    def layer_cam_heatmap(
        self,
        activation: Optional[torch.Tensor],
        outputs,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        if activation is None:
            raise RuntimeError("CAM loss is enabled but no target-layer activation was captured.")

        score = self._foreground_logit(outputs).sum()
        grads = self._activation_grads(score, activation)

        activation = _to_nchw(activation.float())
        grads = _to_nchw(grads.float())
        if activation.dim() != 4 or activation.shape != grads.shape:
            raise RuntimeError(
                "CAM activation and gradient must be matching 4D tensors after "
                f"NCHW conversion, got activation={tuple(activation.shape)} and "
                f"grads={tuple(grads.shape)}."
            )

        weights = torch.relu(grads)
        weights = weights.detach()
        heatmap = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        heatmap = resize_cam(heatmap, masks.shape[-2:])
        return normalize_cam(heatmap, eps=self.eps)

    def grad_cam_loss(
        self,
        activation: Optional[torch.Tensor],
        outputs,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        return self._heatmap_mask_loss(
            self.grad_cam_heatmap(activation, outputs, masks),
            masks,
        )

    def layer_cam_loss(
        self,
        activation: Optional[torch.Tensor],
        outputs,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        return self._heatmap_mask_loss(
            self.layer_cam_heatmap(activation, outputs, masks),
            masks,
        )

    def _foreground_logit(self, outputs) -> torch.Tensor:
        logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        if logits.shape[1] == 1:
            return logits[:, 0].float()
        if self.foreground_channel >= logits.shape[1]:
            raise ValueError(
                "foreground_channel is outside the model output channels: "
                f"{self.foreground_channel} >= {logits.shape[1]}."
            )
        return logits[:, self.foreground_channel].float()

    def _activation_grads(self, score: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
        grads = torch.autograd.grad(
            score,
            activation,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )[0]
        if grads is None:
            raise RuntimeError(
                "CAM target score is not connected to the captured activation. "
                "Check the configured CAM target layer for this backbone."
            )
        if not torch.isfinite(grads).all():
            raise RuntimeError("CAM activation gradients contain non-finite values.")
        return grads

    def _heatmap_mask_loss(self, heatmap: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return self.mask_loss(heatmap, mask_to_foreground(masks))
