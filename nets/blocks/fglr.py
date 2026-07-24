"""Feature-Guided Local Refinement block."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .aspp import ASPP


SpatialSize = Union[int, str, Sequence[int], torch.Size]
FUSE_ASPP_RATES = (2, 4, 6)


def _valid_num_groups(num_channels: int, preferred: int = 8) -> int:
    preferred = max(1, int(preferred))
    num_channels = int(num_channels)
    for groups in range(min(preferred, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


@dataclass
class FGLROutput:

    refined_logits: torch.Tensor
    delta_logits: torch.Tensor
    instance_boxes: List[torch.Tensor]
    stage_logits: List[torch.Tensor]


def _resolve_crop_size(crop_size: SpatialSize, input_size: Tuple[int, int]) -> Tuple[int, int]:
    if crop_size == "input":
        return int(input_size[0]), int(input_size[1])
    if isinstance(crop_size, int):
        return int(crop_size), int(crop_size)
    if isinstance(crop_size, str):
        raise ValueError(f"Unsupported crop_size={crop_size!r}; use 'input', int, or (H, W).")
    if len(crop_size) != 2:
        raise ValueError(f"crop_size must be int, 'input', or length-2 sequence, got {crop_size}.")
    return int(crop_size[0]), int(crop_size[1])


def _foreground_probability(logits: torch.Tensor, foreground_channel: int) -> torch.Tensor:
    if logits.dim() != 4:
        raise ValueError(f"logits must be BCHW, got shape {tuple(logits.shape)}.")
    if logits.shape[1] == 1:
        return torch.sigmoid(logits[:, 0])
    if foreground_channel < 0 or foreground_channel >= logits.shape[1]:
        raise ValueError(
            f"foreground_channel={foreground_channel} is invalid for logits with "
            f"{logits.shape[1]} channels."
        )
    return torch.softmax(logits, dim=1)[:, foreground_channel]


def _boxes_from_binary_mask(
    mask: np.ndarray,
    min_area: int,
    margin_ratio: float,
    height: int,
    width: int,
) -> List[Tuple[int, int, int, int]]:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes: List[Tuple[int, int, int, int]] = []
    for label_idx in range(1, n_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x0_raw = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y0_raw = int(stats[label_idx, cv2.CC_STAT_TOP])
        w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        margin = int(round(max(h, w) * margin_ratio))
        y0 = max(0, y0_raw - margin)
        x0 = max(0, x0_raw - margin)
        y1 = min(height, y0_raw + h + margin)
        x1 = min(width, x0_raw + w + margin)
        if y1 > y0 and x1 > x0:
            boxes.append((y0, x0, y1, x1))
    return boxes


def _flatten_boxes(boxes: Sequence[torch.Tensor], device: torch.device) -> torch.Tensor:
    rows = []
    for batch_idx, batch_boxes in enumerate(boxes):
        if batch_boxes.numel() == 0:
            continue
        batch_col = torch.full(
            (batch_boxes.shape[0], 1),
            float(batch_idx),
            dtype=batch_boxes.dtype,
            device=batch_boxes.device,
        )
        rows.append(torch.cat([batch_col, batch_boxes], dim=1))
    if not rows:
        return torch.empty((0, 5), dtype=torch.float32, device=device)
    return torch.cat(rows, dim=0).to(device=device, dtype=torch.float32)


def _roi_grid(
    boxes_yxyx: torch.Tensor,
    output_size: Tuple[int, int],
    reference_size: Tuple[int, int],
) -> torch.Tensor:
    crop_h, crop_w = int(output_size[0]), int(output_size[1])
    ref_h, ref_w = int(reference_size[0]), int(reference_size[1])
    if ref_h <= 1 or ref_w <= 1:
        raise ValueError(f"reference_size must be larger than 1 pixel, got {reference_size}.")

    y0, x0, y1, x1 = boxes_yxyx.unbind(dim=1)
    y1 = torch.maximum(y1 - 1.0, y0)
    x1 = torch.maximum(x1 - 1.0, x0)
    ys = torch.linspace(0.0, 1.0, crop_h, device=boxes_yxyx.device, dtype=boxes_yxyx.dtype)
    xs = torch.linspace(0.0, 1.0, crop_w, device=boxes_yxyx.device, dtype=boxes_yxyx.dtype)
    yy = y0[:, None] * (1.0 - ys[None, :]) + y1[:, None] * ys[None, :]
    xx = x0[:, None] * (1.0 - xs[None, :]) + x1[:, None] * xs[None, :]
    yy = yy[:, :, None].expand(-1, crop_h, crop_w)
    xx = xx[:, None, :].expand(-1, crop_h, crop_w)
    grid_x = (2.0 * xx / float(ref_w - 1)) - 1.0
    grid_y = (2.0 * yy / float(ref_h - 1)) - 1.0
    return torch.stack([grid_x, grid_y], dim=-1)


def _sample_rois(
    source: torch.Tensor,
    boxes_batched: torch.Tensor,
    crop_size: Tuple[int, int],
    reference_size: Tuple[int, int],
) -> torch.Tensor:
    if boxes_batched.numel() == 0:
        return source.new_zeros((0, source.shape[1], crop_size[0], crop_size[1]))
    batch_idx = boxes_batched[:, 0].long()
    source_per_box = source[batch_idx]
    grid = _roi_grid(boxes_batched[:, 1:], crop_size, reference_size)
    return F.grid_sample(
        source_per_box,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


class ConvGNAct(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size <= 0 or kernel_size % 2 != 1:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}.")
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                int(in_channels),
                int(out_channels),
                kernel_size=kernel_size,
                stride=int(stride),
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=_valid_num_groups(int(out_channels), norm_groups),
                num_channels=int(out_channels),
            ),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TransposeConvGNAct(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                int(in_channels),
                int(out_channels),
                kernel_size=2,
                stride=2,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=_valid_num_groups(int(out_channels), norm_groups),
                num_channels=int(out_channels),
            ),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownsampleFeatureBranch(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        expanded = 2 * int(hidden_channels)
        self.block = nn.Sequential(
            ConvGNAct(in_channels, expanded, kernel_size=1, norm_groups=norm_groups),
            ConvGNAct(expanded, expanded, kernel_size=3, stride=2, norm_groups=norm_groups),
            ConvGNAct(expanded, expanded, kernel_size=5, norm_groups=norm_groups),
        )
        self.out_channels = expanded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FuseUpsampleBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        reduced = int(in_channels) // 2
        hidden = int(hidden_channels)
        self.aspp = ASPP(
            in_channels,
            in_channels,
            rates=FUSE_ASPP_RATES,
            norm_groups=norm_groups,
            residual=False,
        )
        self.reduce = ConvGNAct(in_channels, reduced, kernel_size=1, norm_groups=norm_groups)
        self.context = ConvGNAct(reduced, reduced, kernel_size=5, norm_groups=norm_groups)
        self.up = TransposeConvGNAct(reduced, hidden, norm_groups=norm_groups)
        self.finish = ConvGNAct(hidden, hidden, kernel_size=3, norm_groups=norm_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.aspp(x)
        x = self.reduce(x)
        x = self.context(x)
        x = self.up(x)
        x = self.finish(x)
        return x

ConvNormAct = ConvGNAct


def _conv_gn_act_params(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    include_norm: bool = True,
) -> int:
    total = int(in_channels) * int(out_channels) * int(kernel_size) * int(kernel_size)
    if include_norm:
        total += 2 * int(out_channels)
    return int(total)


def _branch_params(in_channels: int, hidden_channels: int, include_norm: bool = True) -> int:
    expanded = 2 * int(hidden_channels)
    return int(
        _conv_gn_act_params(in_channels, expanded, 1, include_norm)
        + _conv_gn_act_params(expanded, expanded, 3, include_norm)
        + _conv_gn_act_params(expanded, expanded, 5, include_norm)
    )


def _aspp_params(channels: int, rates: Sequence[int], include_norm: bool = True) -> int:
    channels = int(channels)
    num_rates = len(tuple(rates))
    total = channels * channels * (num_rates * 10 + 3)
    if include_norm:
        total += 2 * channels * (num_rates + 3)
    return int(total)


def _fuse_params(hidden_channels: int, include_norm: bool = True) -> int:
    hidden = int(hidden_channels)
    branch_channels = 2 * hidden
    in_channels = 2 * branch_channels
    reduced = in_channels // 2
    transposed = reduced * hidden * 2 * 2 + (2 * hidden if include_norm else 0)
    return int(
        _aspp_params(in_channels, FUSE_ASPP_RATES, include_norm)
        + _conv_gn_act_params(in_channels, reduced, 1, include_norm)
        + _conv_gn_act_params(reduced, reduced, 5, include_norm)
        + transposed
        + _conv_gn_act_params(hidden, hidden, 3, include_norm)
    )


def estimate_fglr_params(
    num_classes: int,
    feature_channels: Sequence[int],
    hidden_channels: int = 16,
    kernel_size: int = 3,
    include_norm: bool = True,
) -> int:
    if len(feature_channels) != 2:
        raise ValueError(f"feature_channels must contain [low_channels, head_channels], got {feature_channels}.")
    low_c, head_c = [int(c) for c in feature_channels]
    n_cls = int(num_classes)
    hidden = int(hidden_channels)
    del kernel_size
    branch = _branch_params(low_c, hidden, include_norm)
    branch += _branch_params(head_c, hidden, include_norm)
    fuse = _fuse_params(hidden, include_norm)
    head = hidden * n_cls + n_cls
    return int(branch + fuse + head)


def estimate_fglr_macs(
    spatial_size: SpatialSize,
    num_classes: int,
    feature_channels: Sequence[int],
    hidden_channels: int = 16,
    kernel_size: int = 3,
) -> int:
    if len(feature_channels) != 2:
        raise ValueError(f"feature_channels must contain [low_channels, head_channels], got {feature_channels}.")
    crop_h, crop_w = _resolve_crop_size(spatial_size, (1, 1))
    low_c, head_c = [int(c) for c in feature_channels]
    n_cls = int(num_classes)
    hidden = int(hidden_channels)
    del kernel_size
    down_h = (crop_h + 1) // 2
    down_w = (crop_w + 1) // 2
    expanded = 2 * hidden
    concat_c = 2 * expanded
    reduced = concat_c // 2
    num_aspp_rates = len(FUSE_ASPP_RATES)
    aspp_macs = (
        down_h * down_w * concat_c * concat_c * (num_aspp_rates * 10 + 3)
        + concat_c * concat_c
    )
    branch_macs = (
        crop_h * crop_w * low_c * expanded
        + down_h * down_w * expanded * expanded * 3 * 3
        + down_h * down_w * expanded * expanded * 5 * 5
        + crop_h * crop_w * head_c * expanded
        + down_h * down_w * expanded * expanded * 3 * 3
        + down_h * down_w * expanded * expanded * 5 * 5
    )
    fuse_macs = (
        down_h * down_w * concat_c * reduced
        + aspp_macs
        + down_h * down_w * reduced * reduced * 5 * 5
        + down_h * down_w * reduced * hidden * 2 * 2
        + crop_h * crop_w * hidden * hidden * 3 * 3
    )
    head_macs = crop_h * crop_w * hidden * n_cls
    return int(branch_macs + fuse_macs + head_macs)


class FGLR(nn.Module):
    """Refine prediction-guided ROIs with encoder-decoder features."""

    def __init__(
        self,
        num_classes: int,
        feature_channels: Sequence[int],
        crop_size: SpatialSize = "input",
        hidden_channels: int = 16,
        kernel_size: int = 3,
        threshold: float = 0.5,
        min_area: int = 16,
        margin_ratio: float = 0.15,
        overlap_mode: str = "mean",
        instance_batch_size: int = 16,
        foreground_channel: int = 1,
        use_checkpoint: bool = True,
        norm_groups: int = 8,
        zero_init_output: bool = True,
        return_stage_logits: bool = False,
    ) -> None:
        super().__init__()
        if len(feature_channels) != 2:
            raise ValueError(
                "FGLR expects exactly two feature channels: "
                f"[low_level_channels, head_feature_channels], got {feature_channels}."
            )
        self.num_classes = int(num_classes)
        self.feature_channels = [int(feature_channels[0]), int(feature_channels[1])]
        self.crop_size = crop_size
        self.hidden_channels = int(hidden_channels)
        self.kernel_size = int(kernel_size)
        self.threshold = float(threshold)
        self.min_area = int(min_area)
        self.margin_ratio = float(margin_ratio)
        self.overlap_mode = str(overlap_mode).lower()
        self.instance_batch_size = max(int(instance_batch_size), 1)
        self.foreground_channel = int(foreground_channel)
        self.use_checkpoint = bool(use_checkpoint)
        self.return_stage_logits = bool(return_stage_logits)

        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}.")
        if self.hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}.")
        if self.overlap_mode != "mean":
            raise ValueError(f"Unsupported overlap_mode={overlap_mode!r}; only 'mean' is implemented.")

        self.low_branch = DownsampleFeatureBranch(
            self.feature_channels[0],
            self.hidden_channels,
            norm_groups=norm_groups,
        )
        self.head_branch = DownsampleFeatureBranch(
            self.feature_channels[1],
            self.hidden_channels,
            norm_groups=norm_groups,
        )
        branch_channels = 2 * self.hidden_channels
        self.fuse = FuseUpsampleBlock(
            2 * branch_channels,
            self.hidden_channels,
            norm_groups=norm_groups,
        )
        self.mask_head = nn.Conv2d(self.hidden_channels, self.num_classes, kernel_size=1)
        if zero_init_output:
            nn.init.zeros_(self.mask_head.weight)
            if self.mask_head.bias is not None:
                nn.init.zeros_(self.mask_head.bias)

    @torch.no_grad()
    def extract_instance_boxes(self, logits: torch.Tensor) -> List[torch.Tensor]:
        probs = _foreground_probability(logits.detach().float(), self.foreground_channel)
        binary = (probs >= self.threshold).to(torch.uint8).cpu().numpy()
        height, width = int(logits.shape[-2]), int(logits.shape[-1])
        boxes: List[torch.Tensor] = []
        for sample_mask in binary:
            sample_boxes = _boxes_from_binary_mask(
                sample_mask,
                min_area=self.min_area,
                margin_ratio=self.margin_ratio,
                height=height,
                width=width,
            )
            if sample_boxes:
                boxes.append(torch.tensor(sample_boxes, dtype=torch.float32, device=logits.device))
            else:
                boxes.append(torch.empty((0, 4), dtype=torch.float32, device=logits.device))
        return boxes

    def forward(
        self,
        logits: torch.Tensor,
        features: Sequence[torch.Tensor],
    ) -> FGLROutput:
        self._check_inputs(logits, features)
        reference_size = (int(logits.shape[-2]), int(logits.shape[-1]))
        crop_hw = _resolve_crop_size(self.crop_size, reference_size)

        # Prediction-guided boxes localize the refinement regions.
        instance_boxes = self.extract_instance_boxes(logits)
        boxes_batched = _flatten_boxes(instance_boxes, logits.device)

        if boxes_batched.numel() == 0:
            refined_logits = logits + self._zero_parameter_dependency(logits)
            return FGLROutput(
                refined_logits=refined_logits,
                delta_logits=torch.zeros_like(logits),
                instance_boxes=instance_boxes,
                stage_logits=[],
            )

        low_feature, head_feature = list(features)

        crop_outputs = []
        for start in range(0, boxes_batched.shape[0], self.instance_batch_size):
            end = min(start + self.instance_batch_size, boxes_batched.shape[0])
            box_chunk = boxes_batched[start:end]
            low_crop = _sample_rois(low_feature, box_chunk, crop_hw, reference_size)
            head_crop = _sample_rois(head_feature, box_chunk, crop_hw, reference_size)

            crop_outputs.append(self._predict_crop_with_optional_checkpoint(low_crop, head_crop))
        crop_logits = torch.cat(crop_outputs, dim=0)

        refined_logits = self._stitch_direct(logits, boxes_batched, crop_logits)
        delta_logits = refined_logits - logits

        return FGLROutput(
            refined_logits=refined_logits,
            delta_logits=delta_logits,
            instance_boxes=instance_boxes,
            stage_logits=[refined_logits] if self.return_stage_logits else [],
        )

    def _predict_crop_with_optional_checkpoint(
        self,
        low_crop: torch.Tensor,
        head_crop: torch.Tensor,
    ) -> torch.Tensor:
        if not self.training or not self.use_checkpoint:
            return self._predict_crop(low_crop, head_crop)
        if not (low_crop.requires_grad or head_crop.requires_grad):
            return self._predict_crop(low_crop, head_crop)
        try:
            return checkpoint(self._predict_crop, low_crop, head_crop, use_reentrant=False)
        except TypeError:
            return checkpoint(self._predict_crop, low_crop, head_crop)

    def _predict_crop(self, low_crop: torch.Tensor, head_crop: torch.Tensor) -> torch.Tensor:
        low_state = self.low_branch(low_crop)
        head_state = self.head_branch(head_crop)
        fused = self.fuse(torch.cat([low_state, head_state], dim=1))
        if fused.shape[-2:] != low_crop.shape[-2:]:
            fused = F.interpolate(
                fused,
                size=low_crop.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
        return self.mask_head(fused)

    def _check_inputs(self, logits: torch.Tensor, features: Sequence[torch.Tensor]) -> None:
        if logits.dim() != 4:
            raise ValueError(f"logits must be BCHW, got shape {tuple(logits.shape)}.")
        if logits.shape[1] != self.num_classes:
            raise ValueError(f"Expected logits with {self.num_classes} channels, got {logits.shape[1]}.")
        if len(features) != 2:
            raise ValueError(
                "FGLR expects exactly two features: "
                "[low_level_feature, head_feature]."
            )
        for idx, (feature, channels) in enumerate(zip(features, self.feature_channels)):
            if feature.dim() != 4:
                raise ValueError(f"features[{idx}] must be BCHW, got {tuple(feature.shape)}.")
            if feature.shape[0] != logits.shape[0]:
                raise ValueError(
                    f"features[{idx}] batch size {feature.shape[0]} does not match "
                    f"logits batch size {logits.shape[0]}."
                )
            if feature.shape[1] != channels:
                raise ValueError(
                    f"features[{idx}] expected {channels} channels, got {feature.shape[1]}."
                )

    def _zero_parameter_dependency(self, logits: torch.Tensor) -> torch.Tensor:
        zero = logits.new_zeros(())
        for param in self.parameters():
            zero = zero + param.sum().to(dtype=logits.dtype) * 0.0
        return zero

    def _stitch_direct(
        self,
        base_logits: torch.Tensor,
        boxes_batched: torch.Tensor,
        crop_logits: torch.Tensor,
    ) -> torch.Tensor:
        return self._stitch_crops(base_logits, boxes_batched, crop_logits, default=base_logits)

    def _stitch_crops(
        self,
        base_logits: torch.Tensor,
        boxes_batched: torch.Tensor,
        crop_values: torch.Tensor,
        default: torch.Tensor | None,
    ) -> torch.Tensor:
        full = torch.zeros_like(base_logits)
        count = base_logits.new_zeros(
            (base_logits.shape[0], 1, base_logits.shape[-2], base_logits.shape[-1])
        )
        boxes_int = boxes_batched.round().long()
        for crop_idx in range(boxes_int.shape[0]):
            b, y0, x0, y1, x1 = boxes_int[crop_idx].tolist()
            y0 = max(0, min(y0, base_logits.shape[-2]))
            y1 = max(0, min(y1, base_logits.shape[-2]))
            x0 = max(0, min(x0, base_logits.shape[-1]))
            x1 = max(0, min(x1, base_logits.shape[-1]))
            if y1 <= y0 or x1 <= x0:
                continue
            patch = F.interpolate(
                crop_values[crop_idx : crop_idx + 1],
                size=(y1 - y0, x1 - x0),
                mode="bilinear",
                align_corners=True,
            )[0]
            full[b, :, y0:y1, x0:x1] += patch
            count[b, :, y0:y1, x0:x1] += 1.0
        has_value = count > 0
        full = full / count.clamp_min(1.0)
        if default is not None:
            full = torch.where(has_value.expand_as(full), full, default)
        return full

    def estimate_macs(self, spatial_size: SpatialSize) -> int:
        return estimate_fglr_macs(
            spatial_size=spatial_size,
            num_classes=self.num_classes,
            feature_channels=self.feature_channels,
            hidden_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
        )


__all__ = [
    "ConvNormAct",
    "ConvGNAct",
    "DownsampleFeatureBranch",
    "FGLR",
    "FGLROutput",
    "FuseUpsampleBlock",
    "TransposeConvGNAct",
    "estimate_fglr_macs",
    "estimate_fglr_params",
]
