"""Atrous spatial pyramid pooling blocks."""

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _valid_num_groups(channels: int, requested: int) -> int:
    requested = max(int(requested), 1)
    channels = int(channels)
    for groups in range(min(requested, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        dilation: int = 1,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                int(in_channels),
                int(out_channels),
                kernel_size=int(kernel_size),
                padding=int(padding),
                dilation=int(dilation),
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


class ASPP(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        rates: Sequence[int] = (1, 2, 4),
        norm_groups: int = 8,
        dropout: float = 0.0,
        residual: bool = False,
    ) -> None:
        super().__init__()
        in_channels = int(in_channels)
        out_channels = in_channels if out_channels is None else int(out_channels)
        rates = tuple(int(rate) for rate in rates)
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}.")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}.")
        if not rates:
            raise ValueError("rates must contain at least one dilation rate.")
        if any(rate <= 0 for rate in rates):
            raise ValueError(f"rates must be positive, got {rates}.")

        self.residual = bool(residual)
        self.branch1 = ConvGNAct(
            in_channels,
            out_channels,
            kernel_size=1,
            norm_groups=norm_groups,
        )
        self.branches = nn.ModuleList([
            ConvGNAct(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=rate,
                dilation=rate,
                norm_groups=norm_groups,
            )
            for rate in rates
        ])
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvGNAct(in_channels, out_channels, kernel_size=1, norm_groups=norm_groups),
        )
        concat_channels = out_channels * (len(rates) + 2)
        projection = [
            ConvGNAct(concat_channels, out_channels, kernel_size=1, norm_groups=norm_groups)
        ]
        if dropout > 0.0:
            projection.append(nn.Dropout2d(p=float(dropout)))
        self.project = nn.Sequential(*projection)

        if in_channels == out_channels:
            self.residual_proj: nn.Module = nn.Identity()
        else:
            self.residual_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        pooled = F.interpolate(
            self.image_pool(x),
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        features = [self.branch1(x), *[branch(x) for branch in self.branches], pooled]
        out = self.project(torch.cat(features, dim=1))
        if self.residual:
            out = out + self.residual_proj(x)
        return out
