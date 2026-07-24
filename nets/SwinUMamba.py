"""SwinUMamba backbone with DA-SS2D and FGLR."""

import math
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import repeat
from timm.models.layers import DropPath, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

from nets.blocks.fglr import FGLR

try:
    from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
    _HAS_DNA = True
except ImportError:
    _HAS_DNA = False


class PatchEmbed2D(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, H // 2, W // 2, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class DASS2D(nn.Module):
    """Direction-Adaptive SS2D (DA-SS2D)."""
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        use_reference_scan=False,
        da_ss2d_enabled=False,
        da_ss2d_group_size=48,
        da_ss2d_hidden_size=None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.da_ss2d_enabled = bool(da_ss2d_enabled)
        self.da_ss2d_group_size = int(da_ss2d_group_size)
        if self.da_ss2d_group_size <= 0:
            raise ValueError(
                f"da_ss2d_group_size must be positive, got {da_ss2d_group_size}."
            )
        if self.d_inner % self.da_ss2d_group_size != 0:
            raise ValueError(
                "da_ss2d_group_size must divide d_inner, got "
                f"d_inner={self.d_inner}, group_size={self.da_ss2d_group_size}."
            )
        self.da_ss2d_num_groups = self.d_inner // self.da_ss2d_group_size
        self.da_ss2d_hidden_size = (
            self.da_ss2d_group_size
            if da_ss2d_hidden_size is None
            else int(da_ss2d_hidden_size)
        )
        if self.da_ss2d_hidden_size <= 0:
            raise ValueError(
                f"da_ss2d_hidden_size must be positive, got {da_ss2d_hidden_size}."
            )

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_ref if use_reference_scan else selective_scan_fn
        if self.da_ss2d_enabled:
            da_ss2d_in = 4 * self.da_ss2d_group_size
            self.da_ss2d = nn.Sequential(
                nn.Linear(
                    da_ss2d_in,
                    self.da_ss2d_hidden_size,
                    bias=True,
                    **factory_kwargs,
                ),
                nn.SiLU(),
                nn.Linear(
                    self.da_ss2d_hidden_size,
                    da_ss2d_in,
                    bias=True,
                    **factory_kwargs,
                ),
            )
            nn.init.zeros_(self.da_ss2d[-1].weight)
            nn.init.zeros_(self.da_ss2d[-1].bias)
            self.da_ss2d[-1].weight._no_reinit = True
            self.da_ss2d[-1].bias._no_reinit = True

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([
            x.view(B, -1, L),
            torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L),
        ], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def apply_da_ss2d(self, y1: torch.Tensor, y2: torch.Tensor, y3: torch.Tensor, y4: torch.Tensor) -> torch.Tensor:
        """Fuse four scan directions with DA-SS2D."""
        if not self.da_ss2d_enabled:
            return y1 + y2 + y3 + y4

        B, C, L = y1.shape
        group_size = self.da_ss2d_group_size
        num_groups = self.da_ss2d_num_groups

        ys = torch.stack([y1, y2, y3, y4], dim=1)
        desc = ys.mean(dim=-1)
        desc = desc.view(B, 4, num_groups, group_size)
        desc = desc.permute(0, 2, 1, 3).contiguous().view(B * num_groups, 4 * group_size)

        logits = self.da_ss2d(desc)
        logits = logits.view(B, num_groups, 4, group_size)
        weights = F.softmax(logits, dim=2)
        weights = weights.permute(0, 2, 1, 3).contiguous().view(B, 4, C, 1)
        return (ys * weights).sum(dim=1) * 4.0

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = self.apply_da_ss2d(y1, y2, y3, y4)
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = DASS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + self.drop_path(self.self_attention(self.ln_1(input)))


class VSSLayer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        attn_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                **kwargs,
            )
            for i in range(depth)
        ])

        def _init_weights(module: nn.Module):
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    p = p.clone().detach_()
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class VSSMEncoder(nn.Module):
    def __init__(
        self,
        patch_size=4,
        in_chans=3,
        depths=(2, 2, 9, 2),
        dims=(96, 192, 384, 768),
        d_state=16,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i) for i in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )

        self.ape = False
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                **kwargs,
            )
            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                self.downsamples.append(PatchMerging2D(dim=dims[i_layer], norm_layer=norm_layer))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            if not getattr(m.weight, "_no_reinit", False):
                trunc_normal_(m.weight, std=.02)
            if m.bias is not None and not getattr(m.bias, "_no_reinit", False):
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x_ret = [x]
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for s, layer in enumerate(self.layers):
            x = layer(x)
            x_ret.append(x.permute(0, 3, 1, 2))
            if s < len(self.downsamples):
                x = self.downsamples[s](x)

        return x_ret


class _InitWeightsHe:
    def __init__(self, neg_slope: float = 1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module: nn.Module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


def _is_mamba_ssm_module(module: nn.Module) -> bool:
    cls = module.__class__
    return cls.__name__ == "Mamba" and cls.__module__.startswith("mamba_ssm")


def _init_non_vssm_modules(model: nn.Module, init_fn: Callable[[nn.Module], None]) -> None:

    def _apply(module: nn.Module) -> None:
        if _is_mamba_ssm_module(module):
            return
        init_fn(module)
        for child in module.children():
            _apply(child)

    for name, module in model.named_children():
        if name == "vssm_encoder":
            continue
        _apply(module)


def _make_norm(name: str, c: int) -> nn.Module:
    if name == "instance":
        return nn.InstanceNorm2d(c, eps=1e-5, affine=True)
    if name == "batch":
        return nn.BatchNorm2d(c, eps=1e-5, affine=True)
    raise ValueError(f"Unsupported norm_name='{name}'.")


class UnetrBasicBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3, stride: int = 1, norm_name: str = "instance"):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=pad, bias=False)
        self.norm1 = _make_norm(norm_name, out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride=1, padding=pad, bias=False)
        self.norm2 = _make_norm(norm_name, out_c)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        if (in_c != out_c) or (stride != 1):
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                _make_norm(norm_name, out_c),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + self.shortcut(x))


class UnetrUpBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel_size: int = 3, upsample_kernel_size: int = 2, norm_name: str = "instance"):
        super().__init__()
        self.transp_conv = nn.ConvTranspose2d(in_c, out_c, kernel_size=upsample_kernel_size, stride=upsample_kernel_size, bias=False)
        self.conv_block = UnetrBasicBlock(out_c * 2, out_c, kernel_size=kernel_size, stride=1, norm_name=norm_name)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.conv_block(torch.cat((self.transp_conv(x), skip), dim=1))


class UnetOutBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, dropout: Optional[float] = None):
        super().__init__()
        self.dropout = nn.Dropout2d(p=dropout) if dropout else None
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        return self.conv(x)


class SwinUMamba(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        feat_size: Sequence[int] = (48, 96, 192, 384, 768),
        hidden_size: int = 768,
        depths: Sequence[int] = (2, 2, 9, 2),
        d_state: int = 16,
        drop_path_rate: float = 0.2,
        norm_name: str = "instance",
        deep_supervision: bool = False,
        da_ss2d_enabled: bool = False,
        da_ss2d_group_size: int = 48,
        da_ss2d_hidden_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        feat_size = list(feat_size)
        if len(feat_size) != 5:
            raise ValueError(f"feat_size must have length 5, got {len(feat_size)}.")
        if len(depths) != 4:
            raise ValueError(f"depths must have length 4, got {len(depths)}.")
        if hidden_size != feat_size[-1]:
            raise ValueError(f"hidden_size ({hidden_size}) must equal feat_size[-1] ({feat_size[-1]}).")

        self.deep_supervision = deep_supervision
        c0, c1, c2, c3, c4 = feat_size

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c0, kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm2d(c0, eps=1e-5, affine=True),
        )
        self.vssm_encoder = VSSMEncoder(
            patch_size=2, in_chans=c0, depths=list(depths),
            dims=feat_size[1:], d_state=d_state, drop_path_rate=drop_path_rate,
            da_ss2d_enabled=da_ss2d_enabled,
            da_ss2d_group_size=da_ss2d_group_size,
            da_ss2d_hidden_size=da_ss2d_hidden_size,
        )

        self.encoder1 = UnetrBasicBlock(in_channels, c0, norm_name=norm_name)
        self.encoder2 = UnetrBasicBlock(c0, c1, norm_name=norm_name)
        self.encoder3 = UnetrBasicBlock(c1, c2, norm_name=norm_name)
        self.encoder4 = UnetrBasicBlock(c2, c3, norm_name=norm_name)
        self.encoder5 = UnetrBasicBlock(c3, c4, norm_name=norm_name)

        self.decoder6 = UnetrUpBlock(hidden_size, c4, norm_name=norm_name)
        self.decoder5 = UnetrUpBlock(c4, c3, norm_name=norm_name)
        self.decoder4 = UnetrUpBlock(c3, c2, norm_name=norm_name)
        self.decoder3 = UnetrUpBlock(c2, c1, norm_name=norm_name)
        self.decoder2 = UnetrUpBlock(c1, c0, norm_name=norm_name)
        self.decoder1 = UnetrBasicBlock(c0, c0, norm_name=norm_name)

        self.out_layer0 = UnetOutBlock(c0, num_classes)
        self.out_layer1 = UnetOutBlock(c1, num_classes)
        self.out_layer2 = UnetOutBlock(c2, num_classes)
        self.out_layer3 = UnetOutBlock(c3, num_classes)

    def forward_with_refinement_features(
        self,
        x_in: torch.Tensor,
    ) -> tuple[Union[torch.Tensor, List[torch.Tensor]], List[torch.Tensor]]:
        x1 = self.stem(x_in)
        v0, v1, v2, v3, v4 = self.vssm_encoder(x1)

        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(v0)
        enc3 = self.encoder3(v1)
        enc4 = self.encoder4(v2)
        enc5 = self.encoder5(v3)

        dec4 = self.decoder6(v4, enc5)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        dec_out = self.decoder1(dec0)
        self._last_fglr_features = [x1, dec_out]

        if self.deep_supervision:
            segmentation = [
                self.out_layer0(dec_out),
                self.out_layer1(dec1),
                self.out_layer2(dec2),
                self.out_layer3(dec3),
            ]
        else:
            segmentation = self.out_layer0(dec_out)
        return segmentation, [v0, v1, v2]

    def forward_with_fglr_features(
        self,
        x_in: torch.Tensor,
    ) -> tuple[Union[torch.Tensor, List[torch.Tensor]], List[torch.Tensor]]:
        segmentation, _ = self.forward_with_refinement_features(x_in)
        return segmentation, self._last_fglr_features

    def forward(self, x_in: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        segmentation, _ = self.forward_with_refinement_features(x_in)
        return segmentation

    @torch.no_grad()
    def freeze_encoder(self) -> None:
        for name, p in self.vssm_encoder.named_parameters():
            if "patch_embed" not in name:
                p.requires_grad = False

    @torch.no_grad()
    def unfreeze_encoder(self) -> None:
        for p in self.vssm_encoder.parameters():
            p.requires_grad = True


class FGLRSwinUMamba(nn.Module):

    cam_target = "backbone.decoder1"

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        deep_supervision: bool = True,
        feat_size: Sequence[int] = (48, 96, 192, 384, 768),
        hidden_size: int = 768,
        depths: Sequence[int] = (2, 2, 9, 2),
        d_state: int = 16,
        drop_path_rate: float = 0.2,
        norm_name: str = "instance",
        da_ss2d_group_size: int = 64,
        da_ss2d_hidden_size: Optional[int] = None,
        fglr_cfg: Optional[Dict] = None,
        build_fglr: bool = True,
    ) -> None:
        super().__init__()
        feat_size = list(feat_size)
        self.num_classes = int(num_classes)
        self.fglr_feature_channels = [feat_size[0], feat_size[0]]
        self._fglr_cfg = dict(fglr_cfg or {})
        self.backbone = SwinUMamba(
            in_channels=in_channels,
            num_classes=num_classes,
            feat_size=feat_size,
            hidden_size=hidden_size,
            depths=depths,
            d_state=d_state,
            drop_path_rate=drop_path_rate,
            norm_name=norm_name,
            deep_supervision=deep_supervision,
            da_ss2d_enabled=True,
            da_ss2d_group_size=da_ss2d_group_size,
            da_ss2d_hidden_size=da_ss2d_hidden_size,
        )
        self.fglr: Optional[FGLR] = None
        self._fglr_runtime_enabled = True
        self._last_aux: Dict[str, object] = {}
        if build_fglr:
            self.build_fglr()

    def build_fglr(self, fglr_cfg: Optional[Dict] = None) -> None:
        if fglr_cfg is not None:
            self._fglr_cfg = dict(fglr_cfg)
        cfg = dict(self._fglr_cfg)
        self.fglr = FGLR(
            num_classes=self.num_classes,
            feature_channels=self.fglr_feature_channels,
            crop_size=cfg.pop("crop_size", 128),
            hidden_channels=cfg.pop("hidden_channels", 24),
            threshold=cfg.pop("threshold", 0.5),
            min_area=cfg.pop("min_area", 16),
            margin_ratio=cfg.pop("margin_ratio", 0.1),
            overlap_mode=cfg.pop("overlap_mode", "mean"),
            instance_batch_size=cfg.pop("instance_batch_size", 16),
            foreground_channel=cfg.pop("foreground_channel", 1),
            use_checkpoint=cfg.pop("use_checkpoint", True),
            zero_init_output=cfg.pop("zero_init_output", True),
            return_stage_logits=cfg.pop("return_stage_logits", False),
        )
        if cfg:
            unknown = ", ".join(sorted(cfg))
            raise TypeError(f"Unknown fglr options: {unknown}")

    def set_fglr_enabled(self, enabled: bool) -> None:
        self._fglr_runtime_enabled = bool(enabled)

    def forward(self, x_in: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        base_outputs, features = self.backbone.forward_with_fglr_features(x_in)
        base_logits = base_outputs[0] if isinstance(base_outputs, (list, tuple)) else base_outputs
        self._last_aux = {
            "base_outputs": base_outputs,
            "base_logits": base_logits,
            "delta_logits": None,
            "instance_boxes": [],
            "fglr_active": self._fglr_runtime_enabled,
            "stage_logits": [],
        }
        if self.fglr is None:
            raise RuntimeError("FGLR has not been built. Call build_fglr() after backbone init.")
        if not self._fglr_runtime_enabled:
            zero = self._zero_fglr_dependency(base_logits)
            return self._add_zero_dependency(base_outputs, zero)

        fglr_output = self.fglr(logits=base_logits, features=features)
        self._last_aux.update(
            {
                "delta_logits": fglr_output.delta_logits,
                "instance_boxes": fglr_output.instance_boxes,
                "stage_logits": fglr_output.stage_logits,
            }
        )
        return fglr_output.refined_logits

    def get_aux_outputs(self) -> Dict[str, object]:
        return dict(self._last_aux)

    def _zero_fglr_dependency(self, logits: torch.Tensor) -> torch.Tensor:
        zero = logits.new_zeros(())
        for param in self.fglr.parameters():
            zero = zero + param.sum().to(dtype=logits.dtype) * 0.0
        return zero

    @staticmethod
    def _add_zero_dependency(outputs, zero: torch.Tensor):
        if isinstance(outputs, tuple):
            return tuple(item + zero for item in outputs)
        if isinstance(outputs, list):
            return [item + zero for item in outputs]
        return outputs + zero


def build_swin_umamba(
    in_channels: int,
    num_classes: int,
    deep_supervision: bool = False,
    feat_size: Sequence[int] = (48, 96, 192, 384, 768),
    hidden_size: int = 768,
    depths: Sequence[int] = (2, 2, 9, 2),
    d_state: int = 16,
    drop_path_rate: float = 0.2,
    norm_name: str = "instance",
    da_ss2d_enabled: bool = False,
    da_ss2d_group_size: int = 48,
    da_ss2d_hidden_size: Optional[int] = None,
    **_kwargs,
) -> SwinUMamba:
    model = SwinUMamba(
        in_channels=in_channels, num_classes=num_classes,
        feat_size=feat_size, hidden_size=hidden_size,
        depths=depths, d_state=d_state, drop_path_rate=drop_path_rate,
        norm_name=norm_name, deep_supervision=deep_supervision,
        da_ss2d_enabled=da_ss2d_enabled,
        da_ss2d_group_size=da_ss2d_group_size,
        da_ss2d_hidden_size=da_ss2d_hidden_size,
    )
    _init_non_vssm_modules(model, _InitWeightsHe(1e-2))
    if _HAS_DNA:
        _init_non_vssm_modules(model, init_last_bn_before_add_to_0)
    return model
