"""Model builder."""

from __future__ import annotations

from copy import deepcopy

from nets.SwinUMamba import (
    FGLRSwinUMamba,
    _HAS_DNA,
    _InitWeightsHe,
    _init_non_vssm_modules,
)


def build_model(cfg: dict) -> FGLRSwinUMamba:
    model_cfg = deepcopy(cfg["model"])
    name = str(model_cfg.pop("name", "FGLRSwinUMamba"))
    if name != "FGLRSwinUMamba":
        raise ValueError(f"This release only supports model.name=FGLRSwinUMamba, got {name!r}.")

    model_cfg.pop("ds_weights", None)
    fglr_cfg = deepcopy(model_cfg.pop("fglr", {}))
    model = FGLRSwinUMamba(
        in_channels=int(model_cfg.pop("in_channels")),
        num_classes=int(model_cfg.pop("num_classes")),
        deep_supervision=bool(model_cfg.pop("deep_supervision", True)),
        feat_size=model_cfg.pop("feat_size", [48, 96, 192, 384, 768]),
        hidden_size=int(model_cfg.pop("hidden_size", 768)),
        depths=model_cfg.pop("depths", [2, 2, 9, 2]),
        d_state=int(model_cfg.pop("d_state", 16)),
        drop_path_rate=float(model_cfg.pop("drop_path_rate", 0.2)),
        norm_name=str(model_cfg.pop("norm_name", "instance")),
        da_ss2d_group_size=int(model_cfg.pop("da_ss2d_group_size", 64)),
        da_ss2d_hidden_size=model_cfg.pop("da_ss2d_hidden_size", None),
        fglr_cfg=fglr_cfg,
        build_fglr=False,
    )
    if model_cfg:
        unknown = ", ".join(sorted(model_cfg))
        raise TypeError(f"Unknown model config keys: {unknown}")

    _init_non_vssm_modules(model.backbone, _InitWeightsHe(1e-2))
    if _HAS_DNA:
        from dynamic_network_architectures.initialization.weight_init import (
            init_last_bn_before_add_to_0,
        )

        _init_non_vssm_modules(model.backbone, init_last_bn_before_add_to_0)
    model.build_fglr(fglr_cfg)
    return model


__all__ = ["build_model"]
