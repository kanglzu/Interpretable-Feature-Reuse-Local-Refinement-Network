"""Training output naming helpers."""

from __future__ import annotations

import re


def slug_name(name: object) -> str:
    raw = str(name or "model").strip()
    compact = re.sub(r"[^A-Za-z0-9]+", "", raw).lower()
    aliases = {
        "fglrswinumamba": "FGLRSwinUMamba",
    }
    if compact in aliases:
        return aliases[compact]
    raw = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_")
    return raw or "model"


def experiment_name(cfg: dict) -> str:
    output_cfg = cfg.get("output", {})
    if output_cfg.get("experiment_name"):
        return str(output_cfg["experiment_name"])

    model_cfg = cfg.get("model", {})
    model_name = str(model_cfg.get("name", "model"))
    return slug_name(model_name)


def fold_output_name(cfg: dict, fold: int) -> str:
    return f"{experiment_name(cfg)}_fold{int(fold)}"
