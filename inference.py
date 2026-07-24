"""Evaluate one checkpoint on its corresponding validation split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from training.naming import fold_output_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one checkpoint.")
    parser.add_argument("--config", default="configs/maskpolish_swinumamba.yaml")
    parser.add_argument("--weights-root", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--splits", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-hd95", action="store_true")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(Path(path).expanduser().resolve(), encoding="utf-8") as file:
        cfg = yaml.safe_load(file) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping.")
    return cfg


def resolve_sample_paths(sample_dict: Dict[str, dict], manifest_path: str) -> None:
    manifest_dir = Path(manifest_path).resolve().parent
    for entry in sample_dict.values():
        for key in ("image", "label"):
            path = Path(entry[key])
            if not path.is_absolute():
                entry[key] = str((manifest_dir / path).resolve())


def default_checkpoint_path(weights_root: Path, cfg: dict, fold: int) -> Path:
    return weights_root / fold_output_name(cfg, fold) / "checkpoint_best.pth"


def load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)


@torch.inference_mode()
def evaluate(
    cfg: dict,
    fold: int,
    checkpoint: Path,
    manifest_path: str,
    splits_path: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    compute_hd95: bool,
) -> Dict[str, float]:
    from data.dataset import SegDataset
    from data.split import get_sample_dict, load_manifest, load_splits
    from data.transforms import get_val_transforms
    from nets.build import build_model
    from training.metrics import aggregate_metrics

    manifest = load_manifest(manifest_path)
    sample_dict = get_sample_dict(manifest)
    resolve_sample_paths(sample_dict, manifest_path)
    _, val_ids = load_splits(splits_path, fold)
    num_channels = len(manifest.get("channel_names", {"0": ""}))
    dataset = SegDataset(
        val_ids,
        sample_dict,
        transform=get_val_transforms(list(cfg["data"].get("image_size", [256, 256])), num_channels),
        num_channels=num_channels,
        cache_in_memory=bool(cfg["data"].get("cache_in_memory", False)),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint for fold {fold} was not found: {checkpoint}")

    model = build_model(cfg).to(device)
    load_checkpoint(model, checkpoint, device)
    model.eval()

    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        predictions.extend(torch.argmax(logits, dim=1).cpu().numpy().astype(np.int32))
        targets.extend(batch["mask"].cpu().numpy().astype(np.int32))

    metrics = aggregate_metrics(
        predictions,
        targets,
        num_classes=int(manifest.get("num_classes", cfg["model"]["num_classes"])),
        compute_hd=compute_hd95,
    )
    metrics["fold"] = fold
    metrics["samples"] = len(dataset)
    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    fields = [f"fold={int(metrics['fold'])}", f"samples={int(metrics['samples'])}"]
    for name in ("dice", "iou", "hd95"):
        if name in metrics:
            fields.append(f"{name}={metrics[name]:.4f}")
    return "  ".join(fields)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    manifest_path = args.manifest or cfg["data"]["manifest_json"]
    splits_path = args.splits or cfg["data"]["splits_json"]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    batch_size = args.batch_size or int(cfg["data"].get("val_batch_size") or cfg["training"]["batch_size"])
    num_workers = int(cfg["data"].get("num_workers", 4) if args.num_workers is None else args.num_workers)
    compute_hd95 = not args.skip_hd95
    weights_root = Path(args.weights_root).expanduser().resolve()

    checkpoint = default_checkpoint_path(weights_root, cfg, args.fold)
    result = evaluate(
        cfg,
        args.fold,
        checkpoint,
        manifest_path,
        splits_path,
        batch_size,
        num_workers,
        device,
        compute_hd95,
    )
    print(format_metrics(result))

    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
