"""Training entry point for FGLR-SwinUMamba."""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, get_worker_info

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data.dataset import SegDataset
from data.split import get_sample_dict, load_manifest, load_splits
from data.transforms import get_train_transforms, get_val_transforms
from nets.build import build_model
from training.cam_loss import CAMLoss, resolve_layer
from training.logger import TrainingLogger
from training.loss import build_loss
from training.metrics import EMATracker, aggregate_metrics
from training.naming import fold_output_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MaskPolish-SwinUMamba.")
    parser.add_argument("--config", default="configs/maskpolish_swinumamba.yaml")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument(
        "--cfg-options",
        nargs="*",
        default=[],
        help="Override config values, e.g. data.manifest_json=/data/manifest.json",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    path = Path(config_path).expanduser().resolve()
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return cfg


def apply_overrides(cfg: dict, overrides: List[str]) -> dict:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override, expected key=value: {item}")
        key_path, value_text = item.split("=", 1)
        node = cfg
        keys = key_path.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = _cast_value(value_text)
        print(f"[Override] {key_path} = {node[keys[-1]]!r}")
    return cfg


def _cast_value(value: str):
    low = value.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    try:
        torch.use_deterministic_algorithms(bool(deterministic), warn_only=bool(deterministic))
    except Exception:
        pass


def _is_main_process() -> bool:
    return os.environ.get("LOCAL_RANK", "0") == "0"


def _resolve_sample_paths(sample_dict: Dict[str, dict], manifest_json: str) -> None:
    manifest_dir = Path(manifest_json).resolve().parent
    for entry in sample_dict.values():
        for key in ("image", "label"):
            path = Path(entry[key])
            if path.is_absolute():
                entry[key] = str(path)
                continue
            candidate = (manifest_dir / path).resolve()
            if not candidate.exists():
                candidate = (PROJECT_DIR / path).resolve()
            entry[key] = str(candidate)


class MaskPolishTrainer:

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.fold = int(cfg["training"].get("fold", 0))
        self.num_epochs = int(cfg["training"]["num_epochs"])
        self.current_epoch = 0

        self._setup_distributed()
        self.seed = int(cfg["training"].get("seed", 42))
        self.deterministic = bool(cfg["training"].get("deterministic", True))
        _set_seed(self.seed, deterministic=self.deterministic)

        self.output_dir = Path(cfg["output"]["save_dir"]) / fold_output_name(cfg, self.fold)
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.use_ddp:
            dist.barrier()

        self._build_data()
        self.network = build_model(cfg).to(self.device)
        if self.use_ddp:
            self._make_param_layout_contiguous()
            dist.barrier()
            self.network = DDP(
                self.network,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )
            _set_seed(self.seed + self.rank, deterministic=self.deterministic)
        self.loss_fn = build_loss(cfg).to(self.device)
        self._build_optimizer()
        self.ema = EMATracker(alpha=float(cfg["training"].get("ema_alpha", 0.9)))
        self.logger = TrainingLogger(str(self.output_dir), self.num_epochs) if self.is_main else None

        tr = cfg["training"]
        self.base_loss_alpha = float(tr.get("base_loss_alpha", 1.0))
        self.fglr_loss_alpha = float(tr.get("fglr_loss_alpha", 1.0))
        self.fglr_start_epoch = int(tr.get("fglr_start_epoch", 0))
        self.grad_cam_alpha = float(tr.get("grad_cam_alpha", 0.0))
        self.layer_cam_gamma = float(tr.get("layer_cam_gamma", 0.0))
        self.cam_loss_start_threshold = tr.get("cam_loss_start_threshold", None)
        self.cam_loss_start_threshold = (
            None if self.cam_loss_start_threshold is None else float(self.cam_loss_start_threshold)
        )
        self.cam_loss_threshold_metric = str(tr.get("cam_loss_threshold_metric", "dice"))
        self._cam_loss_active = self.cam_loss_start_threshold is None
        self._cam_activation: Optional[torch.Tensor] = None
        self._cam_hook_handle = None
        self._cam_loss = CAMLoss(
            smooth=float(tr.get("cam_eps", 1e-6)),
            foreground_channel=int(tr.get("cam_foreground_channel", 1)),
        )
        if self._configured_cam_loss():
            self._register_cam_hook(str(tr.get("cam_target") or ""))

        self._epoch_base_loss_sum = 0.0
        self._epoch_fglr_loss_sum = 0.0
        self._epoch_base_loss_count = 0
        self._epoch_fglr_loss_count = 0
        self._epoch_grad_cam_loss_sum = 0.0
        self._epoch_layer_cam_loss_sum = 0.0
        self._epoch_cam_loss_count = 0
        self.best_epoch = -1
        self.best_dice = -1.0
        self.min_delta = float(cfg["training"].get("min_delta", 1e-4))

        if self.is_main:
            total = sum(p.numel() for p in self._network_module.parameters())
            trainable = sum(p.numel() for p in self._network_module.parameters() if p.requires_grad)
            print(f"[Model] FGLRSwinUMamba params={total/1e6:.2f}M trainable={trainable/1e6:.2f}M")
            batch_size = int(cfg["training"]["batch_size"])
            print(
                f"[Train] batch/rank={batch_size}  accum=1  "
                f"effective_global_batch={batch_size * self.world_size}  amp=False"
            )

    @property
    def _network_module(self):
        return self.network.module if self.use_ddp else self.network

    def _setup_distributed(self) -> None:
        if "LOCAL_RANK" in os.environ:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            dist.init_process_group(backend="nccl")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.use_ddp = self.world_size > 1
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.local_rank = 0
            self.rank = 0
            self.world_size = 1
            self.use_ddp = False
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_main = self.rank == 0
        if self.is_main:
            mode = f"DDP world_size={self.world_size}" if self.use_ddp else "single process"
            print(f"[Runtime] {mode} device={self.device}")

    def _make_param_layout_contiguous(self) -> None:
        with torch.no_grad():
            for param in self.network.parameters():
                if not param.is_contiguous():
                    param.data = param.data.contiguous()
            for buffer in self.network.buffers():
                if buffer.is_floating_point() and not buffer.is_contiguous():
                    buffer.data = buffer.data.contiguous()

    def _build_data(self) -> None:
        data_cfg = self.cfg["data"]
        manifest = load_manifest(data_cfg["manifest_json"])
        sample_dict = get_sample_dict(manifest)
        _resolve_sample_paths(sample_dict, data_cfg["manifest_json"])
        train_ids, val_ids = load_splits(data_cfg["splits_json"], self.fold)
        image_size = list(data_cfg.get("image_size", [256, 256]))
        num_channels = len(manifest.get("channel_names", {"0": ""}))
        cache = bool(data_cfg.get("cache_in_memory", False))
        seeded_transforms = self.seed is not None
        train_transform = get_train_transforms(image_size, num_channels) if not seeded_transforms else None
        val_transform = get_val_transforms(image_size, num_channels) if not seeded_transforms else None
        train_transform_factory = partial(get_train_transforms, image_size, num_channels) if seeded_transforms else None
        val_transform_factory = partial(get_val_transforms, image_size, num_channels) if seeded_transforms else None

        train_ds = SegDataset(
            train_ids,
            sample_dict,
            transform=train_transform,
            transform_factory=train_transform_factory,
            num_channels=num_channels,
            cache_in_memory=cache,
        )
        val_ds = SegDataset(
            val_ids,
            sample_dict,
            transform=val_transform,
            transform_factory=val_transform_factory,
            num_channels=num_channels,
            cache_in_memory=cache,
        )
        batch_size = int(self.cfg["training"]["batch_size"])
        val_batch = data_cfg.get("val_batch_size") or batch_size
        num_workers = int(data_cfg.get("num_workers", 4))
        pin_memory = bool(data_cfg.get("pin_memory", True)) and self.device.type == "cuda"
        persistent = bool(data_cfg.get("persistent_workers", True)) and num_workers > 0

        if self.use_ddp:
            train_sampler: Optional[DistributedSampler] = DistributedSampler(
                train_ds,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=self.seed,
                drop_last=True,
            )
            val_rank_indices = list(range(self.rank, len(val_ds), self.world_size))
            val_ds_for_loader = torch.utils.data.Subset(val_ds, val_rank_indices)
            train_shuffle = False
        else:
            train_sampler = None
            val_ds_for_loader = val_ds
            train_shuffle = True

        train_generator = self._make_loader_generator(offset=101)
        val_generator = self._make_loader_generator(offset=202)
        if seeded_transforms and num_workers == 0:
            train_ds.set_worker_seed(int(self.seed) + (self.rank * 1000) + 11)
            val_ds.set_worker_seed(int(self.seed) + (self.rank * 1000) + 22)
        prefetch = 2 if num_workers > 0 else None

        self.train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            generator=train_generator,
            worker_init_fn=self._seed_data_worker,
            persistent_workers=persistent,
            prefetch_factor=prefetch,
        )
        self.val_loader = DataLoader(
            val_ds_for_loader,
            batch_size=int(val_batch),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=val_generator,
            worker_init_fn=self._seed_data_worker,
            persistent_workers=persistent,
            prefetch_factor=prefetch,
        )
        self.num_classes = int(manifest.get("num_classes", self.cfg["model"]["num_classes"]))
        if self.is_main:
            print(f"[Data] fold={self.fold} train={len(train_ds)} val={len(val_ds)}")

    def _build_optimizer(self) -> None:
        tr = self.cfg["training"]
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=float(tr["initial_lr"]),
            weight_decay=float(tr.get("weight_decay", 1e-4)),
            eps=1e-5,
            betas=(0.9, 0.999),
        )
        warmup_epochs = int(tr.get("warmup_epochs", 0))
        t_max = max(self.num_epochs - warmup_epochs, 1)
        main = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=t_max,
            eta_min=float(tr.get("eta_min", 1e-5)),
        )
        if warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda ep: float(ep + 1) / float(warmup_epochs) if ep < warmup_epochs else 1.0,
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup, main],
                milestones=[warmup_epochs],
            )
        else:
            self.scheduler = main

    def _fglr_should_be_active(self) -> bool:
        return self.current_epoch >= self.fglr_start_epoch

    def _set_fglr_enabled(self) -> None:
        module = self._network_module
        if hasattr(module, "set_fglr_enabled"):
            module.set_fglr_enabled(self._fglr_should_be_active())
        if self.use_ddp:
            dist.barrier()

    def _configured_cam_loss(self) -> bool:
        return self.grad_cam_alpha > 0 or self.layer_cam_gamma > 0

    def _cam_loss_enabled(self) -> bool:
        return self._configured_cam_loss() and self._cam_loss_active

    def _register_cam_hook(self, configured_target: str) -> None:
        module = self._network_module
        target = configured_target or str(getattr(module, "cam_target", "") or "")
        if not target:
            raise RuntimeError("CAM loss is enabled but no cam target is configured.")
        target_layer = resolve_layer(module, target)
        self._cam_hook_handle = target_layer.register_forward_hook(self._save_cam_activation)
        if self.is_main:
            print(f"[CAM] target_layer={target}")

    def _save_cam_activation(self, _module, _inputs, output) -> None:
        self._cam_activation = output if isinstance(output, torch.Tensor) else output[0]

    def _compute_cam_loss(self, outputs, masks: torch.Tensor) -> torch.Tensor:
        if not self._cam_loss_enabled():
            return masks.new_tensor(0.0, dtype=torch.float32)
        total = masks.new_tensor(0.0, dtype=torch.float32)
        grad_loss = None
        layer_loss = None
        if self.grad_cam_alpha > 0:
            grad_loss = self._cam_loss.grad_cam_loss(self._cam_activation, outputs, masks)
            total = total + self.grad_cam_alpha * grad_loss
            self._epoch_grad_cam_loss_sum += float(grad_loss.detach().item())
        if self.layer_cam_gamma > 0:
            layer_loss = self._cam_loss.layer_cam_loss(self._cam_activation, outputs, masks)
            total = total + self.layer_cam_gamma * layer_loss
            self._epoch_layer_cam_loss_sum += float(layer_loss.detach().item())
        self._epoch_cam_loss_count += 1
        return total

    def _compute_loss(self, outputs, masks: torch.Tensor) -> torch.Tensor:
        module = self._network_module
        aux = module.get_aux_outputs()
        if not bool(aux.get("fglr_active", True)):
            base_loss = self.loss_fn(outputs, masks)
            self._epoch_base_loss_sum += float(base_loss.detach().item())
            self._epoch_base_loss_count += 1
            return base_loss + self._compute_cam_loss(outputs, masks)

        base_outputs = aux.get("base_outputs", aux.get("base_logits"))
        if base_outputs is None:
            raise RuntimeError("FGLRSwinUMamba did not expose base outputs.")
        base_loss = self.loss_fn(base_outputs, masks)
        fglr_loss = self.loss_fn(outputs, masks)
        cam_loss = self._compute_cam_loss(base_outputs, masks)
        self._epoch_base_loss_sum += float(base_loss.detach().item())
        self._epoch_fglr_loss_sum += float(fglr_loss.detach().item())
        self._epoch_base_loss_count += 1
        self._epoch_fglr_loss_count += 1
        return self.base_loss_alpha * base_loss + self.fglr_loss_alpha * fglr_loss + cam_loss

    def train_one_epoch(self) -> float:
        self.network.train()
        self._set_fglr_enabled()
        if self.use_ddp and isinstance(self.train_loader.sampler, DistributedSampler):
            self.train_loader.sampler.set_epoch(self.current_epoch)
        self._reset_epoch_loss_trackers()
        total = 0.0
        steps = 0
        self.optimizer.zero_grad(set_to_none=True)
        grad_clip = float(self.cfg["training"].get("grad_clip", 0.0))

        for batch in self.train_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)
            outputs = self.network(images)
            loss = self._compute_loss(outputs, masks)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach().item())
            steps += 1
        return self._reduce_mean(total / max(steps, 1))

    @torch.no_grad()
    def validate_one_epoch(self, compute_hd: bool = False) -> Dict[str, Optional[float]]:
        self.network.eval()
        self._set_fglr_enabled()
        total_loss = 0.0
        n_batches = 0
        inter_accum = torch.zeros(self.num_classes, device=self.device)
        union_accum = torch.zeros(self.num_classes, device=self.device)
        local_preds: List[np.ndarray] = []
        local_targets: List[np.ndarray] = []

        for batch in self.val_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)
            outputs = self.network(images)
            loss = self.loss_fn(outputs, masks)
            total_loss += float(loss.item())
            n_batches += 1
            logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            preds = torch.argmax(logits, dim=1)
            for c in range(1, self.num_classes):
                pred_c = (preds == c).float()
                tgt_c = (masks == c).float()
                inter_accum[c] += (pred_c * tgt_c).sum()
                union_accum[c] += pred_c.sum() + tgt_c.sum()
            for pred, target in zip(preds.cpu().numpy(), masks.cpu().numpy()):
                local_preds.append(pred.astype(np.int32))
                local_targets.append(target.astype(np.int32))

        if self.use_ddp:
            scalars = torch.tensor([total_loss, float(n_batches)], device=self.device)
            dist.all_reduce(scalars, op=dist.ReduceOp.SUM)
            total_loss = scalars[0].item()
            n_batches = int(scalars[1].item())
            dist.all_reduce(inter_accum, op=dist.ReduceOp.SUM)
            dist.all_reduce(union_accum, op=dist.ReduceOp.SUM)
            gathered_preds = [None] * self.world_size
            gathered_targets = [None] * self.world_size
            dist.all_gather_object(gathered_preds, local_preds)
            dist.all_gather_object(gathered_targets, local_targets)
            if self.is_main:
                local_preds = [p for rank_preds in gathered_preds for p in rank_preds]
                local_targets = [t for rank_targets in gathered_targets for t in rank_targets]

        smooth = 1e-5
        pseudo = []
        for c in range(1, self.num_classes):
            pseudo.append((2.0 * inter_accum[c].item() + smooth) / (union_accum[c].item() + smooth))
        result = {
            "val_loss": total_loss / max(n_batches, 1),
            "pseudo_dice": float(np.mean(pseudo)) if pseudo else 0.0,
            "dice": None,
            "iou": None,
            "hd95": None,
        }
        if self.is_main and local_preds:
            exact = aggregate_metrics(local_preds, local_targets, self.num_classes, compute_hd=compute_hd)
            result.update(exact)
        return self._broadcast_from_main(result)

    def run(self) -> None:
        best_ckpt = self.output_dir / "checkpoint_best.pth"
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            t0 = time.perf_counter()
            train_loss = self.train_one_epoch()
            compute_hd = "hd95" in self.cfg.get("metrics", {}).get("val", []) and (
                epoch % int(self.cfg.get("metrics", {}).get("val_hd95_interval", 10)) == 0
            )
            val_metrics = self.validate_one_epoch(compute_hd=compute_hd)
            self._update_cam_gate(val_metrics)
            pseudo = float(val_metrics["pseudo_dice"])
            ema_pseudo = self.ema.update(pseudo)
            dice = float(val_metrics["dice"])
            is_best = (dice - self.best_dice) > self.min_delta
            if is_best:
                self.best_dice = dice
                self.best_epoch = epoch
                self.save_checkpoint(best_ckpt)
            self.scheduler.step()

            if self.is_main:
                assert self.logger is not None
                base_loss, fglr_loss = self._pop_fglr_losses()
                grad_cam_loss, layer_cam_loss = self._pop_cam_losses()
                self.logger.log_epoch(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_metrics["val_loss"],
                    pseudo_dice=pseudo,
                    ema_pseudo_dice=ema_pseudo,
                    dice=val_metrics["dice"],
                    iou=val_metrics["iou"],
                    hd95=val_metrics["hd95"],
                    lr=self.optimizer.param_groups[0]["lr"],
                    epoch_time=time.perf_counter() - t0,
                    is_best=is_best,
                    grad_cam_loss=grad_cam_loss,
                    layer_cam_loss=layer_cam_loss,
                    fglr_base_loss=base_loss,
                    fglr_refined_loss=fglr_loss,
                )
                self.logger.print_epoch_summary(epoch)

        if best_ckpt.exists():
            ckpt = torch.load(str(best_ckpt), map_location=self.device)
            self._network_module.load_state_dict(self._checkpoint_model_state(ckpt))
            self.current_epoch = max(self.best_epoch, 0)
            if self.is_main:
                print(f"[Done] best_epoch={self.best_epoch + 1} checkpoint={best_ckpt}")
        if self.use_ddp:
            dist.destroy_process_group()

    def _update_cam_gate(self, val_metrics: Dict[str, Optional[float]]) -> None:
        if self._cam_loss_active or self.cam_loss_start_threshold is None:
            return
        score = val_metrics.get(self.cam_loss_threshold_metric)
        if score is not None and float(score) >= self.cam_loss_start_threshold:
            self._cam_loss_active = True

    def save_checkpoint(self, path: Path) -> None:
        if self.is_main:
            torch.save({"model_state": self._network_module.state_dict()}, str(path))

    @staticmethod
    def _checkpoint_model_state(ckpt):
        if isinstance(ckpt, dict):
            return ckpt.get("model_state", ckpt.get("state_dict", ckpt))
        return ckpt

    def _reduce_mean(self, value: float) -> float:
        if not self.use_ddp:
            return value
        tensor = torch.tensor(float(value), device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.item() / self.world_size

    def _broadcast_from_main(self, value):
        if not self.use_ddp:
            return value
        payload = [value if self.is_main else None]
        dist.broadcast_object_list(payload, src=0)
        return payload[0]

    def _make_loader_generator(self, offset: int) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(int(self.seed) + (self.rank * 1000) + int(offset))
        return generator

    @staticmethod
    def _seed_data_worker(worker_id: int) -> None:
        del worker_id
        worker_seed = torch.initial_seed() % (2 ** 32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        info = get_worker_info()
        if info is None:
            return
        dataset = info.dataset
        while not hasattr(dataset, "set_worker_seed") and hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        if hasattr(dataset, "set_worker_seed"):
            dataset.set_worker_seed(worker_seed)

    def _reset_epoch_loss_trackers(self) -> None:
        self._epoch_base_loss_sum = 0.0
        self._epoch_fglr_loss_sum = 0.0
        self._epoch_base_loss_count = 0
        self._epoch_fglr_loss_count = 0
        self._epoch_grad_cam_loss_sum = 0.0
        self._epoch_layer_cam_loss_sum = 0.0
        self._epoch_cam_loss_count = 0

    def _pop_fglr_losses(self) -> Tuple[Optional[float], Optional[float]]:
        base = (
            self._epoch_base_loss_sum / self._epoch_base_loss_count
            if self._epoch_base_loss_count
            else None
        )
        fglr = (
            self._epoch_fglr_loss_sum / self._epoch_fglr_loss_count
            if self._epoch_fglr_loss_count
            else None
        )
        return base, fglr

    def _pop_cam_losses(self) -> Tuple[Optional[float], Optional[float]]:
        if self._epoch_cam_loss_count == 0:
            return None, None
        grad = self._epoch_grad_cam_loss_sum / self._epoch_cam_loss_count if self.grad_cam_alpha > 0 else None
        layer = self._epoch_layer_cam_loss_sum / self._epoch_cam_loss_count if self.layer_cam_gamma > 0 else None
        return grad, layer


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.fold is not None:
        cfg.setdefault("training", {})["fold"] = int(args.fold)
    if args.cfg_options:
        apply_overrides(cfg, args.cfg_options)
    for key in ("manifest_json", "splits_json"):
        if not cfg.get("data", {}).get(key):
            raise ValueError(f"data.{key} must be set in config or --cfg-options.")

    if _is_main_process():
        fold = cfg["training"]["fold"]
        print("=" * 60)
        print(f"Model : {cfg['model']['name']} / {cfg['model'].get('backbone', 'SwinUMamba')}")
        print(f"Fold  : {fold}")
        print(f"Output: {Path(cfg['output']['save_dir']) / fold_output_name(cfg, fold)}")
        print("=" * 60)

    trainer = MaskPolishTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
