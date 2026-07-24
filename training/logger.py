"""Training log writer."""

from pathlib import Path
from typing import Dict, List, Optional


class TrainingLogger:

    _METRIC_KEYS = [
        "train_loss",
        "fglr_base_loss",
        "fglr_refined_loss",
        "grad_cam_loss",
        "layer_cam_loss",
        "val_loss",
        "pseudo_dice",
        "ema_pseudo_dice",
        "dice",
        "iou",
        "hd95",
        "lr",
        "epoch_time",
    ]

    def __init__(self, output_dir: str, num_epochs: int):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_epochs = num_epochs

        self.log_file = self.output_dir / "training_log.txt"

        self._data: Dict[str, List] = {k: [] for k in self._METRIC_KEYS}
        self._best_epoch: Optional[int] = None
        self._best_metric_value: Optional[float] = None

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float],
        pseudo_dice: Optional[float],
        ema_pseudo_dice: Optional[float],
        dice: Optional[float],
        iou: Optional[float],
        lr: float,
        epoch_time: float,
        hd95: Optional[float] = None,
        is_best: bool = False,
        grad_cam_loss: Optional[float] = None,
        layer_cam_loss: Optional[float] = None,
        fglr_base_loss: Optional[float] = None,
        fglr_refined_loss: Optional[float] = None,
    ) -> None:
        self._data["train_loss"].append(train_loss)
        self._data["fglr_base_loss"].append(fglr_base_loss)
        self._data["fglr_refined_loss"].append(fglr_refined_loss)
        self._data["grad_cam_loss"].append(grad_cam_loss)
        self._data["layer_cam_loss"].append(layer_cam_loss)
        self._data["val_loss"].append(val_loss)
        self._data["pseudo_dice"].append(pseudo_dice)
        self._data["ema_pseudo_dice"].append(ema_pseudo_dice)
        self._data["dice"].append(dice)
        self._data["iou"].append(iou)
        self._data["hd95"].append(hd95)
        self._data["lr"].append(lr)
        self._data["epoch_time"].append(epoch_time)

        if is_best:
            self._best_epoch = epoch
            self._best_metric_value = dice

        self._write_log_line(epoch, is_best)

    def print_epoch_summary(self, epoch: int) -> None:
        i = -1
        parts = [
            f"[Epoch {epoch+1:04d}/{self.num_epochs:04d}]",
            f"  train_loss={self._data['train_loss'][i]:.4f}",
        ]
        if self._has_cam_loss_data():
            if self._data['grad_cam_loss'][i] is not None:
                parts.append(f"  grad_cam_loss={self._data['grad_cam_loss'][i]:.4f}")
            if self._data['layer_cam_loss'][i] is not None:
                parts.append(f"  layer_cam_loss={self._data['layer_cam_loss'][i]:.4f}")
        if self._has_fglr_loss_data():
            if self._data["fglr_base_loss"][i] is not None:
                parts.append(f"  base_loss={self._data['fglr_base_loss'][i]:.4f}")
            if self._data["fglr_refined_loss"][i] is not None:
                parts.append(f"  fglr_loss={self._data['fglr_refined_loss'][i]:.4f}")
        def _fmt(v: Optional[float], spec: str) -> str:
            return f"{v:{spec}}" if v is not None else "--"

        parts.extend([
            f"  val_loss={_fmt(self._data['val_loss'][i], '.4f')}",
            f"  pseudo_dice={_fmt(self._data['pseudo_dice'][i], '.4f')}",
            f"  ema_pseudo_dice={_fmt(self._data['ema_pseudo_dice'][i], '.4f')}",
        ])
        if self._data["dice"][i] is not None:
            parts.append(f"  dice={self._data['dice'][i]:.4f}")
        if self._data["iou"][i] is not None:
            parts.append(f"  iou={self._data['iou'][i]:.4f}")
        if self._data["hd95"][i] is not None:
            parts.append(f"  hd95={self._data['hd95'][i]:.2f}")
        parts.append(f"  lr={self._data['lr'][i]:.3e}")
        parts.append(f"  epoch={self._data['epoch_time'][i]:.1f}s")
        if epoch == self._best_epoch:
            parts.append("  [BEST]")
        print("".join(parts))

    def _has_cam_loss_data(self) -> bool:
        return (
            any(v is not None for v in self._data["grad_cam_loss"])
            or any(v is not None for v in self._data["layer_cam_loss"])
        )

    def _has_fglr_loss_data(self) -> bool:
        return (
            any(v is not None for v in self._data["fglr_base_loss"])
            or any(v is not None for v in self._data["fglr_refined_loss"])
        )

    def _write_log_line(self, epoch: int, is_best: bool) -> None:
        i = -1
        parts = [
            f"[Epoch {epoch+1:04d}/{self.num_epochs:04d}]",
            f"  train_loss={self._data['train_loss'][i]:.4f}",
        ]
        if self._has_cam_loss_data():
            if self._data['grad_cam_loss'][i] is not None:
                parts.append(f"  grad_cam_loss={self._data['grad_cam_loss'][i]:.4f}")
            if self._data['layer_cam_loss'][i] is not None:
                parts.append(f"  layer_cam_loss={self._data['layer_cam_loss'][i]:.4f}")
        if self._has_fglr_loss_data():
            if self._data["fglr_base_loss"][i] is not None:
                parts.append(f"  base_loss={self._data['fglr_base_loss'][i]:.4f}")
            if self._data["fglr_refined_loss"][i] is not None:
                parts.append(f"  fglr_loss={self._data['fglr_refined_loss'][i]:.4f}")
        parts.extend([
            f"  val_loss={self._data['val_loss'][i]:.4f}",
            f"  pseudo_dice={self._data['pseudo_dice'][i]:.4f}",
            f"  ema_pseudo_dice={self._data['ema_pseudo_dice'][i]:.4f}",
        ])
        if self._data["dice"][i] is not None:
            parts.append(f"  dice={self._data['dice'][i]:.4f}")
        if self._data["iou"][i] is not None:
            parts.append(f"  iou={self._data['iou'][i]:.4f}")
        if self._data["hd95"][i] is not None:
            parts.append(f"  hd95={self._data['hd95'][i]:.2f}")
        parts.append(f"  lr={self._data['lr'][i]:.3e}")
        parts.append(f"  epoch={self._data['epoch_time'][i]:.1f}s")
        if is_best:
            parts.append("  [BEST]")
        with open(self.log_file, "a") as f:
            f.write("".join(parts) + "\n")
