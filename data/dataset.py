"""Manifest-backed 2D segmentation dataset."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SegDataset(Dataset):

    def __init__(
        self,
        sample_ids: List[str],
        sample_dict: Dict[str, dict],
        transform: Optional[Callable] = None,
        transform_factory: Optional[Callable[[int], Callable]] = None,
        num_channels: int = 3,
        cache_in_memory: bool = False,
        transform_seed: Optional[int] = None,
    ) -> None:
        if transform is not None and transform_factory is not None:
            raise ValueError("Specify either transform or transform_factory, not both.")
        self.sample_ids = [str(sid) for sid in sample_ids]
        self.sample_dict = sample_dict
        self.transform = transform
        self.transform_factory = transform_factory
        self._transform_seed = transform_seed
        self._transform_instance = transform
        self.num_channels = int(num_channels)
        self.cache_in_memory = bool(cache_in_memory)
        self._cache: Dict[str, tuple[np.ndarray, np.ndarray]] = {}

        if self.cache_in_memory:
            for sid in self.sample_ids:
                self._cache[sid] = self._load(sid)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict:
        sid = self.sample_ids[idx]
        if self.cache_in_memory:
            image, mask = self._cache[sid]
            image, mask = image.copy(), mask.copy()
        else:
            image, mask = self._load(sid)

        transform = self._get_transform()
        if transform is not None:
            result = transform(image=image, mask=mask)
            image, mask = result["image"], result["mask"]

        return {
            "id": sid,
            "image": image.float() if isinstance(image, torch.Tensor) else torch.from_numpy(image),
            "mask": mask.long() if isinstance(mask, torch.Tensor) else torch.from_numpy(mask).long(),
        }

    def _load(self, sid: str) -> tuple[np.ndarray, np.ndarray]:
        entry = self.sample_dict[str(sid)]
        return _read_image(entry["image"], self.num_channels), _read_mask(entry["label"])

    def set_worker_seed(self, seed: int) -> None:
        self._transform_seed = int(seed)
        if self.transform_factory is not None:
            self._transform_instance = None

    def _get_transform(self) -> Optional[Callable]:
        if self.transform_factory is None:
            return self.transform
        if self._transform_instance is None:
            if self._transform_seed is None:
                raise RuntimeError("transform_factory requires a worker seed before first use.")
            self._transform_instance = self.transform_factory(int(self._transform_seed))
        return self._transform_instance


def _read_image(path: str, num_channels: int) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if int(num_channels) == 1 else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if int(num_channels) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif int(num_channels) == 1:
        image = image[:, :, None]
    else:
        raise ValueError(f"Only 1 or 3 input channels are supported, got {num_channels}.")
    return image


def _read_mask(path: str) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    return (mask > 0).astype(np.uint8)
