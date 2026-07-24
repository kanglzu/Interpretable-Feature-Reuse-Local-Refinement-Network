"""Albumentations transforms for segmentation."""

from typing import Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2


def _compose(transforms, seed: Optional[int] = None) -> A.Compose:
    try:
        keypoint_params = A.KeypointParams(coord_format="xy", remove_invisible=False)
    except TypeError:
        keypoint_params = A.KeypointParams(format="xy", remove_invisible=False)
    if seed is None:
        return A.Compose(transforms, keypoint_params=keypoint_params)
    try:
        return A.Compose(transforms, seed=int(seed), keypoint_params=keypoint_params)
    except TypeError:
        compose = A.Compose(transforms, keypoint_params=keypoint_params)
        if hasattr(compose, "set_random_seed"):
            compose.set_random_seed(int(seed))
            return compose
        raise RuntimeError(
            "This albumentations version does not support seeded Compose construction. "
            "Use the strict reproducibility lockfile environment."
        )


def get_train_transforms(
    image_size: list,
    num_channels: int = 3,
    seed: Optional[int] = None,
) -> A.Compose:
    h, w = image_size

    spatial = [
        A.Resize(h, w, interpolation=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=30,
            border_mode=0,
            p=0.5,
        ),
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.2),
    ]

    pixel = [
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(std_range=(0.04, 0.22), p=0.3),
        A.CLAHE(clip_limit=4.0, p=0.2),
        A.HueSaturationValue(
            hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3
        ),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(1, h // 16),
            hole_width_range=(1, w // 16),
            fill=0,
            p=0.2,
        ),
    ]

    normalize = _get_normalize(num_channels)

    return _compose(spatial + pixel + [normalize, ToTensorV2()], seed=seed)


def get_val_transforms(
    image_size: list,
    num_channels: int = 3,
    seed: Optional[int] = None,
) -> A.Compose:
    h, w = image_size
    normalize = _get_normalize(num_channels)
    return _compose([A.Resize(h, w, interpolation=1), normalize, ToTensorV2()], seed=seed)


def _get_normalize(num_channels: int) -> A.Normalize:
    if num_channels == 1:
        mean = (0.5,)
        std = (0.5,)
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    return A.Normalize(mean=mean, std=std, max_pixel_value=255.0)
