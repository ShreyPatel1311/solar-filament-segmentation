"""Augmentation and preprocessing pipelines.

H-alpha frames are grayscale, full-disk and rotationally arbitrary, so flips
and 90-degree rotations are label-preserving. Brightness/contrast jitter stands
in for the seeing and exposure differences between GONG sites.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def train_transforms(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.08, rotate_limit=20, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(2.0, 12.0), p=0.2),
            A.Normalize(mean=(0.5,), std=(0.25,), max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


def val_transforms(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=1),
            A.Normalize(mean=(0.5,), std=(0.25,), max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )
