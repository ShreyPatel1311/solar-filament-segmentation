"""Augmentation and preprocessing pipelines.

H-alpha frames are grayscale, full-disk and rotationally arbitrary, so flips
and 90-degree rotations are label-preserving. Brightness/contrast jitter stands
in for the seeing and exposure differences between GONG sites.
"""

from __future__ import annotations

import albumentations as A
import numpy as np
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


def diercke_normalize(image: np.ndarray, disk: np.ndarray | None = None) -> np.ndarray:
    """Intensity normalization from Diercke et al. (2024), Section 2.

    "All filtergrams are scaled to a solar radius of r = 1000 pixels ... the
    images are normalized to the median intensity of the solar disk, limb
    darkening corrected, and the off-limb region is truncated ... Finally, we
    clip values to [0.8, 1.3], followed by normalizing the data to the
    interval [-1, 1]."

    Dividing by the *median disk intensity* rather than a fixed constant is
    what makes this robust across GONG's six sites and across cloud cover:
    each frame is placed on a common intensity scale where 1.0 is quiet Sun,
    so the [0.8, 1.3] clip means the same physical thing everywhere. Filaments
    are absorption features, so they sit below 1.0; the asymmetric window
    keeps more range above quiet Sun than below, preserving plage and flare
    brightenings that would otherwise saturate.

    The paper corrects limb darkening with Zernike polynomials, which needs
    the fitted disk geometry; that step is not reproduced here. The
    ``disk`` mask restricts the median to on-disk pixels, which is the part
    that matters most for the scaling to be comparable between frames.

    Args:
        image: grayscale frame.
        disk: optional binary on-disk mask; the median is taken over it when
            given, and off-disk pixels are zeroed after normalization.

    Returns:
        float32 array in [-1, 1].
    """
    data = image.astype(np.float32)

    on_disk = data[disk > 0] if disk is not None else data
    median = float(np.median(on_disk))
    if median <= 0:  # an all-dark frame would otherwise divide by zero
        median = float(data.mean()) or 1.0

    normalized = np.clip(data / median, 0.8, 1.3)
    # Map [0.8, 1.3] onto [-1, 1].
    normalized = (normalized - 0.8) / (1.3 - 0.8) * 2.0 - 1.0

    if disk is not None:
        normalized = normalized * (disk > 0)
    return normalized.astype(np.float32)
