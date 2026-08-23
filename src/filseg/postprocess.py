"""Turn a probability map into individual filament instances.

The network predicts a single foreground channel; instances are separated by
connected components. Because Panoptic Quality punishes fragmentation, a
morphological closing runs first so that a filament broken into beads by noise
is recovered as one segment, and specks below ``min_area`` are dropped rather
than shipped as false positives.
"""

from __future__ import annotations

import cv2
import numpy as np

from filseg.config import PostprocessConfig


def _disk_kernel(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def binarize(probability: np.ndarray, cfg: PostprocessConfig,
             valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Threshold, clean and (optionally) restrict a probability map to the disk."""
    binary = (probability >= cfg.threshold).astype(np.uint8)
    if valid_mask is not None:
        binary *= valid_mask.astype(np.uint8)

    if cfg.closing_radius > 0:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _disk_kernel(cfg.closing_radius))
    if cfg.dilate_radius > 0:
        binary = cv2.dilate(binary, _disk_kernel(cfg.dilate_radius))
    return binary


def split_instances(binary: np.ndarray, cfg: PostprocessConfig) -> list[np.ndarray]:
    """Connected components of a binary mask, largest first.

    Returns at most ``cfg.max_instances`` masks, each ``uint8`` and the same
    shape as the input.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    areas = [(int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, count)]
    areas = [(a, i) for a, i in areas if a >= cfg.min_area]
    areas.sort(reverse=True)

    return [(labels == i).astype(np.uint8) for _, i in areas[: cfg.max_instances]]


def instances_from_probability(probability: np.ndarray, cfg: PostprocessConfig,
                               valid_mask: np.ndarray | None = None) -> list[np.ndarray]:
    """Full probability map -> list of per-filament binary masks."""
    return split_instances(binarize(probability, cfg, valid_mask), cfg)
