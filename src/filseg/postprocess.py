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
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components as _sparse_connected_components

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


def instances_from_affinity(probability: np.ndarray, affinity_right: np.ndarray,
                            affinity_down: np.ndarray, cfg: PostprocessConfig,
                            valid_mask: np.ndarray | None = None,
                            affinity_threshold: float = 0.5) -> list[np.ndarray]:
    """Split a binary mask into instances using predicted affinity, not blind
    8-connectivity.

    :func:`split_instances` merges any two touching foreground pixels
    unconditionally, which is exactly what over-merges two filaments that
    cross or run alongside each other. Here, a foreground pixel is only
    grouped with its right/bottom neighbor when the model's predicted
    same-instance affinity (see :mod:`filseg.data.affinity`) actually says
    they belong together -- so a predicted "cut" between two touching
    filaments is respected instead of erased by connectivity.

    ``affinity_right``/``affinity_down`` are the model's affinity channels
    after ``sigmoid``, same shape as ``probability``.
    """
    binary = binarize(probability, cfg, valid_mask)
    height, width = binary.shape

    foreground = np.flatnonzero(binary)
    if foreground.size == 0:
        return []

    node_id = np.full(height * width, -1, dtype=np.int64)
    node_id[foreground] = np.arange(foreground.size)

    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    both_fg_right = (binary[:, :-1] > 0) & (binary[:, 1:] > 0)
    right_join = both_fg_right & (affinity_right[:, :-1] >= affinity_threshold)
    ys, xs = np.nonzero(right_join)
    sources.append(node_id[ys * width + xs])
    targets.append(node_id[ys * width + xs + 1])

    both_fg_down = (binary[:-1, :] > 0) & (binary[1:, :] > 0)
    down_join = both_fg_down & (affinity_down[:-1, :] >= affinity_threshold)
    ys, xs = np.nonzero(down_join)
    sources.append(node_id[ys * width + xs])
    targets.append(node_id[(ys + 1) * width + xs])

    src = np.concatenate(sources)
    dst = np.concatenate(targets)
    n = foreground.size
    graph = coo_matrix((np.ones(src.size, dtype=np.uint8), (src, dst)), shape=(n, n))
    _, labels = _sparse_connected_components(graph, directed=False)

    counts = np.bincount(labels, minlength=labels.max() + 1 if labels.size else 0)
    kept = [i for i in np.argsort(-counts) if counts[i] >= cfg.min_area][: cfg.max_instances]

    masks = []
    for component in kept:
        flat = np.zeros(height * width, dtype=np.uint8)
        flat[foreground[labels == component]] = 1
        masks.append(flat.reshape(height, width))
    return masks
