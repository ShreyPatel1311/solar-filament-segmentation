"""Pixel-affinity ground truth: does a pixel share its instance with its
right/bottom neighbor?

The semantic mask (union of every filament) tells the network *where*
filament pixels are, but nothing about which filament a pixel belongs to --
that separation is currently reconstructed after the fact by connected
components (:mod:`filseg.postprocess`), which merges any two filaments that
physically touch or cross. Affinity gives the network a direct training
signal for that boundary: for each pixel, "am I the same instance as my
right neighbor?" and "...as my neighbor below?".

Only right/bottom are needed -- affinity is symmetric, so left/right and
up/down duplicate the same edges from the other pixel's perspective; this is
the standard 2-channel grid-affinity formulation from connectomics-style
instance segmentation.

Background counts as its own shared "instance" (id 0) for this purpose, so a
background-background pair is trivially affinity=1. That case dominates a
typical frame and carries no information the semantic mask doesn't already
give, so ``valid`` marks it out of the loss -- training only sees pairs where
at least one side is foreground, which is where merging/fragmentation
actually happens.
"""

from __future__ import annotations

import numpy as np


def instance_id_map(instance_masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Label each instance's pixels with a distinct id (1..N); background is 0.

    Where two ground-truth instances overlap (rare, but MAGFiLO does not
    guarantee disjoint masks), the later one in the list wins -- consistent
    with how :func:`filseg.data.coco.MagfiloAnnotations.semantic_mask` already
    unions them.
    """
    ids = np.zeros(shape, dtype=np.int32)
    for instance_id, mask in enumerate(instance_masks, start=1):
        ids[mask > 0] = instance_id
    return ids


def affinity_targets(instance_masks: list[np.ndarray], shape: tuple[int, int]
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth (affinity, valid), each ``(2, H, W)`` float32: [right, down].

    ``affinity[c, y, x] == 1`` iff pixel (y, x) and its neighbor in direction
    ``c`` belong to the same instance (including both-background).
    ``valid[c, y, x] == 1`` iff at least one of the pair is foreground -- the
    only pairs worth training on; the last row/column has no "down"/"right"
    neighbor and is invalid by construction.
    """
    ids = instance_id_map(instance_masks, shape)
    height, width = shape

    affinity = np.zeros((2, height, width), dtype=np.float32)
    valid = np.zeros((2, height, width), dtype=np.float32)

    # Right neighbor: pixel x paired with x+1.
    same = (ids[:, :-1] == ids[:, 1:])
    either_fg = (ids[:, :-1] > 0) | (ids[:, 1:] > 0)
    affinity[0, :, :-1] = same
    valid[0, :, :-1] = either_fg

    # Down neighbor: pixel y paired with y+1.
    same = (ids[:-1, :] == ids[1:, :])
    either_fg = (ids[:-1, :] > 0) | (ids[1:, :] > 0)
    affinity[1, :-1, :] = same
    valid[1, :-1, :] = either_fg

    return affinity, valid
