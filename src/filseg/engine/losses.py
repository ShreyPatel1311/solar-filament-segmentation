"""Training objectives.

Filaments cover well under 1% of a frame, so a plain BCE optimum is "predict
background". Pairing it with a soft Dice term keeps the gradient informative
for the positive class.
"""

from __future__ import annotations

import torch
from torch import nn


class DiceBCELoss(nn.Module):
    """Weighted sum of BCE-with-logits and soft Dice."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5,
                 smooth: float = 1.0, pos_weight: float | None = None):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        weight = None if pos_weight is None else torch.tensor([pos_weight])
        self.bce = nn.BCEWithLogitsLoss(pos_weight=weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, target)

        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = (probs * target).sum(dims)
        denominator = probs.sum(dims) + target.sum(dims)
        dice = 1.0 - ((2 * intersection + self.smooth) / (denominator + self.smooth)).mean()

        return self.bce_weight * bce + self.dice_weight * dice
