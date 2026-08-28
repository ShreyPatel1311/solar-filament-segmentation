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


class AffinityDiceBCELoss(nn.Module):
    """DiceBCELoss on a semantic channel, plus masked BCE on two affinity
    channels: does each pixel share its instance with its right/bottom
    neighbor? (see :mod:`filseg.data.affinity`).

    ``logits`` is ``(N, 3, H, W)``: channel 0 is the semantic logit (same
    contract as :class:`DiceBCELoss`), channels 1:3 are the affinity logits
    ``[right, down]``. Background-background pairs dominate a typical frame
    and carry no signal the semantic loss doesn't already give, so the
    affinity term is masked to ``affinity_valid`` (pairs with at least one
    foreground side) rather than averaged over the whole image -- otherwise
    the rare, hard boundary pixels this exists for would be drowned out.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5,
                 affinity_weight: float = 1.0, smooth: float = 1.0):
        super().__init__()
        self.semantic = DiceBCELoss(bce_weight, dice_weight, smooth)
        self.affinity_weight = affinity_weight
        self.affinity_bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                affinity: torch.Tensor | None = None,
                affinity_valid: torch.Tensor | None = None) -> torch.Tensor:
        semantic_loss = self.semantic(logits[:, :1], target)
        if affinity is None or affinity_valid is None:
            return semantic_loss

        per_pixel = self.affinity_bce(logits[:, 1:3], affinity)
        denominator = affinity_valid.sum().clamp_min(1.0)
        affinity_loss = (per_pixel * affinity_valid).sum() / denominator

        return semantic_loss + self.affinity_weight * affinity_loss
