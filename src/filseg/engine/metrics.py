"""Evaluation metrics, mirroring the competition's scoring.

Panoptic Quality is the leaderboard metric: predictions and ground-truth
segments are matched one-to-one at IoU > 0.5, and

    PQ = sum(IoU over TP) / (|TP| + 0.5|FP| + 0.5|FN|)

so both fragmentation (extra predictions) and over-merging (missed segments)
are penalised directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Per-image Dice for binary masks in ``(N, 1, H, W)`` form."""
    dims = (1, 2, 3)
    intersection = (pred * target).sum(dims)
    return (2 * intersection + eps) / (pred.sum(dims) + target.sum(dims) + eps)


def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    dims = (1, 2, 3)
    intersection = (pred * target).sum(dims)
    union = pred.sum(dims) + target.sum(dims) - intersection
    return (intersection + eps) / (union + eps)


def _pairwise_iou(preds: list[np.ndarray], targets: list[np.ndarray]) -> np.ndarray:
    """IoU between every predicted and every ground-truth instance."""
    if not preds or not targets:
        return np.zeros((len(preds), len(targets)), dtype=np.float32)

    pred_flat = np.stack([p.reshape(-1).astype(bool) for p in preds])
    target_flat = np.stack([t.reshape(-1).astype(bool) for t in targets])

    intersection = (pred_flat.astype(np.uint32) @ target_flat.astype(np.uint32).T)
    intersection = intersection.astype(np.float32)
    pred_area = pred_flat.sum(1)[:, None]
    target_area = target_flat.sum(1)[None, :]
    union = pred_area + target_area - intersection
    return np.divide(intersection, np.maximum(union, 1), dtype=np.float32)


@dataclass
class PanopticQuality:
    """Accumulates PQ (and its components) over a set of images."""

    iou_threshold: float = 0.5
    tp: int = 0
    fp: int = 0
    fn: int = 0
    iou_sum: float = 0.0
    matched_ious: list[float] = field(default_factory=list)
    one_to_many: int = 0  # gt segments overlapped by >1 prediction
    many_to_one: int = 0  # predictions overlapping >1 gt segment

    def update(self, preds: list[np.ndarray], targets: list[np.ndarray]) -> float:
        """Score one image; returns that image's PQ."""
        iou = _pairwise_iou(preds, targets)

        matches: list[tuple[int, int]] = []
        if iou.size:
            # IoU > 0.5 admits at most one match per segment, so a greedy pass
            # over descending IoU is exact.
            order = np.dstack(np.unravel_index(np.argsort(iou, axis=None)[::-1], iou.shape))[0]
            used_pred: set[int] = set()
            used_target: set[int] = set()
            for p, t in order:
                if iou[p, t] <= self.iou_threshold:
                    break
                if p in used_pred or t in used_target:
                    continue
                used_pred.add(int(p))
                used_target.add(int(t))
                matches.append((int(p), int(t)))

            overlapping = iou > 0
            self.one_to_many += int((overlapping.sum(axis=0) > 1).sum())
            self.many_to_one += int((overlapping.sum(axis=1) > 1).sum())

        image_iou = sum(float(iou[p, t]) for p, t in matches)
        tp = len(matches)
        fp = len(preds) - tp
        fn = len(targets) - tp

        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.iou_sum += image_iou
        self.matched_ious.extend(float(iou[p, t]) for p, t in matches)

        denominator = tp + 0.5 * fp + 0.5 * fn
        return image_iou / denominator if denominator else 1.0

    @property
    def score(self) -> float:
        denominator = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.iou_sum / denominator if denominator else 0.0

    @property
    def segmentation_quality(self) -> float:
        return self.iou_sum / self.tp if self.tp else 0.0

    @property
    def recognition_quality(self) -> float:
        denominator = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.tp / denominator if denominator else 0.0

    def summary(self) -> dict[str, float]:
        return {
            "pq": self.score,
            "sq": self.segmentation_quality,
            "rq": self.recognition_quality,
            "tp": float(self.tp),
            "fp": float(self.fp),
            "fn": float(self.fn),
            "one_to_many": float(self.one_to_many),
            "many_to_one": float(self.many_to_one),
        }
