"""Panoptic Quality behaves as the competition defines it."""

import numpy as np

from filseg.engine.metrics import PanopticQuality


def _blob(shape, slices):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[slices] = 1
    return mask


def test_perfect_match_scores_one():
    gt = [_blob((32, 32), (slice(2, 10), slice(2, 10)))]
    pq = PanopticQuality()
    pq.update(list(gt), gt)
    assert pq.score == 1.0


def test_fragmentation_is_penalised():
    gt = [_blob((32, 32), (slice(0, 20), slice(0, 4)))]
    halves = [
        _blob((32, 32), (slice(0, 9), slice(0, 4))),
        _blob((32, 32), (slice(11, 20), slice(0, 4))),
    ]
    pq = PanopticQuality()
    pq.update(halves, gt)
    assert pq.score < 0.5
    assert pq.many_to_one == 0 and pq.one_to_many == 1


def test_false_positive_lowers_score():
    gt = [_blob((32, 32), (slice(2, 10), slice(2, 10)))]
    preds = gt + [_blob((32, 32), (slice(20, 28), slice(20, 28)))]
    pq = PanopticQuality()
    pq.update(preds, gt)
    assert pq.score == 1.0 / 1.5


def test_pooled_pq_and_mean_per_image_pq_are_genuinely_different_statistics():
    """One missed-everything image among many near-perfect ones barely moves
    the pooled PQ (dominated by images with many instances) but craters a
    mean of per-image PQ (every image counted equally). Regression for
    evaluate.py reporting only the pooled number, which can look much
    healthier locally than a per-image-averaged leaderboard scorer would.
    """

    def blob(shape, sl):
        mask = np.zeros(shape, dtype=np.uint8)
        mask[sl] = 1
        return mask

    pq = PanopticQuality()
    per_image = []

    for i in range(9):
        gt = [blob((64, 64), (slice(i * 4, i * 4 + 3), slice(0, 20))) for i in range(5)]
        per_image.append(pq.update(gt, gt))  # perfect prediction

    missed = [blob((64, 64), (slice(30, 34), slice(0, 20)))]
    per_image.append(pq.update([], missed))  # predicted nothing at all

    mean_per_image = sum(per_image) / len(per_image)
    assert per_image[-1] == 0.0
    assert pq.score > 0.95          # pooled: barely affected
    assert mean_per_image < 0.95    # per-image mean: heavily affected
    assert pq.score - mean_per_image > 0.05  # genuinely different numbers
