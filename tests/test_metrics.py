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
