"""Affinity ground truth and affinity-aware instance splitting."""

import numpy as np

from filseg.config import PostprocessConfig
from filseg.data.affinity import affinity_targets, instance_id_map
from filseg.postprocess import instances_from_affinity, split_instances


def _touching_pair(shape=(20, 20)):
    """Two filaments sharing a seam -- exactly the case connected components
    over-merges and affinity is supposed to fix."""
    a = np.zeros(shape, np.uint8)
    a[4:16, 2:6] = 1
    b = np.zeros(shape, np.uint8)
    b[4:16, 6:9] = 1  # touches a's right edge at x=5/6
    return a, b


def test_ground_truth_marks_the_seam_and_nothing_else_as_cut():
    a, b = _touching_pair()
    affinity, valid = affinity_targets([a, b], a.shape)

    # every right-edge pair strictly inside one instance is "same"
    assert affinity[0, 8, 3] == 1  # inside a
    assert affinity[0, 8, 7] == 1  # inside b
    # the seam itself is "different"
    assert affinity[0, 8, 5] == 0
    assert valid[0, 8, 5] == 1


def test_plain_connected_components_merges_the_touching_pair():
    """Establishes the failure mode this feature exists to fix."""
    a, b = _touching_pair()
    binary = ((a + b) > 0).astype(np.uint8)
    cfg = PostprocessConfig(min_area=5, closing_radius=0)
    instances = split_instances(binary, cfg)
    assert len(instances) == 1  # merged into one blob -- the bug


def test_affinity_correctly_separates_the_touching_pair():
    a, b = _touching_pair()
    binary = (a + b) > 0
    probability = binary.astype(np.float32)  # a "perfect" semantic prediction

    height, width = a.shape
    ids = instance_id_map([a, b], a.shape)
    # A "perfect" affinity predictor: 1 where truly same instance, 0 where not.
    affinity_right = np.ones((height, width), np.float32)
    affinity_right[:, :-1] = (ids[:, :-1] == ids[:, 1:]).astype(np.float32)
    affinity_down = np.ones((height, width), np.float32)
    affinity_down[:-1, :] = (ids[:-1, :] == ids[1:, :]).astype(np.float32)

    cfg = PostprocessConfig(min_area=5, closing_radius=0)
    instances = instances_from_affinity(probability, affinity_right, affinity_down, cfg)

    assert len(instances) == 2
    areas = sorted(int(m.sum()) for m in instances)
    assert areas == sorted([int(a.sum()), int(b.sum())])


def test_low_affinity_threshold_falls_back_to_merging_everything():
    """A degenerate affinity predictor (always confident 'same') should behave
    like plain connected components -- sanity check that the mechanism is
    actually affinity-gated, not always splitting regardless of prediction."""
    a, b = _touching_pair()
    probability = ((a + b) > 0).astype(np.float32)
    height, width = a.shape
    always_same = np.ones((height, width), np.float32)

    cfg = PostprocessConfig(min_area=5, closing_radius=0)
    instances = instances_from_affinity(probability, always_same, always_same, cfg)
    assert len(instances) == 1


def test_empty_prediction_returns_no_instances():
    shape = (16, 16)
    zeros = np.zeros(shape, np.float32)
    cfg = PostprocessConfig(min_area=5, closing_radius=0)
    assert instances_from_affinity(zeros, zeros, zeros, cfg) == []


def test_instance_id_map_labels_disjoint_regions_distinctly():
    a, b = _touching_pair()
    ids = instance_id_map([a, b], a.shape)
    assert set(np.unique(ids)) == {0, 1, 2}
    assert (ids[a > 0] == 1).all()
    assert (ids[b > 0] == 2).all()
