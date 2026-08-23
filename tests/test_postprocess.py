"""Instance splitting drops specks and bridges small gaps."""

import numpy as np

from filseg.config import PostprocessConfig
from filseg.postprocess import instances_from_probability


def test_speck_is_dropped_and_gap_is_closed():
    probability = np.zeros((64, 64), dtype=np.float32)
    probability[10:40, 10:14] = 0.9      # one filament, split by a 2 px gap
    probability[22:24, 10:14] = 0.0
    probability[60, 60] = 0.9            # a single-pixel speck

    cfg = PostprocessConfig(threshold=0.5, min_area=20, closing_radius=2)
    instances = instances_from_probability(probability, cfg)

    assert len(instances) == 1
    assert instances[0].sum() >= 100
