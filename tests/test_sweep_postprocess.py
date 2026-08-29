"""The postprocess sweep must genuinely find better settings, not just run."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from sweep_postprocess import sweep_grid  # noqa: E402

from filseg.config import PostprocessConfig  # noqa: E402


def test_sweep_finds_the_threshold_that_matches_ground_truth():
    """Two filaments at known probability levels, one real speck of noise.
    Only a threshold strictly between the noise and the real signal should
    recover both filaments cleanly with no false positive.
    """
    shape = (64, 64)
    probability = np.zeros(shape, dtype=np.float32)
    probability[10:14, 10:30] = 0.9   # a confident real filament
    probability[40:44, 10:30] = 0.6   # a less confident real filament
    probability[50:53, 50:56] = 0.4   # a noise blob, big enough that min_area can't filter it

    target_a = np.zeros(shape, dtype=np.uint8)
    target_a[10:14, 10:30] = 1
    target_b = np.zeros(shape, dtype=np.uint8)
    target_b[40:44, 10:30] = 1
    targets = [target_a, target_b]

    valid = np.ones(shape, dtype=np.uint8)
    cache = [(probability, None, None, valid, targets)]

    grid = [(t, 5, 1) for t in (0.2, 0.5, 0.8)]
    results = sweep_grid(cache, grid, PostprocessConfig())

    # threshold=0.2 lets the noise speck through as a false positive;
    # threshold=0.8 loses the second, less-confident filament (false negative).
    # threshold=0.5 is the only one that recovers exactly the two real filaments.
    best = results[0]
    assert best["threshold"] == 0.5
    assert best["tp"] == 2
    assert best["fp"] == 0
    assert best["fn"] == 0
    assert best["pq"] == 1.0

    scores = {r["threshold"]: r["pq"] for r in results}
    assert scores[0.5] > scores[0.2]
    assert scores[0.5] > scores[0.8]


def test_sweep_result_is_sorted_descending_by_pq():
    shape = (32, 32)
    probability = np.zeros(shape, dtype=np.float32)
    probability[5:9, 5:20] = 0.7
    target = np.zeros(shape, dtype=np.uint8)
    target[5:9, 5:20] = 1
    cache = [(probability, None, None, None, [target])]

    grid = [(t, 5, 1) for t in (0.1, 0.5, 0.9)]
    results = sweep_grid(cache, grid, PostprocessConfig())

    pqs = [r["pq"] for r in results]
    assert pqs == sorted(pqs, reverse=True)


def test_affinity_branch_is_used_when_cache_has_affinity_channels():
    """A cache entry with non-None affinity channels must go through
    instances_from_affinity, not the plain semantic path."""
    shape = (32, 32)
    probability = np.zeros(shape, dtype=np.float32)
    probability[5:9, 5:25] = 0.8
    right = np.ones(shape, dtype=np.float32)   # perfect affinity: never cuts
    down = np.ones(shape, dtype=np.float32)
    target = np.zeros(shape, dtype=np.uint8)
    target[5:9, 5:25] = 1
    cache = [(probability, right, down, None, [target])]

    results = sweep_grid(cache, [(0.5, 5, 1)], PostprocessConfig())
    assert results[0]["tp"] == 1
