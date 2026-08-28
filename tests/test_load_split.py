"""load_split must never leak an observing day across train/val.

Regression for a real incident: a naive per-image shuffle put 62% of a
173-image val split on the same calendar day as a training image, inflating
local PQ (~0.41) far above what the same checkpoint scored on Kaggle's
genuinely unseen test set (~0.29) -- filaments barely evolve within a day, so
those "held-out" images were near-duplicates of something the model trained on.
"""

import pytest

from filseg.data.coco import ImageRecord
from filseg.data.dataset import load_split


def _records(specs: list[tuple[str, int]]) -> list[ImageRecord]:
    """specs: [(YYYYMMDD, images_that_day), ...] -> ImageRecord list."""
    out = []
    for date, count in specs:
        for i in range(count):
            out.append(ImageRecord(image_id=f"{date}{i}", file_name=f"{date}{i:06d}Mh.jpeg",
                                    height=64, width=64))
    return out


def test_no_date_appears_on_both_sides():
    # Mirrors the shape of the real incident: many days, uneven image counts
    # per day (some days have several frames a few hours apart).
    specs = [(f"202201{d:02d}", (d % 4) + 1) for d in range(1, 29)]
    records = _records(specs)

    train, val = load_split(records, val_fraction=0.15, seed=1311)

    train_dates = {r.stem[:8] for r in train}
    val_dates = {r.stem[:8] for r in val}
    assert train_dates.isdisjoint(val_dates)
    assert len(train) + len(val) == len(records)


def test_split_is_deterministic_across_calls():
    records = _records([(f"202203{d:02d}", 3) for d in range(1, 21)])
    first = load_split(records, val_fraction=0.2, seed=42)
    second = load_split(records, val_fraction=0.2, seed=42)
    assert [r.stem for r in first[0]] == [r.stem for r in second[0]]
    assert [r.stem for r in first[1]] == [r.stem for r in second[1]]


def test_val_fraction_is_approximately_respected():
    # Grouping by day means the exact fraction can't be hit precisely, but it
    # shouldn't be wildly off for a reasonably large, evenly-spread dataset.
    records = _records([(f"2022{m:02d}{d:02d}", 2)
                        for m in range(1, 7) for d in range(1, 26)])
    train, val = load_split(records, val_fraction=0.15, seed=7)
    fraction = len(val) / len(records)
    assert 0.08 < fraction < 0.25


def test_a_single_observing_day_cannot_be_split():
    records = _records([("20220101", 10)])
    with pytest.raises(ValueError, match="observing day"):
        load_split(records, val_fraction=0.15, seed=1311)
