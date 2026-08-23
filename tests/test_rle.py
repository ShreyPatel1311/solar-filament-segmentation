"""RLE round-trips at the submission's fixed 2048 x 2048 size."""

import numpy as np

from filseg.data.coco import decode_rle, encode_rle


def test_round_trip():
    mask = np.zeros((2048, 2048), dtype=np.uint8)
    mask[100:180, 400:404] = 1
    mask[900:905, 900:1200] = 1

    restored = decode_rle(encode_rle(mask))
    assert np.array_equal(mask, restored)
