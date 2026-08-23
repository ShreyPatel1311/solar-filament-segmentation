"""MagfiloAnnotations reads a COCO file directly, without pycocotools' COCO index.

On the real MAGFiLO training file, going through pycocotools' COCO class
(``getImgIds`` -> ``loadImgs``) raised ``KeyError`` even though the id came
straight out of ``getImgIds()`` on the same object -- consistent with
``images[].id`` and ``annotations[].image_id`` not sharing one type
throughout the file. These tests pin the behaviour that avoids it: ids are
normalized to ``str`` once, directly from the parsed JSON, on both sides of
the join.
"""

import json

import numpy as np
from pycocotools import mask as mask_utils

from filseg.data.coco import MagfiloAnnotations


def _rle_ann(ann_id, image_id, mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("utf-8")
    return {
        "id": ann_id, "image_id": image_id, "category_id": 1, "iscrowd": 0,
        "segmentation": rle, "area": int(mask.sum()), "bbox": [0, 0, 1, 1],
    }


def _polygon_ann(ann_id, image_id) -> dict:
    return {
        "id": ann_id, "image_id": image_id, "category_id": 1, "iscrowd": 0,
        "segmentation": [[2, 2, 2, 6, 6, 6, 6, 2]], "area": 16, "bbox": [2, 2, 4, 4],
    }


def _write(tmp_path, payload) -> str:
    path = tmp_path / "ann.json"
    path.write_text(json.dumps(payload))
    return str(path)


def test_mixed_int_and_string_image_ids(tmp_path):
    """Some images use an int id, others a string id -- both must resolve."""
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:8, 4:8] = 1
    payload = {
        "images": [
            {"id": 0, "file_name": "a.jpeg", "height": 16, "width": 16},
            {"id": "1", "file_name": "b.jpeg", "height": 16, "width": 16},
        ],
        "annotations": [
            _rle_ann(1, 0, mask),
            _polygon_ann(2, "1"),
        ],
    }
    annotations = MagfiloAnnotations(_write(tmp_path, payload))
    records = annotations.records

    assert [r.file_name for r in records] == ["a.jpeg", "b.jpeg"]
    assert len(annotations.instance_masks(records[0])) == 1
    assert len(annotations.instance_masks(records[1])) == 1


def test_annotation_image_id_type_differs_from_image_id(tmp_path):
    """An annotation's image_id may be the string form of the image's int id."""
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    payload = {
        "images": [{"id": 7, "file_name": "c.jpeg", "height": 16, "width": 16}],
        "annotations": [_rle_ann(1, "7", mask)],
    }
    annotations = MagfiloAnnotations(_write(tmp_path, payload))
    record = annotations.records[0]

    assert len(annotations.instance_masks(record)) == 1


def test_iscrowd_annotations_are_excluded(tmp_path):
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[0:2, 0:2] = 1
    crowd = _rle_ann(1, 0, mask)
    crowd["iscrowd"] = 1
    payload = {
        "images": [{"id": 0, "file_name": "a.jpeg", "height": 16, "width": 16}],
        "annotations": [crowd],
    }
    annotations = MagfiloAnnotations(_write(tmp_path, payload))
    assert annotations.instance_masks(annotations.records[0]) == []


def test_semantic_mask_is_union_of_instances(tmp_path):
    first = np.zeros((16, 16), dtype=np.uint8)
    first[0:4, 0:4] = 1
    second = np.zeros((16, 16), dtype=np.uint8)
    second[10:14, 10:14] = 1
    payload = {
        "images": [{"id": 0, "file_name": "a.jpeg", "height": 16, "width": 16}],
        "annotations": [_rle_ann(1, 0, first), _rle_ann(2, 0, second)],
    }
    annotations = MagfiloAnnotations(_write(tmp_path, payload))
    union = annotations.semantic_mask(annotations.records[0])

    assert np.array_equal(union, np.maximum(first, second))
