"""FilamentDataset(include_affinity=True): batches stay collate-consistent."""

import json

import numpy as np
import torch
from pycocotools import mask as mask_utils

from filseg.data.coco import MagfiloAnnotations
from filseg.data.dataset import FilamentDataset
from filseg.data.transforms import val_transforms


def _rle(mask):
    rle = mask_utils.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode()
    return rle


def _write_dataset(tmp_path):
    size = 32
    images = [
        {"id": 0, "file_name": "two.jpeg", "height": size, "width": size},
        {"id": 1, "file_name": "empty.jpeg", "height": size, "width": size},
    ]
    a = np.zeros((size, size), np.uint8)
    a[4:16, 2:6] = 1
    b = np.zeros((size, size), np.uint8)
    b[4:16, 6:9] = 1
    anns = [
        {"id": 1, "image_id": 0, "category_id": 1, "iscrowd": 0, "segmentation": _rle(a),
         "area": int(a.sum()), "bbox": [2, 4, 4, 12]},
        {"id": 2, "image_id": 0, "category_id": 1, "iscrowd": 0, "segmentation": _rle(b),
         "area": int(b.sum()), "bbox": [6, 4, 3, 12]},
        # image 1 (empty.jpeg) deliberately gets zero annotations
    ]
    (tmp_path / "ann.json").write_text(json.dumps({"images": images, "annotations": anns}))

    import cv2
    cv2.imwrite(str(tmp_path / "two.jpeg"), np.full((size, size), 100, np.uint8))
    cv2.imwrite(str(tmp_path / "empty.jpeg"), np.full((size, size), 100, np.uint8))
    return tmp_path / "ann.json"


def test_batches_with_and_without_instances_collate_together(tmp_path):
    annotation_file = _write_dataset(tmp_path)
    annotations = MagfiloAnnotations(annotation_file)
    dataset = FilamentDataset(
        annotations.records, images_dir=tmp_path, annotations=annotations,
        transform=val_transforms(16), apply_disk_mask=False, include_affinity=True,
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    batch = next(iter(loader))

    assert batch["affinity"].shape == (2, 2, 16, 16)
    assert batch["affinity_valid"].shape == (2, 2, 16, 16)
    # the empty-instance image contributes zero valid pairs, not a crash
    empty_index = batch["stem"].index("empty")
    assert batch["affinity_valid"][empty_index].sum() == 0


def test_without_include_affinity_no_affinity_keys_present(tmp_path):
    annotation_file = _write_dataset(tmp_path)
    annotations = MagfiloAnnotations(annotation_file)
    dataset = FilamentDataset(
        annotations.records, images_dir=tmp_path, annotations=annotations,
        transform=val_transforms(16), apply_disk_mask=False, include_affinity=False,
    )
    item = dataset[0]
    assert "affinity" not in item and "affinity_valid" not in item
