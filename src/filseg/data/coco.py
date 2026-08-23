"""Thin wrapper around the MAGFiLO COCO annotation file.

Only the segmentation masks are used; the dataset's bounding boxes, spines and
class labels are deliberately ignored, per the challenge's scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

IMAGE_SIZE = 2048  # every MAGFiLO frame is 2048 x 2048


@dataclass(frozen=True)
class ImageRecord:
    """One annotated H-alpha observation."""

    image_id: int
    file_name: str
    height: int
    width: int

    @property
    def stem(self) -> str:
        """Kaggle image id, e.g. ``20150125172714Mh``."""
        return Path(self.file_name).stem


class MagfiloAnnotations:
    """Read-only view over the training annotations."""

    def __init__(self, annotation_file: str | Path):
        self.coco = COCO(str(annotation_file))

    @property
    def records(self) -> list[ImageRecord]:
        out = []
        for image_id in sorted(self.coco.getImgIds()):
            info = self.coco.loadImgs(image_id)[0]
            out.append(
                ImageRecord(
                    image_id=image_id,
                    file_name=info["file_name"],
                    height=int(info.get("height", IMAGE_SIZE)),
                    width=int(info.get("width", IMAGE_SIZE)),
                )
            )
        return out

    def instance_masks(self, record: ImageRecord) -> list[np.ndarray]:
        """Per-filament binary masks at native resolution."""
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=record.image_id, iscrowd=None))
        masks = []
        for ann in anns:
            m = self.coco.annToMask(ann)
            if m.shape != (record.height, record.width):
                continue
            if m.any():
                masks.append(m.astype(np.uint8))
        return masks

    def semantic_mask(self, record: ImageRecord) -> np.ndarray:
        """Union of every filament in the image (the model's training target)."""
        canvas = np.zeros((record.height, record.width), dtype=np.uint8)
        for m in self.instance_masks(record):
            np.maximum(canvas, m, out=canvas)
        return canvas


def encode_rle(mask: np.ndarray) -> str:
    """COCO RLE *counts* string for a binary mask, as the submission expects."""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    return counts.decode("utf-8") if isinstance(counts, bytes) else counts


def decode_rle(counts: str, shape: tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE)) -> np.ndarray:
    """Inverse of :func:`encode_rle`."""
    rle = {"counts": counts.encode("utf-8"), "size": [shape[0], shape[1]]}
    return mask_utils.decode(rle).astype(np.uint8)
