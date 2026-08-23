"""Thin reader for the MAGFiLO COCO annotation file.

Only the segmentation masks are used; the dataset's bounding boxes, spines and
class labels are deliberately ignored, per the challenge's scope.

This deliberately does not instantiate ``pycocotools.coco.COCO``. Its index
(``self.imgs``, ``getImgIds`` / ``loadImgs`` / ``getAnnIds`` / ``annToMask``)
assumes ``images[].id`` and ``annotations[].image_id`` share one consistent
type across the whole file; MAGFiLO's export does not guarantee that; e.g. a
mix of int and string ids across records reliably reproduces the exact crash
seen on the real training file (``KeyError`` inside ``loadImgs``, even though
the id came straight out of ``getImgIds()``). Reading the JSON ourselves and
normalizing every id to ``str`` up front — the same normalization on both
sides of the join — sidesteps the whole class of mismatch, and
``pycocotools.mask``'s pure encode/decode functions need no index at all.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

IMAGE_SIZE = 2048  # every MAGFiLO frame is 2048 x 2048


@dataclass(frozen=True)
class ImageRecord:
    """One annotated H-alpha observation."""

    image_id: str
    file_name: str
    height: int
    width: int

    @property
    def stem(self) -> str:
        """Kaggle image id, e.g. ``20150125172714Mh``."""
        return Path(self.file_name).stem


def _segmentation_to_mask(segmentation, height: int, width: int) -> np.ndarray:
    """Polygon or RLE annotation -> binary mask, without touching a COCO index."""
    if isinstance(segmentation, list):
        rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation.get("counts"), list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation
    mask = mask_utils.decode(rle)
    return mask[..., 0] if mask.ndim == 3 else mask


class MagfiloAnnotations:
    """Read-only view over the training annotations."""

    def __init__(self, annotation_file: str | Path):
        payload = json.loads(Path(annotation_file).read_text())

        self._images: dict[str, dict] = {
            str(img["id"]): img for img in payload.get("images", [])
        }
        self._anns_by_image: dict[str, list[dict]] = defaultdict(list)
        for ann in payload.get("annotations", []):
            if ann.get("iscrowd"):
                continue
            self._anns_by_image[str(ann["image_id"])].append(ann)

    @property
    def records(self) -> list[ImageRecord]:
        out = [
            ImageRecord(
                image_id=image_id,
                file_name=info["file_name"],
                height=int(info.get("height", IMAGE_SIZE)),
                width=int(info.get("width", IMAGE_SIZE)),
            )
            for image_id, info in self._images.items()
        ]
        out.sort(key=lambda record: record.file_name)
        return out

    def instance_masks(self, record: ImageRecord) -> list[np.ndarray]:
        """Per-filament binary masks at native resolution."""
        masks = []
        for ann in self._anns_by_image.get(record.image_id, []):
            m = _segmentation_to_mask(ann["segmentation"], record.height, record.width)
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
