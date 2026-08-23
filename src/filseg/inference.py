"""Inference and submission writing.

Prediction runs at the model's training resolution and the probability map is
upsampled back to the native 2048x2048 grid before thresholding, so the RLE
written to the submission is always in the coordinate frame the scorer expects.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from filseg.config import Config
from filseg.data.coco import IMAGE_SIZE, encode_rle
from filseg.data.dataset import read_gray
from filseg.data.solar_disk import disk_mask
from filseg.data.transforms import val_transforms
from filseg.postprocess import instances_from_probability
from filseg.utils.logging import get_logger

logger = get_logger(__name__)

SUBMISSION_COLUMNS = ("filament_id", "segmentation_rle")


def list_test_images(images_dir: str | Path) -> list[Path]:
    return sorted(Path(images_dir).glob("*.jpeg")) + sorted(Path(images_dir).glob("*.jpg"))


@torch.no_grad()
def predict_probability(model: nn.Module, image: np.ndarray, cfg: Config,
                        device: torch.device, tta: bool = True) -> np.ndarray:
    """Foreground probability at native resolution, optionally flip-averaged."""
    transform = val_transforms(cfg.data.image_size)
    tensor = transform(image=image)["image"].float().unsqueeze(0).to(device)

    variants = [tensor]
    if tta:
        variants += [torch.flip(tensor, dims=[3]), torch.flip(tensor, dims=[2])]

    accumulator = torch.zeros(
        (1, 1, cfg.data.image_size, cfg.data.image_size), device=device, dtype=torch.float32
    )
    for index, variant in enumerate(variants):
        logits = model(variant).float()
        if index == 1:
            logits = torch.flip(logits, dims=[3])
        elif index == 2:
            logits = torch.flip(logits, dims=[2])
        accumulator += torch.sigmoid(logits)

    probability = (accumulator / len(variants)).squeeze().cpu().numpy()
    height, width = image.shape[:2]
    if probability.shape != (height, width):
        probability = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)
    return probability


def predict_instances(model: nn.Module, image_path: Path, cfg: Config,
                      device: torch.device, tta: bool = True) -> list[np.ndarray]:
    """All filament masks found in one image."""
    image = read_gray(image_path)
    probability = predict_probability(model, image, cfg, device, tta=tta)
    valid = disk_mask(image, cfg.data.limbo_margin)
    return instances_from_probability(probability, cfg.postprocess, valid_mask=valid)


def iter_submission_rows(model: nn.Module, image_paths: Iterable[Path], cfg: Config,
                         device: torch.device, tta: bool = True) -> Iterator[tuple[str, str]]:
    """Yield ``(filament_id, rle)`` pairs for every predicted instance."""
    for path in image_paths:
        stem = path.stem
        masks = predict_instances(model, path, cfg, device, tta=tta)
        if not masks:
            logger.warning("no filaments predicted for %s", stem)
        for index, mask in enumerate(masks, start=1):
            if mask.shape != (IMAGE_SIZE, IMAGE_SIZE):
                logger.warning("%s: mask is %s, expected %d x %d",
                               stem, mask.shape, IMAGE_SIZE, IMAGE_SIZE)
            yield f"{stem}_{index}", encode_rle(mask)


def write_submission(rows: Iterable[tuple[str, str]], path: str | Path) -> int:
    """Write the competition CSV; returns the number of rows written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(SUBMISSION_COLUMNS)
        for filament_id, rle in rows:
            writer.writerow([filament_id, rle])
            written += 1
    logger.info("wrote %d predicted filaments to %s", written, path)
    return written
