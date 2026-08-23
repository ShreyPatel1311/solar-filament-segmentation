"""Dataset and dataloader construction."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from filseg.config import Config
from filseg.data.coco import ImageRecord, MagfiloAnnotations
from filseg.data.solar_disk import disk_mask
from filseg.data.transforms import train_transforms, val_transforms


def read_gray(path: str | Path) -> np.ndarray:
    """Load an H-alpha frame as a single-channel uint8 array."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


class FilamentDataset(Dataset):
    """Grayscale image + binary filament mask, both resized to ``image_size``.

    The target is the *union* of filaments; individual instances are recovered
    after inference by connected components (see :mod:`filseg.postprocess`),
    which keeps the model small enough to train inside a Kaggle session.
    """

    def __init__(
        self,
        records: list[ImageRecord],
        images_dir: Path,
        annotations: MagfiloAnnotations,
        transform,
        apply_disk_mask: bool = True,
        disk_margin: float = 0.98,
    ):
        self.records = records
        self.images_dir = Path(images_dir)
        self.annotations = annotations
        self.transform = transform
        self.apply_disk_mask = apply_disk_mask
        self.disk_margin = disk_margin

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        image = read_gray(self.images_dir / record.file_name)
        mask = self.annotations.semantic_mask(record)

        if self.apply_disk_mask:
            mask = mask * disk_mask(image, self.disk_margin)

        augmented = self.transform(image=image, mask=mask)
        target = augmented["mask"].float().unsqueeze(0)
        return {"image": augmented["image"].float(), "mask": target, "stem": record.stem}


def load_split(records: list[ImageRecord], val_fraction: float, seed: int
               ) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Deterministic train/validation split, shared by every environment."""
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    return shuffled[n_val:], shuffled[:n_val]


def build_dataloaders(cfg: Config, images_dir: Path, annotation_file: Path
                      ) -> tuple[DataLoader, DataLoader]:
    annotations = MagfiloAnnotations(annotation_file)
    train_records, val_records = load_split(
        annotations.records, cfg.data.val_fraction, cfg.data.split_seed
    )

    common = dict(
        images_dir=images_dir,
        annotations=annotations,
        disk_margin=cfg.data.limbo_margin,
    )
    train_ds = FilamentDataset(
        train_records, transform=train_transforms(cfg.data.image_size), **common
    )
    val_ds = FilamentDataset(
        val_records, transform=val_transforms(cfg.data.image_size), **common
    )

    loader_kwargs = dict(
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.data.num_workers > 0,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True, **loader_kwargs
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, cfg.train.batch_size // 2), shuffle=False, **loader_kwargs
    )
    return train_loader, val_loader
