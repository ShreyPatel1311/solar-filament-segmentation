"""Dataset and dataloader construction."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from filseg.config import Config
from filseg.data.affinity import affinity_targets
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

    ``include_affinity=True`` additionally returns per-pixel "same instance as
    my right/bottom neighbor?" ground truth (see :mod:`filseg.data.affinity`),
    for models trained with an affinity head to directly learn where touching
    filaments should be split rather than merged by connected components.
    """

    def __init__(
        self,
        records: list[ImageRecord],
        images_dir: Path,
        annotations: MagfiloAnnotations,
        transform,
        apply_disk_mask: bool = True,
        disk_margin: float = 0.98,
        include_affinity: bool = False,
    ):
        self.records = records
        self.images_dir = Path(images_dir)
        self.annotations = annotations
        self.transform = transform
        self.apply_disk_mask = apply_disk_mask
        self.disk_margin = disk_margin
        # Opt-in: existing configs/checkpoints are unaffected, since batches
        # without "affinity"/"affinity_valid" keys behave exactly as before
        # everywhere downstream (Trainer, DiceBCELoss).
        self.include_affinity = include_affinity

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        image = read_gray(self.images_dir / record.file_name)

        if self.include_affinity:
            instance_masks = self.annotations.instance_masks(record)
            mask = np.zeros_like(image, dtype=np.uint8)
            for instance in instance_masks:
                np.maximum(mask, instance, out=mask)
        else:
            instance_masks = None
            mask = self.annotations.semantic_mask(record)

        valid_region = disk_mask(image, self.disk_margin) if self.apply_disk_mask else None
        if valid_region is not None:
            mask = mask * valid_region
            if instance_masks is not None:
                instance_masks = [m * valid_region for m in instance_masks]

        if instance_masks is not None:
            # Pass even an empty list: an image with zero filaments must still
            # produce "affinity"/"affinity_valid" keys, or a batch mixing it
            # with images that do have instances would collate inconsistently.
            augmented = self.transform(image=image, mask=mask, masks=instance_masks)
        else:
            augmented = self.transform(image=image, mask=mask)

        target = augmented["mask"].float().unsqueeze(0)
        item: dict[str, torch.Tensor | str] = {
            "image": augmented["image"].float(), "mask": target, "stem": record.stem,
        }

        if instance_masks is not None:
            resized_instances = [
                m.numpy() if hasattr(m, "numpy") else np.asarray(m) for m in augmented["masks"]
            ]
            affinity, valid = affinity_targets(resized_instances, target.shape[-2:])
            item["affinity"] = torch.from_numpy(affinity)
            item["affinity_valid"] = torch.from_numpy(valid)

        return item


def load_split(records: list[ImageRecord], val_fraction: float, seed: int
               ) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Deterministic train/validation split, grouped by observing day.

    Filaments evolve over hours, not seconds, so two frames from the same
    calendar day are often nearly identical -- splitting by individual image
    lets near-duplicates leak across train/val, inflating validation metrics
    without reflecting real generalization to genuinely unseen days. Measured
    directly on a naive image-level split of the real MAGFiLO data: 62% of
    the "held-out" val images shared a day with a training image. Every image
    from one observing day (``ImageRecord.stem[:8]``, its YYYYMMDD prefix) now
    goes entirely to train or entirely to val, never split between them.
    """
    by_date: dict[str, list[ImageRecord]] = {}
    for record in records:
        by_date.setdefault(record.stem[:8], []).append(record)

    dates = list(by_date)  # insertion order follows MagfiloAnnotations.records'
    if len(dates) < 2:     # sort-by-filename, so this is already deterministic
        raise ValueError(
            f"Only {len(dates)} distinct observing day(s) in {len(records)} "
            "records -- cannot make a leak-free train/val split."
        )
    random.Random(seed).shuffle(dates)

    target_val_images = max(1, round(len(records) * val_fraction))
    val_dates: set[str] = set()
    val_count = 0
    for date in dates:
        val_dates.add(date)
        val_count += len(by_date[date])
        if val_count >= target_val_images:
            break

    train_records = [r for d in dates if d not in val_dates for r in by_date[d]]
    val_records = [r for d in val_dates for r in by_date[d]]
    return train_records, val_records


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
        include_affinity=cfg.model.affinity_head,
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
