#!/usr/bin/env python
"""Score a checkpoint on the held-out validation split with the leaderboard metric.

    python scripts/evaluate.py --checkpoint artifacts/checkpoints/unet_r34_best.pt
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))  # run without installing

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from filseg.config import config_from_dict, load_config, parse_overrides
from filseg.data.coco import MagfiloAnnotations
from filseg.data.dataset import load_split
from filseg.engine.metrics import PanopticQuality
from filseg.inference import predict_instances
from filseg.models.build import load_checkpoint
from filseg.paths import resolve_paths
from filseg.utils.logging import get_logger

logger = get_logger("evaluate")


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    denominator = int(a.sum()) + int(b.sum())
    return 2 * float(np.logical_and(a, b).sum()) / denominator if denominator else 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N images.")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="section.key=value")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args.data_root, args.output_root)
    paths.ensure_output_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, stored_config = load_checkpoint(args.checkpoint, device)
    overrides = parse_overrides(args.overrides)
    cfg = (load_config(args.config, overrides) if args.config
           else config_from_dict(stored_config, overrides))

    annotations = MagfiloAnnotations(paths.train_annotations)
    _, val_records = load_split(annotations.records, cfg.data.val_fraction, cfg.data.split_seed)
    if args.limit:
        val_records = val_records[: args.limit]

    pq = PanopticQuality()
    image_dice: list[float] = []
    for i, record in enumerate(val_records, start=1):
        predictions = predict_instances(
            model, paths.train_images / record.file_name, cfg, device
        )
        targets = annotations.instance_masks(record)
        pq.update(predictions, targets)

        pred_union = np.zeros((record.height, record.width), dtype=np.uint8)
        for m in predictions:
            np.maximum(pred_union, m, out=pred_union)
        image_dice.append(_dice(pred_union, annotations.semantic_mask(record)))

        if i % 10 == 0:
            logger.info("%d/%d images | running PQ %.4f", i, len(val_records), pq.score)

    summary = {
        **pq.summary(),
        "mean_image_dice": float(np.mean(image_dice)) if image_dice else 0.0,
        "mean_matched_iou": float(np.mean(pq.matched_ious)) if pq.matched_ious else 0.0,
        "images": len(val_records),
    }
    report = paths.submissions / f"{cfg.name}_val_report.json"
    report.write_text(json.dumps(summary, indent=2))
    logger.info("%s", json.dumps(summary, indent=2))
    logger.info("report written to %s", report)


if __name__ == "__main__":
    main()
