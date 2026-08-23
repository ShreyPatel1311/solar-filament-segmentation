#!/usr/bin/env python
"""Train a filament segmentation model.

    python scripts/train.py --config configs/unet_resnet34.yaml
    python scripts/train.py --config configs/smoke.yaml --set train.epochs=1
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))  # run without installing

import argparse
import json
from pathlib import Path

from filseg.config import load_config, parse_overrides
from filseg.data.dataset import build_dataloaders
from filseg.engine.trainer import Trainer
from filseg.models.build import build_model
from filseg.paths import resolve_paths
from filseg.utils.logging import get_logger
from filseg.utils.seed import seed_everything

logger = get_logger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/unet_resnet34.yaml"))
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Overrides FILSEG_DATA_ROOT / auto-detection.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="section.key=value", help="Ad-hoc config override; repeatable.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, parse_overrides(args.overrides))
    paths = resolve_paths(args.data_root, args.output_root)
    paths.ensure_output_dirs()

    logger.info("config: %s", json.dumps(cfg.to_dict()))
    logger.info("data root: %s", paths.data_root)

    seed_everything(cfg.train.seed)
    train_loader, val_loader = build_dataloaders(
        cfg, paths.train_images, paths.train_annotations
    )
    logger.info("train batches: %d | val batches: %d", len(train_loader), len(val_loader))

    model = build_model(cfg.model)
    trainer = Trainer(model, cfg, paths.checkpoints)
    result = trainer.fit(train_loader, val_loader)
    logger.info("done: %s", json.dumps(result))


if __name__ == "__main__":
    main()
