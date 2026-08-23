#!/usr/bin/env python
"""Run a trained checkpoint over the test set and write submission.csv.

    python scripts/predict.py --checkpoint artifacts/checkpoints/unet_r34_best.pt
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))  # run without installing

import argparse
from pathlib import Path

import torch

from filseg.config import config_from_dict, load_config, parse_overrides
from filseg.inference import iter_submission_rows, list_test_images, write_submission
from filseg.models.build import load_checkpoint
from filseg.paths import resolve_paths
from filseg.utils.logging import get_logger

logger = get_logger("predict")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None,
                        help="Defaults to the config stored inside the checkpoint.")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="Submission CSV path.")
    parser.add_argument("--images", type=Path, default=None,
                        help="Directory of images to predict; defaults to the test split.")
    parser.add_argument("--no-tta", action="store_true", help="Disable flip test-time averaging.")
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

    images_dir = args.images or paths.test_images
    image_paths = list_test_images(images_dir)
    logger.info("predicting %d images from %s on %s", len(image_paths), images_dir, device)

    out = args.out or paths.submissions / "submission.csv"
    rows = iter_submission_rows(model, image_paths, cfg, device, tta=not args.no_tta)
    write_submission(rows, out)


if __name__ == "__main__":
    main()
