#!/usr/bin/env python
"""Run a trained checkpoint over the test set and write submission.csv.

    python scripts/predict.py --checkpoint artifacts/checkpoints/unet_r34_best.pt
    python scripts/predict.py --hf-repo-id me/filament-unet-r34
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))  # run without installing

import argparse
from pathlib import Path

import torch

from filseg import hub
from filseg.config import config_from_dict, load_config, parse_overrides
from filseg.inference import iter_submission_rows, list_test_images, write_submission
from filseg.models.build import load_checkpoint
from filseg.paths import resolve_paths
from filseg.utils.logging import get_logger

logger = get_logger("predict")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Local checkpoint. Mutually exclusive with --hf-repo-id.")
    parser.add_argument("--hf-repo-id", default=None, metavar="USER/REPO",
                        help="Download the checkpoint from this Hub model repo instead of "
                             "--checkpoint.")
    parser.add_argument("--hf-filename", default=hub.DEFAULT_FILENAME)
    parser.add_argument("--hf-token", default=None, help="Defaults to the HF_TOKEN env var.")
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
    args = parser.parse_args()
    if bool(args.checkpoint) == bool(args.hf_repo_id):
        parser.error("pass exactly one of --checkpoint or --hf-repo-id")
    return args


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args.data_root, args.output_root)
    paths.ensure_output_dirs()

    if args.hf_repo_id:
        checkpoint_path = hub.download_checkpoint(
            args.hf_repo_id, filename=args.hf_filename,
            local_dir=paths.checkpoints, token=args.hf_token,
        )
        logger.info("downloaded %s/%s -> %s", args.hf_repo_id, args.hf_filename, checkpoint_path)
    else:
        checkpoint_path = args.checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, stored_config = load_checkpoint(checkpoint_path, device)

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
