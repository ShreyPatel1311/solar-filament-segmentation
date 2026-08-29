#!/usr/bin/env python
"""Grid-search postprocessing against PQ, without retraining anything.

The model forward pass is the expensive part; postprocessing (threshold,
closing, min_area) is nearly free by comparison. So this runs inference
*once* per validation image, caches the raw probability map (and affinity
channels, if the model has that head), then tries every postprocess
combination purely on the cached arrays -- a full grid over N settings costs
one inference pass per image, not N.

    python scripts/sweep_postprocess.py --checkpoint artifacts/checkpoints/unet_r34_affinity_best.pt

Prints every combination sorted by pooled PQ (best first) and writes the full
grid to a JSON report. Put the winning settings into your config's
`postprocess:` block, or override at predict time with `--set`.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))  # run without installing

import argparse
import itertools
import json
from dataclasses import replace
from pathlib import Path

import torch

from filseg.config import PostprocessConfig, config_from_dict, parse_overrides
from filseg.data.coco import MagfiloAnnotations
from filseg.data.dataset import load_split, read_gray
from filseg.data.solar_disk import disk_mask
from filseg.engine.metrics import PanopticQuality
from filseg.inference import predict_probability, predict_probability_and_affinity
from filseg.models.build import load_checkpoint
from filseg.paths import resolve_paths
from filseg.postprocess import instances_from_affinity, instances_from_probability
from filseg.utils.logging import get_logger

logger = get_logger("sweep_postprocess")


def sweep_grid(cache, grid, base_postprocess):
    """Score every (threshold, min_area, closing_radius) combo against cached
    model output, sorted by pooled PQ descending. Pure function, no I/O or
    model access, so it's directly testable without a trained checkpoint.
    """
    results = []
    for threshold, min_area, closing_radius in grid:
        pp_cfg = replace(base_postprocess, threshold=threshold, min_area=min_area,
                         closing_radius=closing_radius)
        pq = PanopticQuality()
        for probability, right, down, valid, targets in cache:
            if right is not None:
                predictions = instances_from_affinity(probability, right, down, pp_cfg,
                                                      valid_mask=valid)
            else:
                predictions = instances_from_probability(probability, pp_cfg, valid_mask=valid)
            pq.update(predictions, targets)

        results.append({
            "threshold": threshold, "min_area": min_area, "closing_radius": closing_radius,
            **pq.summary(),
        })

    results.sort(key=lambda r: r["pq"], reverse=True)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="Cache only the first N val images (faster iteration).")
    parser.add_argument("--threshold", default="0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--min-area", default="30,60,120,250,500")
    parser.add_argument("--closing-radius", default="1,2,3,5,8")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="section.key=value")
    return parser.parse_args()


def _floats(csv: str) -> list[float]:
    return [float(x) for x in csv.split(",")]


def _ints(csv: str) -> list[int]:
    return [int(x) for x in csv.split(",")]


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args.data_root, args.output_root)
    paths.ensure_output_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, stored_config = load_checkpoint(args.checkpoint, device)
    cfg = config_from_dict(stored_config, parse_overrides(args.overrides))

    annotations = MagfiloAnnotations(paths.train_annotations)
    _, val_records = load_split(annotations.records, cfg.data.val_fraction, cfg.data.split_seed)
    if args.limit:
        val_records = val_records[: args.limit]

    logger.info("caching model output for %d images (one inference pass each)...",
               len(val_records))
    cache = []
    for i, record in enumerate(val_records, start=1):
        image = read_gray(paths.train_images / record.file_name)
        valid = disk_mask(image, cfg.data.limbo_margin)
        targets = annotations.instance_masks(record)

        if cfg.model.affinity_head:
            probability, right, down = predict_probability_and_affinity(model, image, cfg, device)
            cache.append((probability, right, down, valid, targets))
        else:
            probability = predict_probability(model, image, cfg, device, tta=True)
            cache.append((probability, None, None, valid, targets))

        if i % 20 == 0:
            logger.info("cached %d/%d", i, len(val_records))

    grid = list(itertools.product(
        _floats(args.threshold), _ints(args.min_area), _ints(args.closing_radius)
    ))
    logger.info("sweeping %d postprocess combinations over %d cached images...",
               len(grid), len(cache))
    results = sweep_grid(cache, grid, cfg.postprocess)

    logger.info("top 10 by pooled PQ:")
    for r in results[:10]:
        logger.info("  threshold=%.2f min_area=%4d closing=%d -> PQ %.4f  "
                   "(fp=%.0f fn=%.0f 1:many=%.0f many:1=%.0f)",
                   r["threshold"], r["min_area"], r["closing_radius"], r["pq"],
                   r["fp"], r["fn"], r["one_to_many"], r["many_to_one"])

    default = PostprocessConfig()
    baseline = next((r for r in results
                     if r["threshold"] == default.threshold
                     and r["min_area"] == default.min_area
                     and r["closing_radius"] == default.closing_radius), None)
    if baseline:
        logger.info("current config's setting scored PQ %.4f -- best found improves it by %.4f",
                   baseline["pq"], results[0]["pq"] - baseline["pq"])

    report = paths.submissions / f"{cfg.name}_postprocess_sweep.json"
    report.write_text(json.dumps(results, indent=2))
    logger.info("full grid written to %s", report)
    logger.info("best: --set postprocess.threshold=%s --set postprocess.min_area=%s "
               "--set postprocess.closing_radius=%s",
               results[0]["threshold"], results[0]["min_area"], results[0]["closing_radius"])


if __name__ == "__main__":
    main()
