#!/usr/bin/env python
"""Train several configs at once, one per GPU.

Kaggle's "T4 x2" accelerator is two physically separate 15GB T4s, but a single
training run only ever touches cuda:0 -- the second card sits idle for the
whole session. This launcher fills it: two configs training concurrently on
two GPUs finish in roughly the wall-clock time of one, which matters against
Kaggle's session limit and weekly GPU quota.

    python scripts/train_parallel.py configs/flat_unet.yaml configs/diercke_unet.yaml

Each config runs as its own ``scripts/train.py`` subprocess with
``CUDA_VISIBLE_DEVICES`` pinned to one device, so every process sees exactly
one GPU, numbered cuda:0. Nothing in the training code needs to know it is
sharing a machine, and each run writes its own ``{name}_best.pt``,
``{name}_last.pt`` and ``{name}_history.json`` under the usual output
directory -- no collision as long as the configs' ``name:`` fields differ,
which is checked before anything launches.

This is deliberately process-parallel, not DistributedDataParallel: the goal
is two *different* models compared under identical data, not one model trained
faster. A crash in one run leaves the other untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))  # run without installing

import argparse
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

import yaml

from filseg.utils.logging import get_logger

logger = get_logger("train_parallel")

_TRAIN_SCRIPT = Path(__file__).resolve().parent / "train.py"


def available_gpus() -> int:
    """GPU count, without importing torch into the launcher process."""
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:  # torch missing or CUDA unavailable: fall back to CPU
        return 0


def config_name(path: Path) -> str:
    payload = yaml.safe_load(path.read_text()) or {}
    return payload.get("name", path.stem)


def _pump(stream, label: str, sink: queue.Queue) -> None:
    """Forward one subprocess's output line by line, tagged with its config."""
    for line in iter(stream.readline, ""):
        sink.put(f"[{label}] {line.rstrip()}")
    stream.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="+", type=Path,
                        help="Config YAMLs to train, one per GPU.")
    parser.add_argument("--gpus", default=None,
                        help="Comma-separated device ids, e.g. '0,1'. "
                             "Defaults to every visible GPU.")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="section.key=value",
                        help="Override applied to every run; repeatable.")
    parser.add_argument("--hf-repo-id", default=None, metavar="USER/REPO",
                        help="Push each run's best checkpoint here. Each is uploaded "
                             "under its own '{name}_best.pt' so the runs cannot "
                             "overwrite one another.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    for path in args.configs:
        if not path.exists():
            raise SystemExit(f"No such config: {path}")

    # Distinct name: fields keep the runs' checkpoints from overwriting each other.
    names = [config_name(p) for p in args.configs]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise SystemExit(
            f"Configs share name: {sorted(duplicates)} -- their checkpoints would "
            "overwrite each other. Give each config a distinct 'name:'."
        )

    if args.gpus is not None:
        devices = [d.strip() for d in args.gpus.split(",") if d.strip()]
    else:
        devices = [str(i) for i in range(available_gpus())]

    if not devices:
        logger.warning("no GPUs visible -- running on CPU, sequentially in parallel "
                       "processes (this will be slow)")
        devices = [""]

    if len(args.configs) > len(devices):
        logger.warning(
            "%d configs but only %d device(s): %d run(s) will share a GPU and may "
            "run out of memory.", len(args.configs), len(devices), 
            len(args.configs) - len(devices),
        )

    output = queue.Queue()
    processes: list[tuple[str, subprocess.Popen]] = []

    for index, (config, name) in enumerate(zip(args.configs, names, strict=True)):
        device = devices[index % len(devices)]

        env = dict(os.environ)
        if device:
            # The child sees exactly one GPU, as cuda:0 -- so train.py's plain
            # torch.device("cuda") lands on the right card with no changes.
            env["CUDA_VISIBLE_DEVICES"] = device
        env["PYTHONUNBUFFERED"] = "1"  # otherwise logs arrive only at exit

        command = [sys.executable, str(_TRAIN_SCRIPT), "--config", str(config)]
        if args.data_root:
            command += ["--data-root", str(args.data_root)]
        if args.output_root:
            command += ["--output-root", str(args.output_root)]
        for override in args.overrides:
            command += ["--set", override]
        if args.hf_repo_id:
            command += ["--hf-repo-id", args.hf_repo_id,
                        "--hf-filename", f"{name}_best.pt"]

        logger.info("launching %s on GPU %s", name, device or "cpu")
        process = subprocess.Popen(
            command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        processes.append((name, process))
        threading.Thread(target=_pump, args=(process.stdout, name, output),
                         daemon=True).start()

    started = time.time()
    try:
        while any(p.poll() is None for _, p in processes):
            try:
                print(output.get(timeout=0.2), flush=True)
            except queue.Empty:
                pass
    except KeyboardInterrupt:
        logger.warning("interrupted -- terminating %d run(s)", len(processes))
        for _, process in processes:
            process.terminate()
        raise

    while not output.empty():  # drain whatever landed after the last poll
        print(output.get(), flush=True)

    elapsed = time.time() - started
    failures = [(name, p.returncode) for name, p in processes if p.returncode != 0]

    logger.info("all runs finished in %.1f min", elapsed / 60)
    for name, process in processes:
        status = "ok" if process.returncode == 0 else f"FAILED ({process.returncode})"
        logger.info("  %-20s %s", name, status)

    if failures:
        logger.error("%d of %d run(s) failed", len(failures), len(processes))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
