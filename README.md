# Solar Filament Segmentation — MAGFiLO / IEEE BigData Cup 2026

Instance segmentation of solar filaments in GONG H-alpha observations, for the
[Filament Segmentation Challenge 2026](https://kaggle.com/competitions/filament-segmentation-2026).

**The working model:** the code lives here and is edited locally; Kaggle only
*runs* it. A Kaggle notebook clones this repository at a pinned revision and
calls the same CLI entry points you use on your machine, so a run is fully
described by a commit SHA plus a config file. Nothing is copy-pasted into a
notebook cell.

```
local edit  ->  git push  ->  Kaggle notebook clones the revision  ->  train/predict  ->  submission.csv
```

## Layout

```
configs/                 experiment definitions (YAML) - one file per run
requirements.txt         full pinned environment (local / reference)
requirements-kaggle*.txt the few packages the Kaggle image lacks
notebooks/
  kaggle_runner.ipynb    train (+ optionally push the best checkpoint to HF Hub)
  kaggle_submit.ipynb    submission-only: pull a checkpoint from HF Hub, predict
scripts/
  train.py               fit a model; --hf-repo-id pushes the best checkpoint
  evaluate.py            score a checkpoint with the leaderboard metric (PQ)
  predict.py             write submission.csv; --checkpoint or --hf-repo-id
src/filseg/
  paths.py               resolves data/output roots for Kaggle vs. local
  config.py              typed config, YAML + `--set section.key=value` overrides
  hub.py                 push/pull the best checkpoint to a HF Hub model repo
  data/
    coco.py              MAGFiLO annotations, RLE encode/decode
    dataset.py           torch Dataset + dataloaders, deterministic split
    transforms.py        augmentation pipelines
    solar_disk.py        fits the solar disk; suppresses off-limb false positives
  models/build.py        architecture zoo + checkpoint I/O
  engine/
    losses.py            BCE + soft Dice
    metrics.py           Panoptic Quality, Dice, IoU
    trainer.py            training loop, best-checkpoint tracking
  postprocess.py         probability map -> individual filament instances
  inference.py           prediction, TTA, submission writing
tests/                   unit tests for the metric, RLE, post-processing, HF sync
```

## Running on Kaggle

Training and submission are two separate notebooks, so you can iterate on
post-processing or TTA and resubmit in minutes without paying for a GPU
training run each time.

**Train** (`notebooks/kaggle_runner.ipynb`):

1. Push your changes to the public GitHub repository.
2. Open the notebook on Kaggle (*File -> Import Notebook*, or create a notebook
   and paste its cells).
3. Attach the competition dataset, enable **Internet** and a **GPU** accelerator.
4. Edit the first cell — `REPO_URL`, `REVISION`, `CONFIG`, `OVERRIDES`, and
   optionally `HF_REPO_ID` (see below) — and run all cells.

It writes `/kaggle/working/submission.csv`, submittable directly from the
notebook output, and if `HF_REPO_ID` is set, pushes the best checkpoint there.

**Submit from a checkpoint already on HF Hub** (`notebooks/kaggle_submit.ipynb`):

1. Set `REPO_URL`/`REVISION` and `HF_REPO_ID` (the repo `train.py` pushed to)
   in the first cell, and run all cells. No training code runs here; a GPU is
   optional.

Dependency versions are pinned in `requirements-kaggle.txt` and
`requirements-kaggle-nodeps.txt`, not in either notebook, so fixing a broken
dependency is a commit and a re-run rather than a notebook edit. The `--no-deps`
half exists so pip cannot replace the image's preinstalled CUDA build of torch.

For a final, reproducible run, set `REVISION` to a full commit SHA rather than
`main`; both notebooks print the commit they actually ran.

### Pushing / pulling checkpoints via Hugging Face Hub

`train.py --hf-repo-id USER/REPO` uploads the best checkpoint to that model
repo under a fixed filename (`best_model.pt` by default, `--hf-filename` to
change it). Every push overwrites that same file, so the repo always holds
exactly one live checkpoint — a new training run's best replaces the old one,
rather than accumulating one file per run. (Hub keeps prior versions in its own
git history if you ever need to roll back; the repo's file tree itself never
grows.)

`predict.py --hf-repo-id USER/REPO` downloads that same file and predicts with
it, as an alternative to `--checkpoint <local path>` (pass exactly one).

Both need write (`train.py`) or read (`predict.py`) access to the Hub repo via
`HF_TOKEN` — set it as a Kaggle secret (*Add-ons -> Secrets*, name it
`HF_TOKEN`) and both notebooks load it automatically; locally, `export
HF_TOKEN=hf_...` or pass `--hf-token`. A public model repo only needs a token
for `train.py`'s push.

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . && pip install pytest ruff
```

Place (or symlink) the competition data at `data/MAGFiLO_1.0_Kaggle_2026/`, or
point `FILSEG_DATA_ROOT` at it. Then:

```bash
python scripts/train.py --config configs/smoke.yaml           # one tiny epoch, CPU-friendly
python scripts/train.py --config configs/unet_resnet34.yaml
python scripts/evaluate.py --checkpoint artifacts/checkpoints/unet_r34_best.pt
python scripts/predict.py  --checkpoint artifacts/checkpoints/unet_r34_best.pt
```

Any config field can be overridden without editing YAML:

```bash
python scripts/train.py --config configs/unet_resnet34.yaml --set train.epochs=25 --set data.image_size=768
```

| Variable | Meaning | Default |
| --- | --- | --- |
| `FILSEG_DATA_ROOT` | Dataset root containing `train/` and `test/` | auto-detected (`/kaggle/input/...` or `./data`) |
| `FILSEG_OUTPUT_ROOT` | Where checkpoints and submissions go | `/kaggle/working` on Kaggle, else `./artifacts` |
| `HF_TOKEN` | Hugging Face token for `--hf-repo-id` (or pass `--hf-token`) | none |

To push/pull via HF Hub locally, the same flags as on Kaggle apply:

```bash
python scripts/train.py   --config configs/unet_resnet34.yaml --hf-repo-id you/filament-unet-r34
python scripts/predict.py --hf-repo-id you/filament-unet-r34
```

## Method

1. **Preprocess** — grayscale frame, solar disk fitted by Otsu + minimum
   enclosing circle, everything outside 98% of the radius discarded.
2. **Segment** — U-Net (ResNet-34 encoder, ImageNet weights) predicting a single
   foreground channel at 1024 px, trained with BCE + soft Dice against the union
   of the annotated filament masks.
3. **Post-process** — upsample the probability map to the native 2048x2048 grid,
   threshold, morphologically close so a filament broken by noise stays one
   object, drop components under `min_area`, and emit each remaining connected
   component as one instance.
4. **Submit** — each instance is COCO-RLE encoded as `<image_id>_<n>`.

Steps 1, 3 and 4 target Panoptic Quality specifically: PQ charges 0.5 for every
spurious segment and every missed one, so fragmentation and speckle cost more
than a slightly loose boundary does.

## Evaluation

`scripts/evaluate.py` scores the held-out split with the same definition used on
the leaderboard — greedy one-to-one matching at IoU > 0.5, then
`PQ = sum(IoU over TP) / (|TP| + 0.5|FP| + 0.5|FN|)` — and also reports SQ, RQ,
mean image Dice, and the counts of one-to-many / many-to-one relations that the
rubric asks about.

## Development

```bash
pytest -q          # metric, RLE round-trip, post-processing, config
ruff check src scripts tests
```

CI runs both on every push and pull request.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgment

This work utilizes GONG data obtained by the NSO Integrated Synoptic Program,
managed by the National Solar Observatory, operated by AURA, Inc. under a
cooperative agreement with the National Science Foundation and with contribution
from the National Oceanic and Atmospheric Administration.
