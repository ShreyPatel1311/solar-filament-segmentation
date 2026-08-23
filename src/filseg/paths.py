"""Environment-aware path resolution.

The same code runs in two places:

* locally, where the repo is the working directory and data lives under ``data/``;
* on Kaggle, where the repo is cloned into ``/kaggle/working/repo`` and the
  competition data is mounted read-only under ``/kaggle/input``.

Every path used by the pipeline is resolved here so that no script needs to
know which environment it is in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

COMPETITION = "filament-segmentation-2026"
DATASET_DIRNAME = "MAGFiLO_1.0_Kaggle_2026"


def on_kaggle() -> bool:
    """True when running inside a Kaggle notebook/kernel."""
    return Path("/kaggle/input").exists() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def repo_root() -> Path:
    """Root of this repository, regardless of where it was cloned."""
    return Path(__file__).resolve().parents[2]


def _first_existing(*candidates: Path) -> Path | None:
    for c in candidates:
        if c.exists():
            return c
    return None


@dataclass(frozen=True)
class Paths:
    """Resolved locations for data, checkpoints and submissions."""

    data_root: Path
    output_root: Path

    @property
    def train_images(self) -> Path:
        return self.data_root / "train" / "train_images"

    @property
    def train_annotations(self) -> Path:
        train = self.data_root / "train"
        found = sorted(train.glob("*.json"))
        if not found:
            raise FileNotFoundError(f"No COCO annotation JSON under {train}")
        return found[0]

    @property
    def test_images(self) -> Path:
        return self.data_root / "test" / "test_images"

    @property
    def checkpoints(self) -> Path:
        return self.output_root / "checkpoints"

    @property
    def submissions(self) -> Path:
        return self.output_root / "submissions"

    def ensure_output_dirs(self) -> None:
        for d in (self.checkpoints, self.submissions):
            d.mkdir(parents=True, exist_ok=True)


def resolve_paths(data_root: str | os.PathLike | None = None,
                  output_root: str | os.PathLike | None = None) -> Paths:
    """Resolve data/output roots from an explicit override, env vars, or defaults.

    Precedence: explicit argument > ``FILSEG_DATA_ROOT`` / ``FILSEG_OUTPUT_ROOT``
    environment variable > environment default.
    """
    root = data_root or os.environ.get("FILSEG_DATA_ROOT")
    if root is None:
        guess = _first_existing(
            Path("/kaggle/input") / COMPETITION / DATASET_DIRNAME,
            Path("/kaggle/input/competitions") / COMPETITION / DATASET_DIRNAME,
            repo_root() / "data" / DATASET_DIRNAME,
            repo_root() / "data",
        )
        if guess is None:
            raise FileNotFoundError(
                "Could not locate the dataset. Pass --data-root or set FILSEG_DATA_ROOT."
            )
        root = guess

    out = output_root or os.environ.get("FILSEG_OUTPUT_ROOT")
    if out is None:
        out = Path("/kaggle/working") if on_kaggle() else repo_root() / "artifacts"

    return Paths(data_root=Path(root), output_root=Path(out))
