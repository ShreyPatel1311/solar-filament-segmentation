"""A diverged run must fail fast, not silently burn through every epoch.

Regression for a real incident: dilation-122436 with model.norm=false went to
train nan on epoch 1 and stayed there for 15 epochs (~90 minutes of Kaggle GPU
time) before anyone noticed, because nothing detected the divergence.
"""

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from filseg.config import Config
from filseg.engine.trainer import Trainer


class _ConstantMaskDataset(Dataset):
    def __init__(self, n: int, size: int = 8):
        self.n = n
        self.size = size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index):
        image = torch.randn(1, self.size, self.size)
        mask = torch.zeros(1, self.size, self.size)
        mask[0, 2:4, 2:4] = 1.0
        return {"image": image, "mask": mask, "stem": str(index)}


class _AlwaysNaN(torch.nn.Module):
    """Stands in for a diverged model: every forward pass is already non-finite."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x) * float("nan")


def _loader(n: int, batch_size: int = 2) -> DataLoader:
    return DataLoader(_ConstantMaskDataset(n), batch_size=batch_size)


def test_diverged_training_raises_within_the_first_epoch(tmp_path):
    cfg = Config()
    cfg.train.epochs = 5
    cfg.train.amp = False

    trainer = Trainer(_AlwaysNaN(), cfg, tmp_path, device="cpu")

    with pytest.raises(FloatingPointError, match="non-finite"):
        trainer.fit(_loader(64), _loader(8))


def test_occasional_non_finite_batches_do_not_abort_training(tmp_path, monkeypatch):
    """A handful of isolated bad batches should be skipped, not fatal."""
    cfg = Config()
    cfg.train.epochs = 1
    cfg.train.amp = False

    model = torch.nn.Conv2d(1, 1, kernel_size=1)
    trainer = Trainer(model, cfg, tmp_path, device="cpu")

    calls = {"n": 0}
    real_criterion = trainer.criterion

    def flaky(logits, targets):
        calls["n"] += 1
        if calls["n"] <= 3:  # fewer than NON_FINITE_PATIENCE
            return logits.sum() * float("nan")
        return real_criterion(logits, targets)

    monkeypatch.setattr(trainer, "criterion", flaky)

    result = trainer.fit(_loader(20), _loader(4))
    assert "best_val_dice" in result  # completed without raising
