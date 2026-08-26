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


class _PoisonThenCleanModel(torch.nn.Module):
    """A model that poisons its own BatchNorm on the first batch, then behaves.

    Reproduces the real incident: flat_unet's attention block produced a
    non-finite forward pass under amp on the actual GPU run, which corrupted
    every downstream BatchNorm layer's running stats permanently -- training
    loss looked healthy afterward (train-mode BN uses batch stats), but
    validation stayed nan forever (eval-mode BN uses the poisoned running
    stats). The Trainer must self-heal from this, not just skip the batch.
    """

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=1)
        self.bn = torch.nn.BatchNorm2d(1)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        if self.calls == 1:
            # Simulate an internal fp16 overflow: the forward pass itself is
            # non-finite, so BatchNorm's running-stat update is poisoned by it.
            x = self.conv(x) * float("nan")
        else:
            x = self.conv(x)
        return self.bn(x)


def test_recovers_from_a_poisoned_batchnorm_instead_of_validating_nan_forever(tmp_path):
    cfg = Config()
    cfg.train.epochs = 1
    cfg.train.amp = False

    model = _PoisonThenCleanModel()
    trainer = Trainer(model, cfg, tmp_path, device="cpu")

    # The first forward poisons every BatchNorm layer; _train_epoch's guard
    # detects and resets it inline, in the same batch, before it ever reaches
    # validation. Without the fix, running_mean would stay nan forever here.
    trainer._train_epoch(_loader(4))
    assert torch.isfinite(model.bn.running_mean).all()
    assert torch.isfinite(model.bn.running_var).all()

    # And validation, which uses running stats, must now be finite again.
    metrics = trainer._validate(_loader(4))
    assert torch.isfinite(torch.tensor(metrics["val_loss"]))


def test_full_fit_recovers_and_reaches_finite_validation(tmp_path):
    """End-to-end: fit() must not leave every epoch's val nan after one bad batch."""
    cfg = Config()
    cfg.train.epochs = 2
    cfg.train.amp = False

    trainer = Trainer(_PoisonThenCleanModel(), cfg, tmp_path, device="cpu")
    result = trainer.fit(_loader(4), _loader(4))

    assert torch.isfinite(torch.tensor(trainer.history[-1]["val_loss"]))
    assert "best_val_dice" in result
