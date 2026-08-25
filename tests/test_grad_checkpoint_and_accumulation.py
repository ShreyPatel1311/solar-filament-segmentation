"""Gradient checkpointing must not change what the model computes, and
gradient accumulation must not change how the optimizer updates it.
"""

import torch

from filseg.config import Config
from filseg.engine.trainer import Trainer
from filseg.models.improved_unet import build_improved_unet


def test_grad_checkpoint_matches_plain_forward_and_backward():
    torch.manual_seed(0)
    plain = build_improved_unet("dilation122436", base_channels=4, grad_checkpoint=False)
    torch.manual_seed(0)
    checkpointed = build_improved_unet("dilation122436", base_channels=4, grad_checkpoint=True)

    x = torch.randn(1, 1, 32, 32, requires_grad=True)
    x2 = x.detach().clone().requires_grad_(True)

    out_plain = plain(x)
    out_ckpt = checkpointed(x2)
    assert torch.allclose(out_plain, out_ckpt, atol=1e-5)

    out_plain.sum().backward()
    out_ckpt.sum().backward()
    for (name, p_plain), (_, p_ckpt) in zip(plain.named_parameters(),
                                             checkpointed.named_parameters(), strict=True):
        assert torch.allclose(p_plain.grad, p_ckpt.grad, atol=1e-4), name


def test_grad_checkpoint_is_a_noop_in_eval_mode():
    """Checkpointing only matters for backward; eval() must skip it (no grad to save)."""
    torch.manual_seed(0)
    model = build_improved_unet("dilation122436", base_channels=4, grad_checkpoint=True)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 1, 32, 32))
    assert torch.isfinite(out).all()


class _TinyDataset(torch.utils.data.Dataset):
    def __init__(self, n: int, size: int = 8):
        self.n, self.size = n, size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index):
        image = torch.randn(1, self.size, self.size)
        mask = torch.zeros(1, self.size, self.size)
        mask[0, 2:4, 2:4] = 1.0
        return {"image": image, "mask": mask, "stem": str(index)}


def _loader(n: int, batch_size: int) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(_TinyDataset(n), batch_size=batch_size)


def test_accumulation_matches_the_equivalent_single_step_update():
    """batch_size=2, accumulation_steps=2 should update weights the same as
    one batch_size=4 step over the same four examples."""
    torch.manual_seed(0)
    model_a = torch.nn.Conv2d(1, 1, kernel_size=1)
    torch.manual_seed(0)
    model_b = torch.nn.Conv2d(1, 1, kernel_size=1)
    assert torch.equal(model_a.weight, model_b.weight)

    cfg_a, cfg_b = Config(), Config()
    cfg_a.train.epochs = cfg_b.train.epochs = 1
    cfg_a.train.amp = cfg_b.train.amp = False
    cfg_a.train.grad_clip = cfg_b.train.grad_clip = 0  # isolate accumulation, not clipping
    cfg_a.train.batch_size, cfg_a.train.accumulation_steps = 4, 1
    cfg_b.train.batch_size, cfg_b.train.accumulation_steps = 2, 2

    trainer_a = Trainer(model_a, cfg_a, "/tmp", device="cpu")
    trainer_b = Trainer(model_b, cfg_b, "/tmp", device="cpu")

    torch.manual_seed(1311)
    data = [{"image": torch.randn(1, 4, 4), "mask": torch.zeros(1, 4, 4), "stem": str(i)}
            for i in range(4)]

    trainer_a._train_epoch(torch.utils.data.DataLoader(data, batch_size=4))
    trainer_b._train_epoch(torch.utils.data.DataLoader(data, batch_size=2))

    assert torch.allclose(model_a.weight, model_b.weight, atol=1e-6)
    assert torch.allclose(model_a.bias, model_b.bias, atol=1e-6)


def test_accumulation_flushes_a_trailing_partial_group():
    """5 batches with accumulation_steps=2 must still update on the odd one out."""
    cfg = Config()
    cfg.train.epochs = 1
    cfg.train.amp = False
    cfg.train.batch_size, cfg.train.accumulation_steps = 1, 2

    model = torch.nn.Conv2d(1, 1, kernel_size=1)
    before = model.weight.detach().clone()
    trainer = Trainer(model, cfg, "/tmp", device="cpu")
    trainer._train_epoch(_loader(5, batch_size=1))

    assert not torch.equal(before, model.weight)  # the trailing 5th batch's update landed
