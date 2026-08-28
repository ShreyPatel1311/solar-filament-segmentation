"""AffinityDiceBCELoss: semantic term unchanged, affinity term masked correctly."""

import torch

from filseg.engine.losses import AffinityDiceBCELoss, DiceBCELoss


def test_matches_plain_dice_bce_when_no_affinity_given():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 8, 8)  # 3 channels: semantic + 2 affinity
    target = (torch.rand(2, 1, 8, 8) > 0.9).float()

    combined = AffinityDiceBCELoss(bce_weight=0.5, dice_weight=0.5)
    semantic_only = DiceBCELoss(bce_weight=0.5, dice_weight=0.5)

    assert torch.allclose(combined(logits, target), semantic_only(logits[:, :1], target))


def test_affinity_term_only_counts_valid_pairs():
    torch.manual_seed(0)
    logits = torch.zeros(1, 3, 4, 4, requires_grad=True)
    target = torch.zeros(1, 1, 4, 4)

    affinity = torch.zeros(1, 2, 4, 4)
    valid = torch.zeros(1, 2, 4, 4)
    valid[0, 0, 0, 0] = 1  # exactly one valid pair, everything else masked out
    affinity[0, 0, 0, 0] = 1

    loss = AffinityDiceBCELoss(bce_weight=0.0, dice_weight=0.0, affinity_weight=1.0)
    value = loss(logits, target, affinity=affinity, affinity_valid=valid)

    # BCEWithLogits(0, target=1) = -log(sigmoid(0)) = log(2)
    assert torch.allclose(value, torch.log(torch.tensor(2.0)), atol=1e-5)


def test_all_invalid_pairs_gives_zero_affinity_contribution():
    logits = torch.randn(1, 3, 4, 4, requires_grad=True)
    target = torch.zeros(1, 1, 4, 4)
    affinity = torch.zeros(1, 2, 4, 4)
    valid = torch.zeros(1, 2, 4, 4)  # nothing valid (e.g. an all-background frame)

    loss = AffinityDiceBCELoss(bce_weight=0.0, dice_weight=0.0, affinity_weight=1.0)
    value = loss(logits, target, affinity=affinity, affinity_valid=valid)
    assert torch.allclose(value, torch.tensor(0.0), atol=1e-6)


def test_gradient_flows_through_both_terms():
    logits = torch.randn(1, 3, 4, 4, requires_grad=True)
    target = torch.ones(1, 1, 4, 4)
    affinity = torch.ones(1, 2, 4, 4)
    valid = torch.ones(1, 2, 4, 4)

    loss = AffinityDiceBCELoss()(logits, target, affinity=affinity, affinity_valid=valid)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert (logits.grad[:, 1:3] != 0).any()  # affinity channels actually got gradient
