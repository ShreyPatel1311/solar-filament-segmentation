"""Flat U-Net (Zhu et al. 2025) matches the paper's stated properties."""

import pytest
import torch

from filseg.config import ModelConfig
from filseg.models.build import build_model
from filseg.models.flat_unet import CSAConvBlock, FlatUNet, SCAConvBlock, build_flat_unet


@pytest.mark.parametrize("variant", ["flat-unet", "flat-unet-sca", "flat-unet-csa"])
def test_output_matches_input_resolution(variant):
    model = build_flat_unet(variant, channels=8, depth=2)
    out = model(torch.randn(2, 1, 32, 32))
    assert out.shape == (2, 1, 32, 32)


def test_channels_stay_flat_through_the_network():
    """The whole point of the paper: no channel doubling anywhere."""
    model = FlatUNet(channels=16, depth=3)
    widths = {
        m.out_channels for m in model.modules()
        if isinstance(m, torch.nn.Conv2d) and m.out_channels != 1  # excl. the 1-logit head
    }
    assert widths == {16}


def test_csa_hoisting_equals_the_papers_literal_double_sum():
    """CSAConvBlock computes sum_s(F_q^(c) . F_k^(s)^T) as F_q^(c) . (sum_s F_k^(s))^T.

    That is an exact algebraic rearrangement (matmul is linear in its right
    argument), turning O(C^2) matrix products into O(C). If it ever stops
    matching the literal form, the block is no longer the paper's operator.
    """
    torch.manual_seed(0)
    n, c, h, w = 2, 6, 12, 12
    query, key = torch.randn(n, c, h, w), torch.randn(n, c, h, w)

    literal = torch.zeros(n, c, h, h)
    for i in range(n):
        for channel in range(c):
            for other in range(c):
                literal[i, channel] += query[i, channel] @ key[i, other].T

    block = CSAConvBlock(c)
    assert torch.allclose(literal, block._similarity(query, key), atol=1e-4)


def test_sca_scores_each_channel_against_itself_only():
    torch.manual_seed(0)
    n, c, h, w = 1, 4, 8, 8
    query, key = torch.randn(n, c, h, w), torch.randn(n, c, h, w)

    expected = torch.stack([query[0, i] @ key[0, i].T for i in range(c)]).unsqueeze(0)
    assert torch.allclose(expected, SCAConvBlock(c)._similarity(query, key), atol=1e-5)


def test_attention_blocks_preserve_shape():
    """Section 2.2: 'the dimensions of the input X and output Y remain consistent'."""
    x = torch.randn(2, 8, 16, 16)
    for block in (SCAConvBlock(8), CSAConvBlock(8)):
        assert block(x).shape == x.shape


def test_is_ultralightweight_relative_to_a_classical_unet():
    """Table 2: 0.26M params at C=32, against the classical U-Net's 28.95M."""
    params = sum(p.numel() for p in build_flat_unet(channels=32, depth=4).parameters())
    assert params < 1_000_000


def test_build_model_dispatches_flat_unet():
    model = build_model(ModelConfig(architecture="flat-unet", encoder_weights=None))
    assert isinstance(model, FlatUNet)


def test_unknown_block_kind_is_rejected():
    with pytest.raises(ValueError, match="'sca' or 'csa'"):
        FlatUNet(encoder_blocks="nonsense")
