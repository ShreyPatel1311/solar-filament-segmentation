"""Liu et al. (2021) improved U-Nets: shape, spec fidelity, and config wiring."""

import pytest
import torch

from filseg.config import Config, ModelConfig
from filseg.models.build import build_model, load_checkpoint, save_checkpoint
from filseg.models.improved_unet import ASPP, VARIANTS, ImprovedUNet, build_improved_unet


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_output_matches_input_resolution(variant):
    """Same padding throughout -- Section 3.3.1's whole point is no cropping."""
    model = build_improved_unet(variant)
    out = model(torch.zeros(2, 1, 64, 64))
    assert out.shape == (2, 1, 64, 64)


def test_dilation122436_matches_paper_spec():
    """Figure 6: three stages, ASPP with a 1x1 branch plus rates 12, 24, 36."""
    model = build_improved_unet("dilation-122436")

    assert len(model.encoders) == 3
    rates = [
        m.dilation[0] for m in model.aspp.modules()
        if isinstance(m, torch.nn.Conv2d) and m.kernel_size == (3, 3)
    ]
    assert rates == [12, 24, 36]
    assert len(model.aspp.branches) == 4  # 1x1 + three dilated


def test_u4floor_has_no_aspp_and_four_poolings():
    model = build_improved_unet("u-4floor")
    assert isinstance(model.aspp, torch.nn.Identity)
    assert len(model.encoders) == 5  # five stages == four downsamplings


def test_dilation_u4floor_uses_smaller_rates():
    model = build_improved_unet("dilation-u4floor")
    rates = [
        m.dilation[0] for m in model.aspp.modules()
        if isinstance(m, torch.nn.Conv2d) and m.kernel_size == (3, 3)
    ]
    assert rates == [6, 12, 18]


def test_aspp_batchnorm_follows_every_dilated_convolution():
    """'a Batch Normalization algorithm is used after each dilated convolution'."""
    aspp = ASPP(8, 8, rates=(2, 4))
    for branch in aspp.branches:
        assert any(isinstance(m, torch.nn.BatchNorm2d) for m in branch)


def test_aspp_fusion_modes_agree_on_shape():
    x = torch.randn(1, 8, 16, 16)
    assert ASPP(8, 8, (2, 4), fusion="sum")(x).shape == (1, 8, 16, 16)
    assert ASPP(8, 8, (2, 4), fusion="concat")(x).shape == (1, 8, 16, 16)


def test_aspp_rejects_unknown_fusion():
    with pytest.raises(ValueError, match="fusion"):
        ASPP(8, 8, (2,), fusion="average")


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="Unknown improved U-Net variant"):
        build_improved_unet("dilation-999")


def test_dropout_lives_in_the_expansion_path_only():
    """Section 3.3.1 moves dropout out of the contraction path."""
    model = ImprovedUNet(stages=3, dropout=0.5)
    assert not any(
        isinstance(m, torch.nn.Dropout2d) for enc in model.encoders for m in enc.modules()
    )
    assert any(
        isinstance(m, torch.nn.Dropout2d) for dec in model.decoders for m in dec.modules()
    )


def test_build_model_dispatches_by_architecture_name():
    cfg = ModelConfig(architecture="dilation-122436", encoder_weights=None)
    assert isinstance(build_model(cfg), ImprovedUNet)


def test_build_model_ignores_pretrained_weights_with_a_warning(caplog):
    cfg = ModelConfig(architecture="dilation-122436", encoder_weights="imagenet")
    with caplog.at_level("WARNING"):
        build_model(cfg)
    assert "no pretrained weights" in caplog.text


def test_checkpoint_round_trip_rebuilds_the_same_architecture(tmp_path):
    """predict.py rebuilds from the stored config, so it must survive a save."""
    cfg = Config()
    cfg.model = ModelConfig(architecture="dilation-122436", encoder_weights=None)
    model = build_model(cfg.model)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, cfg)
    restored, stored = load_checkpoint(path)

    assert isinstance(restored, ImprovedUNet)
    assert stored["model"]["architecture"] == "dilation-122436"
    for a, b in zip(model.state_dict().values(), restored.state_dict().values(), strict=True):
        assert torch.equal(a, b)


def test_dropout_uses_exactly_the_papers_two_values_not_a_halving_schedule():
    """Every figure's legend shows only 0.5 and 0.2 -- deepest step gets the
    former, every shallower step (any depth) gets the latter, unchanged."""
    model = ImprovedUNet(stages=5, dropout=0.5, dropout_shallow=0.2)  # u-4floor depth
    rates = [
        m.p for dec in model.decoders for m in dec.modules()
        if isinstance(m, torch.nn.Dropout2d)
    ]
    assert rates == [0.5, 0.2, 0.2, 0.2]
