"""The Diercke et al. (2024) segmentation U-Net."""

import torch

from filseg.config import ModelConfig
from filseg.models.build import build_model
from filseg.models.vanilla_unet import VanillaUNet


def test_has_the_papers_twenty_three_convolutional_layers():
    """'U-net is a fully convolutional network with 23 convolutional layers.'"""
    model = VanillaUNet(base_channels=64, depth=5)
    convs = sum(
        1 for m in model.modules()
        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d))
    )
    assert convs == 23


def test_output_matches_input_resolution():
    model = VanillaUNet(base_channels=8, depth=3)
    assert model(torch.randn(2, 1, 32, 32)).shape == (2, 1, 32, 32)


def test_channels_double_each_stage_unlike_flat_unet():
    model = VanillaUNet(base_channels=8, depth=4)
    encoder_widths = [enc.block[0].out_channels for enc in model.encoders]
    assert encoder_widths == [8, 16, 32, 64]


def test_build_model_dispatches_diercke_unet():
    model = build_model(ModelConfig(architecture="diercke-unet", encoder_weights=None))
    assert isinstance(model, VanillaUNet)


def test_is_distinct_from_the_pretrained_resnet34_unet():
    """configs/unet_resnet34.yaml is a different network, not a rename."""
    diercke = build_model(ModelConfig(architecture="diercke-unet", encoder_weights=None))
    smp_unet = build_model(ModelConfig(architecture="unet", encoder_weights=None))
    assert type(diercke) is not type(smp_unet)
