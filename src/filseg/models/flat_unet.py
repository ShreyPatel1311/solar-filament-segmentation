"""Flat U-Net from Zhu et al. (2025), arXiv:2502.07259.

    "Flat U-Net: An Efficient Ultralightweight Model for Solar Filament
    Segmentation in Full-disk H-alpha Images"

The idea: a classical U-Net doubles its channel count at every downsampling
step to compensate for lost spatial detail, which is where nearly all of its
28.95M parameters live. Flat U-Net keeps the channel count *flat* at ``C``
throughout, and recovers the lost interchannel expressiveness with an
attention block that reweights channels instead of adding more of them.
Result: Table 2 reports DSC 0.79 at C=32 with 0.26M parameters (0.98 MB)
against the classical U-Net's 0.69 DSC at 28.95M parameters.

Two block types, both mapping ``R^(CxHxW) -> R^(CxHxW)``:

* ``CSAConvBlock`` (Section 2.2) -- channel self-attention. Each channel's
  similarity score is computed against *every* other channel.
* ``SCAConvBlock`` (Section 2.3) -- "simplified alternative". Each channel is
  scored against itself only, which the authors introduce specifically to cut
  GPU memory during training.

The paper's recommended configuration, which ``flat-unet`` reproduces, is SCA
throughout the encoder/decoder with CSA at the bottleneck: Table 3 shows that
single bottleneck CSA lifting recall 0.6402 -> 0.6948 and DSC 0.7581 -> 0.7943
over an all-SCA network.

No pretrained weights exist for this architecture; it trains from random init.

Deviations from the paper are called out at each site. The authors released
code (doi:10.12149/101545) but it is not vendored here -- this is a from-scratch
implementation from the paper's equations, which are given in full.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class _ChannelAttention(nn.Module):
    """Shared machinery for the CSA and SCA blocks (Figures 2 and Section 2.3).

    Both variants follow the same five steps; they differ only in how the
    similarity tensor ``T`` is formed, which :meth:`_similarity` supplies.

        F_q, F_k, F_v = C'(X, W1), C'(X, W2), C'(X, W3)   (unbiased convs)
        T^(c)         = <variant-specific>  / sqrt(H)
        s_c           = mean over space of T^(c)          (global average pool)
        s~            = softmax(s)
        F~_v          = [s~_1 * F_v^(1), ..., s~_C * F_v^(C)]
        Y             = g(b(F~_v + X))                    (ReLU o BatchNorm)

    The paper writes the convolutions as ``C'(X, W)`` without stating a kernel
    size. 3x3 is used here rather than 1x1: every convolution in this network
    lives inside one of these blocks, so with 1x1 kernels the model would have
    no spatial receptive field at all beyond what pooling provides.
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        # "unbiased convolution operation" -- bias=False, per Section 2.2.
        self.query = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.key = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.value = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        # "b is used for normalizing the residual result (Ioffe & Szegedy 2015),
        # and g refers to the ReLU function (Nair & Hinton 2010)."
        self.norm = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)

    def _similarity(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query, key, value = self.query(x), self.key(x), self.value(x)

        similarity = self._similarity(query, key)
        # "the normalization factor, sqrt(H), is set to balance the scale
        # differences arising from features of different dimensions, thus
        # stabilizing the gradients. H denotes the height of the feature map,
        # and in the current work, H and W are equal."
        similarity = similarity / math.sqrt(x.shape[-2])

        # Global average pooling over the spatial dims -> one score per channel.
        scores = similarity.mean(dim=(-2, -1))
        weights = torch.softmax(scores, dim=1)

        reweighted = value * weights[:, :, None, None]
        return self.activation(self.norm(reweighted + x))


class CSAConvBlock(_ChannelAttention):
    """Channel self-attention block (Section 2.2, Figure 2).

    Each channel is scored against every other channel:

        T^(c) = sum_s ( F_q^(c) . (F_k^(s))^T ) / sqrt(H)

    Written literally that is ``C`` matrix products per channel, i.e. O(C^2)
    of them. Matrix multiplication is linear in its right argument, so the
    sum can be hoisted:

        sum_s ( F_q^(c) . (F_k^(s))^T ) == F_q^(c) . ( sum_s F_k^(s) )^T

    This computes the identical tensor with ``C`` matrix products instead of
    ``C^2`` -- an exact algebraic rearrangement, not an approximation, and the
    reason CSA is affordable here at more than just the bottleneck.
    """

    def _similarity(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        key_sum = key.sum(dim=1, keepdim=True)            # (N, 1, H, W)
        # (N, C, H, W) @ (N, 1, W, H) broadcasts to (N, C, H, H).
        return query @ key_sum.transpose(-2, -1)


class SCAConvBlock(_ChannelAttention):
    """Simplified channel attention block (Section 2.3).

        T^(c) = F_q^(c) . (F_k^(c))^T / sqrt(H)

    Each channel is scored against itself only. Introduced by the authors
    because CSA's "extensive matrix operations involved in computing
    interchannel correlations" strain limited GPU memory; it "effectively
    handles slightly lower hardware environments without changing the number
    of parameters".
    """

    def _similarity(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        return query @ key.transpose(-2, -1)              # (N, C, H, H)


def _make_block(kind: str, channels: int) -> _ChannelAttention:
    if kind == "csa":
        return CSAConvBlock(channels)
    if kind == "sca":
        return SCAConvBlock(channels)
    raise ValueError(f"block kind must be 'sca' or 'csa', got {kind!r}")


class FlatUNet(nn.Module):
    """U-Net whose channel count never changes, per Zhu et al. (2025).

    Args:
        in_channels: input image channels (1 for grayscale H-alpha).
        channels: the flat width ``C``. The paper's Table 1/2 sweep found the
            model only starts learning at C=9, and recommends C=32 as the
            accuracy/size knee (DSC 0.79 at 0.98 MB); C=128 buys DSC 0.81 for
            16x the parameters.
        depth: encoder layers, each halving the spatial resolution. The paper
            uses 4.
        encoder_blocks / decoder_blocks / bottleneck_block: ``"sca"`` or
            ``"csa"`` per position. Defaults reproduce the paper's recommended
            SCA-backbone/CSA-bottleneck configuration.
        grad_checkpoint: recompute block activations in backward instead of
            storing them. The attention blocks hold a (N, C, H, H) similarity
            tensor, which at full resolution is the model's memory high-water
            mark despite its tiny parameter count.
    """

    def __init__(self, in_channels: int = 1, channels: int = 32, depth: int = 4,
                 encoder_blocks: str = "sca", decoder_blocks: str = "sca",
                 bottleneck_block: str = "csa", grad_checkpoint: bool = False):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        self.grad_checkpoint = grad_checkpoint

        # "Initially, a classical convolution module is employed to expand the
        # single-channel solar image to C channels, and the number of channels
        # is maintained consistently at C throughout the network."
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.encoders = nn.ModuleList(
            _make_block(encoder_blocks, channels) for _ in range(depth)
        )
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _make_block(bottleneck_block, channels)

        # "The skip connections are identical to those found in the classical
        # U-Net." With flat channels, concatenating a skip gives 2C, so each
        # decoder step needs a fusion convolution back down to C before its
        # attention block -- the paper's figures do not label this explicitly.
        self.fuse = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            for _ in range(depth)
        )
        self.decoders = nn.ModuleList(
            _make_block(decoder_blocks, channels) for _ in range(depth)
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # Single logit per pixel, matching the rest of the pipeline (the loss
        # applies the sigmoid, not the model).
        self.head = nn.Conv2d(channels, 1, kernel_size=1)

    def _run(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.grad_checkpoint and self.training:
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        skips: list[torch.Tensor] = []
        for encoder in self.encoders:
            x = self._run(encoder, x)
            skips.append(x)
            x = self.pool(x)

        x = self._run(self.bottleneck, x)

        for fuse, decoder, skip in zip(self.fuse, self.decoders, reversed(skips), strict=True):
            x = self.upsample(x)
            x = fuse(torch.cat([x, skip], dim=1))
            x = self._run(decoder, x)

        return self.head(x)


#: Paper configurations, keyed by the names accepted in ``model.architecture``.
VARIANTS: dict[str, dict] = {
    # Recommended: SCA backbone, CSA bottleneck (Table 3).
    "flatunet": {"encoder_blocks": "sca", "decoder_blocks": "sca",
                 "bottleneck_block": "csa"},
    # Ablations from the same table.
    "flatunetsca": {"encoder_blocks": "sca", "decoder_blocks": "sca",
                    "bottleneck_block": "sca"},
    "flatunetcsa": {"encoder_blocks": "csa", "decoder_blocks": "csa",
                    "bottleneck_block": "csa"},
}


def build_flat_unet(variant: str = "flat-unet", in_channels: int = 1, **overrides) -> FlatUNet:
    """Instantiate a named Flat U-Net configuration."""
    key = variant.lower().replace("-", "").replace("_", "")
    if key not in VARIANTS:
        raise ValueError(
            f"Unknown Flat U-Net variant {variant!r}. Available: {sorted(VARIANTS)}"
        )
    kwargs = {**VARIANTS[key], "in_channels": in_channels}
    kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return FlatUNet(**kwargs)
