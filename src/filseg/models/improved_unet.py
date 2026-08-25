"""Improved U-Nets from Liu et al. (2021), Solar Physics 296:176.

    "Solar Filament Segmentation Based on Improved U-Nets"
    https://doi.org/10.1007/s11207-021-01920-3

The authors released no code (the paper carries a data link but no code
availability statement; the GF-Zhu/Filament-Unet repository implements the
*earlier* Zhu et al. 2019 Enhanced U-Net, which this paper uses only as a
comparison baseline). So this is a from-scratch PyTorch reimplementation from
the paper's Table 1 and Figures 2/6/7, trained from random init on MAGFiLO --
no pretrained weights are involved anywhere in this module.

One parameterized class covers three of the paper's seven variants:

    variant           stages  ASPP rates      Table 1 description
    ----------------  ------  --------------  -----------------------------
    u-4floor               5  none            same padding, dropout moved to
                                              the expansion path
    dilation-122436        3  1, 12, 24, 36   "add ASPP module, the number of
                                              layers is reduced to 3"
    dilation-u4floor       3  1, 6, 12, 18    as above, smaller rates

"stages" counts encoder convolution stages, so u-4floor's five stages give the
four poolings its name refers to, and the ASPP variants' three stages give two.
The rate 1 branch is the ASPP module's 1x1 convolution.

Deviations from the paper are called out at each site; the paper specifies the
architecture through figures rather than equations, so a few details (exact
dropout placement, ASPP fusion rule) are reasoned choices rather than
transcription. Each is exposed as a config field so it can be ablated.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class ConvBlock(nn.Module):
    """The paper's two stacked 3x3 convolutions, with *same* padding.

    Section 3.3.1 is explicit that all seven improved networks abandon U-Net's
    valid padding, precisely so the skip connections need no cropping and the
    output keeps the input's size. Normalization is off by default because the
    paper only mentions BatchNorm inside the ASPP module; enable it via
    ``model.norm`` when training from scratch proves unstable.
    """

    def __init__(self, in_channels: int, out_channels: int, norm: bool = False,
                 grad_checkpoint: bool = False):
        super().__init__()
        layers: list[nn.Module] = []
        for channels in (in_channels, out_channels):
            layers.append(nn.Conv2d(channels, out_channels, kernel_size=3, padding=1,
                                    bias=not norm))
            if norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)
        self.grad_checkpoint = grad_checkpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.grad_checkpoint and self.training:
            return checkpoint(self.block, x, use_reentrant=False)
        return self.block(x)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling, as described in Section 3.2/3.3.3.

    Parallel branches -- one 1x1 convolution plus one 3x3 dilated convolution
    per rate -- widen the receptive field without downsampling, which is the
    whole point for filaments: barbs are thin, so every pooling step that buys
    context also destroys the detail being segmented.

    "In the ASPP module, a Batch Normalization algorithm is used after each
    dilated convolution to avoid the phenomenon of gradient disappearance and
    accelerate the network's convergence speed."

    Fusion defaults to ``sum``, following the DeepLab V2 module the paper cites
    (V2 sums its branches; V3 concatenates and projects). ``concat`` is offered
    because it keeps each branch's response separable and often scores better.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 rates: tuple[int, ...] = (12, 24, 36), fusion: str = "sum",
                 grad_checkpoint: bool = False):
        super().__init__()
        if fusion not in {"sum", "concat"}:
            raise ValueError(f"ASPP fusion must be 'sum' or 'concat', got {fusion!r}")
        self.fusion = fusion
        self.grad_checkpoint = grad_checkpoint

        branches: list[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        ]
        for rate in rates:
            branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate,
                              dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.branches = nn.ModuleList(branches)

        self.project = (
            nn.Sequential(
                nn.Conv2d(out_channels * len(branches), out_channels, kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            if fusion == "concat"
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.grad_checkpoint and self.training:
            outputs = [checkpoint(branch, x, use_reentrant=False) for branch in self.branches]
        else:
            outputs = [branch(x) for branch in self.branches]
        if self.fusion == "concat":
            return self.project(torch.cat(outputs, dim=1))
        return torch.stack(outputs, dim=0).sum(dim=0)


class UpBlock(nn.Module):
    """Expansion-path step: upsample, 2x2 convolution, concatenate skip, convolve.

    Figure 6's legend pairs "Up Sampling 2x2" with a separate "Convolution 2x2"
    rather than a transposed convolution -- Zhu et al., whose design this builds
    on, replaced deconvolution with nearest-neighbour interpolation, and the
    checkerboard artifacts deconvolution produces would be indistinguishable
    from the small-scale noise this paper is trying to remove.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 dropout: float = 0.0, norm: bool = False, grad_checkpoint: bool = False):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, padding="same"),
            nn.ReLU(inplace=True),
        )
        self.block = ConvBlock(out_channels + skip_channels, out_channels, norm=norm,
                               grad_checkpoint=grad_checkpoint)
        # Section 3.3.1: dropout sits in the expansion path, not the contraction
        # path -- "dropout operation in the upsampling stage can effectively
        # avoid overfitting".
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.dropout(self.block(x))


class ImprovedUNet(nn.Module):
    """U-Net variant with same padding, expansion-path dropout and optional ASPP.

    Emits raw logits (no sigmoid), matching the rest of the pipeline: the loss
    is BCEWithLogits + soft Dice, and inference applies the sigmoid itself.
    The paper's own loss is plain binary cross-entropy (Equation 4); Dice is
    retained here because filaments cover well under 1% of a MAGFiLO frame.

    Args:
        in_channels: input image channels (1 for grayscale H-alpha).
        stages: encoder convolution stages. 3 reproduces dilation-122436,
            5 reproduces u-4floor.
        base_channels: channels at the first stage; doubles each stage.
        aspp_rates: dilation rates for the ASPP module at the deepest stage.
            ``None`` or empty disables ASPP entirely (giving u-4floor). The
            paper's naming counts the 1x1 branch as "1", so ``(12, 24, 36)``
            is dilation-122436.
        aspp_fusion: ``sum`` (DeepLab V2, the paper's citation) or ``concat``.
        dropout: dropout at the deepest expansion step. Every figure's legend
            (2, 6, 7, ...) shows exactly two dropout values, 0.5 and 0.2 --
            not a progressively halved schedule -- so every shallower step
            uses ``dropout_shallow`` instead, unchanged with depth.
        dropout_shallow: dropout at every expansion step except the deepest.
        norm: BatchNorm inside the U-Net body (always on inside ASPP).
        grad_checkpoint: recompute activations in backward instead of storing
            them. Matters most for the ASPP variants -- the parallel dilated
            branches at the deepest, largest-spatial-resolution stage (only
            ``stages - 1`` poolings happen before ASPP) each hold a full
            feature map at once, which is the dominant memory cost at 1024px.
    """

    def __init__(self, in_channels: int = 1, stages: int = 3, base_channels: int = 64,
                 aspp_rates: tuple[int, ...] | None = (12, 24, 36),
                 aspp_fusion: str = "sum", dropout: float = 0.5, dropout_shallow: float = 0.2,
                 norm: bool = False, grad_checkpoint: bool = False):
        super().__init__()
        if stages < 2:
            raise ValueError(f"stages must be at least 2, got {stages}")

        channels = [base_channels * 2**i for i in range(stages)]

        self.encoders = nn.ModuleList()
        previous = in_channels
        for width in channels:
            self.encoders.append(ConvBlock(previous, width, norm=norm,
                                           grad_checkpoint=grad_checkpoint))
            previous = width
        self.pool = nn.MaxPool2d(2)

        self.aspp = (
            ASPP(channels[-1], channels[-1], tuple(aspp_rates), fusion=aspp_fusion,
                grad_checkpoint=grad_checkpoint)
            if aspp_rates
            else nn.Identity()
        )

        self.decoders = nn.ModuleList()
        for level, index in enumerate(range(stages - 1, 0, -1)):
            # Every figure's legend shows exactly two dropout values (0.5 and
            # 0.2), not a schedule -- the deepest expansion step gets the
            # higher one, every shallower step (which carries finer filament
            # detail) gets the lower one, unchanged with depth.
            level_dropout = dropout if level == 0 else dropout_shallow
            self.decoders.append(
                UpBlock(channels[index], channels[index - 1], channels[index - 1],
                        dropout=level_dropout, norm=norm, grad_checkpoint=grad_checkpoint)
            )

        # Figure 2/6 end with a narrow 2-channel 3x3 convolution before the 1x1
        # scoring layer, which the original U-Net used for its 2-class softmax.
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for index, encoder in enumerate(self.encoders):
            if index > 0:
                x = self.pool(x)
            x = encoder(x)
            skips.append(x)

        x = self.aspp(skips[-1])

        for decoder, skip in zip(self.decoders, reversed(skips[:-1]), strict=True):
            x = decoder(x, skip)

        return self.head(x)


#: Paper variant -> constructor keywords, keyed by the names used in Table 1.
VARIANTS: dict[str, dict] = {
    "dilation122436": {"stages": 3, "aspp_rates": (12, 24, 36)},
    "dilationu4floor": {"stages": 3, "aspp_rates": (6, 12, 18)},
    "u4floor": {"stages": 5, "aspp_rates": None},
}


def build_improved_unet(variant: str, in_channels: int = 1, **overrides) -> ImprovedUNet:
    """Instantiate a named variant, e.g. ``build_improved_unet("dilation122436")``."""
    key = variant.lower().replace("-", "").replace("_", "")
    if key not in VARIANTS:
        raise ValueError(
            f"Unknown improved U-Net variant {variant!r}. Available: {sorted(VARIANTS)}"
        )
    kwargs = {**VARIANTS[key], "in_channels": in_channels}
    kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return ImprovedUNet(**kwargs)
