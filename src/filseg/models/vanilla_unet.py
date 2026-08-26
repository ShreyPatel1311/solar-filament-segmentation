"""The U-Net used by Diercke et al. (2024), arXiv:2402.15407.

    "A Universal Method for Solar Filament Detection from H-alpha
    Observations using Semi-supervised Deep Learning"

Scope note, because this paper's headline contribution is *not* an
architecture. Diercke et al. train YOLOv5 on manually labelled ChroTel
filtergrams, run it across the unlabelled GONG archive to generate
bounding boxes, threshold inside those boxes to synthesise pixel-wise masks,
and train a U-Net on the resulting 18,472 noisy auto-labels. The segmentation
network itself is stated to be plain Ronneberger et al. (2015): "we build on a
u-net architecture ... U-net is a fully convolutional network with 23
convolutional layers."

That semi-supervised label-generation pipeline is not reproducible here, and
would not help if it were:

* it needs ChroTel manual labels, i.e. external ground-truth data, which the
  competition's rules exclude ("the models may not use any other ground-truth
  metadata for training");
* its entire purpose is to manufacture labels where none exist. MAGFiLO
  already provides expert human annotations, which are strictly better than
  the thresholded pseudo-labels it would produce.

So what is reproduced here is the segmentation half: a plain U-Net trained
from random init, which is also a genuinely useful baseline to have. It is
deliberately *not* the same as ``model.architecture: unet``, which is
segmentation_models_pytorch's U-Net with a pretrained ImageNet ResNet-34
encoder -- a different network with different weights and 10x the parameters.

The paper's transferable preprocessing (limb-darkening correction, clipping to
[0.8, 1.3], normalising to [-1, 1]) lives in
:func:`filseg.data.transforms.diercke_normalize`, since it is a data concern
rather than a model one.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class DoubleConv(nn.Module):
    """The U-Net contracting unit: two 3x3 convolutions, each followed by ReLU.

    "The contracting path contains a series of convolutional layers, each
    followed by a rectified linear unit (ReLU), which is used to downsample
    the input maps."

    Ronneberger's original uses valid padding and no normalization. Same
    padding is used here so skip connections need no cropping and the output
    keeps the input's size -- the competition scores masks at the input
    resolution, so a shrinking output would need resampling before RLE
    encoding. BatchNorm is optional (``norm``) and off by default to stay
    faithful; it is worth enabling if training from scratch proves unstable,
    exactly as it was for the Liu et al. improved U-Nets.
    """

    def __init__(self, in_channels: int, out_channels: int, norm: bool = False):
        super().__init__()
        layers: list[nn.Module] = []
        for channels in (in_channels, out_channels):
            layers.append(nn.Conv2d(channels, out_channels, kernel_size=3, padding=1,
                                    bias=not norm))
            if norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VanillaUNet(nn.Module):
    """Ronneberger et al. (2015) U-Net, trained from random init.

    "In the expansive path, the process is reversed and with each step of the
    alternating convolutional layer and the ReLU, the maps are upsampled by a
    2x2 convolution and the number of feature channels is halved. This is a
    concatenation with the feature maps from the contracting path."

    Args:
        in_channels: input image channels (1 for grayscale H-alpha).
        base_channels: channels after the first stage; doubles each stage,
            giving the classical 64/128/256/512/1024 ladder at the default.
        depth: encoder stages including the bottleneck. 5 is the original.
        norm: add BatchNorm after each convolution (not in the original).
        dropout: dropout before the two deepest decoder steps. The original
            has none; Zhu et al. (2019), whose filament U-Net this lineage
            descends from, added it to curb overfitting.
        grad_checkpoint: recompute activations in backward instead of storing
            them, for memory-constrained training.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 64, depth: int = 5,
                 norm: bool = False, dropout: float = 0.0,
                 grad_checkpoint: bool = False):
        super().__init__()
        if depth < 2:
            raise ValueError(f"depth must be at least 2, got {depth}")
        self.grad_checkpoint = grad_checkpoint

        channels = [base_channels * 2**i for i in range(depth)]

        self.encoders = nn.ModuleList()
        previous = in_channels
        for width in channels:
            self.encoders.append(DoubleConv(previous, width, norm=norm))
            previous = width
        self.pool = nn.MaxPool2d(2)

        # "the maps are upsampled by a 2x2 convolution" -- a 2x2 transposed
        # convolution, which is Ronneberger's up-convolution.
        self.ups = nn.ModuleList(
            nn.ConvTranspose2d(channels[i], channels[i - 1], kernel_size=2, stride=2)
            for i in range(depth - 1, 0, -1)
        )
        self.decoders = nn.ModuleList(
            DoubleConv(channels[i - 1] * 2, channels[i - 1], norm=norm)
            for i in range(depth - 1, 0, -1)
        )
        self.dropouts = nn.ModuleList(
            nn.Dropout2d(dropout) if dropout > 0 and level < 2 else nn.Identity()
            for level in range(depth - 1)
        )

        self.head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def _run(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.grad_checkpoint and self.training:
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for index, encoder in enumerate(self.encoders):
            if index > 0:
                x = self.pool(x)
            x = self._run(encoder, x)
            skips.append(x)

        x = skips[-1]
        for up, decoder, dropout, skip in zip(
            self.ups, self.decoders, self.dropouts, reversed(skips[:-1]), strict=True
        ):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = dropout(self._run(decoder, x))

        return self.head(x)
