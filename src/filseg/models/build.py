"""Model construction and checkpoint I/O.

Two families are available, both selected by ``model.architecture``:

* ``segmentation_models_pytorch`` backbones (``unet``, ``fpn``, ...), which
  accept a pretrained ImageNet encoder;
* the Liu et al. (2021) improved U-Nets (``dilation-122436``, ``u-4floor``,
  ``dilation-u4floor``), which have no pretrained weights and train from
  random init -- see :mod:`filseg.models.improved_unet`;
* the Zhu et al. (2025) Flat U-Nets (``flat-unet``, ``flat-unet-sca``,
  ``flat-unet-csa``) -- see :mod:`filseg.models.flat_unet`;
* the plain Ronneberger U-Net used by Diercke et al. (2024)
  (``diercke-unet``) -- see :mod:`filseg.models.vanilla_unet`.

The last three families all train from random init and ignore
``encoder``/``encoder_weights``.

Swapping between them is a config change rather than a code change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import segmentation_models_pytorch as smp
import torch
from torch import nn

from filseg.config import Config, ModelConfig
from filseg.models.flat_unet import VARIANTS as _FLAT_VARIANTS
from filseg.models.flat_unet import build_flat_unet
from filseg.models.improved_unet import VARIANTS as _IMPROVED_VARIANTS
from filseg.models.improved_unet import build_improved_unet
from filseg.models.vanilla_unet import VanillaUNet
from filseg.utils.logging import get_logger

logger = get_logger(__name__)

_ARCHITECTURES = {
    "unet": smp.Unet,
    "unetplusplus": smp.UnetPlusPlus,
    "fpn": smp.FPN,
    "deeplabv3plus": smp.DeepLabV3Plus,
    "manet": smp.MAnet,
}


def _normalize(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "")


def _warn_if_pretrained(cfg: ModelConfig) -> None:
    """Silently ignoring encoder_weights would look like weights were loaded."""
    if cfg.encoder_weights:
        logger.warning(
            "model.encoder_weights=%r is ignored by %s: it has no pretrained "
            "weights and trains from random init.",
            cfg.encoder_weights, cfg.architecture,
        )


def build_model(cfg: ModelConfig) -> nn.Module:
    """Instantiate a single-logit segmentation network."""
    key = _normalize(cfg.architecture)

    if key in _FLAT_VARIANTS:
        _warn_if_pretrained(cfg)
        return build_flat_unet(
            key,
            in_channels=cfg.in_channels,
            channels=cfg.channels,
            depth=cfg.depth,
            encoder_blocks=cfg.encoder_blocks,
            decoder_blocks=cfg.decoder_blocks,
            bottleneck_block=cfg.bottleneck_block,
            grad_checkpoint=cfg.grad_checkpoint,
        )

    if key == "dierckeunet":
        _warn_if_pretrained(cfg)
        kwargs = dict(
            in_channels=cfg.in_channels,
            base_channels=cfg.base_channels,
            norm=cfg.norm,
            dropout=cfg.dropout,
            grad_checkpoint=cfg.grad_checkpoint,
        )
        if cfg.depth is not None:
            kwargs["depth"] = cfg.depth
        return VanillaUNet(**kwargs)

    if key in _IMPROVED_VARIANTS:
        _warn_if_pretrained(cfg)
        return build_improved_unet(
            key,
            in_channels=cfg.in_channels,
            stages=cfg.stages,
            base_channels=cfg.base_channels,
            aspp_rates=tuple(cfg.aspp_rates) if cfg.aspp_rates else None,
            aspp_fusion=cfg.aspp_fusion,
            dropout=cfg.dropout,
            dropout_shallow=cfg.dropout_shallow,
            norm=cfg.norm,
            grad_checkpoint=cfg.grad_checkpoint,
        )

    try:
        factory = _ARCHITECTURES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown architecture '{cfg.architecture}'. Available: "
            f"{sorted({*_ARCHITECTURES, *_IMPROVED_VARIANTS, *_FLAT_VARIANTS, 'diercke-unet'})}"
        ) from exc

    return factory(
        encoder_name=cfg.encoder,
        encoder_weights=cfg.encoder_weights,
        in_channels=cfg.in_channels,
        classes=1,
    )


def save_checkpoint(path: str | Path, model: nn.Module, cfg: Config,
                    extra: dict[str, Any] | None = None) -> None:
    """Persist weights together with the config that produced them."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "config": cfg.to_dict(), **(extra or {})}, path
    )


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"
                    ) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild the model described by a checkpoint and load its weights."""
    payload = torch.load(str(path), map_location=device, weights_only=False)
    model_cfg = ModelConfig(**payload["config"]["model"])
    model = build_model(model_cfg)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload["config"]
