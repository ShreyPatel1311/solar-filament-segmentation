"""Model construction and checkpoint I/O.

Two families are available, both selected by ``model.architecture``:

* ``segmentation_models_pytorch`` backbones (``unet``, ``fpn``, ...), which
  accept a pretrained ImageNet encoder;
* the Liu et al. (2021) improved U-Nets (``dilation-122436``, ``u-4floor``,
  ``dilation-u4floor``), which have no pretrained weights and train from
  random init -- see :mod:`filseg.models.improved_unet`.

Swapping between them is a config change rather than a code change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import segmentation_models_pytorch as smp
import torch
from torch import nn

from filseg.config import Config, ModelConfig
from filseg.models.improved_unet import VARIANTS as _IMPROVED_VARIANTS
from filseg.models.improved_unet import build_improved_unet
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


def build_model(cfg: ModelConfig) -> nn.Module:
    """Instantiate a single-logit segmentation network."""
    key = _normalize(cfg.architecture)

    if key in _IMPROVED_VARIANTS:
        if cfg.encoder_weights:
            # Silently ignoring this would look like pretrained weights loaded.
            logger.warning(
                "model.encoder_weights=%r is ignored by %s: the improved U-Nets have "
                "no pretrained weights and train from random init.",
                cfg.encoder_weights, cfg.architecture,
            )
        return build_improved_unet(
            key,
            in_channels=cfg.in_channels,
            stages=cfg.stages,
            base_channels=cfg.base_channels,
            aspp_rates=tuple(cfg.aspp_rates) if cfg.aspp_rates else None,
            aspp_fusion=cfg.aspp_fusion,
            dropout=cfg.dropout,
            norm=cfg.norm,
        )

    try:
        factory = _ARCHITECTURES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown architecture '{cfg.architecture}'. Available: "
            f"{sorted(_ARCHITECTURES) + sorted(_IMPROVED_VARIANTS)}"
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
