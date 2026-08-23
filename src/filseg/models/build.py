"""Model construction and checkpoint I/O.

Architectures come from ``segmentation_models_pytorch`` so that swapping the
encoder or decoder is a config change rather than a code change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import segmentation_models_pytorch as smp
import torch
from torch import nn

from filseg.config import Config, ModelConfig

_ARCHITECTURES = {
    "unet": smp.Unet,
    "unetplusplus": smp.UnetPlusPlus,
    "fpn": smp.FPN,
    "deeplabv3plus": smp.DeepLabV3Plus,
    "manet": smp.MAnet,
}


def build_model(cfg: ModelConfig) -> nn.Module:
    """Instantiate a single-logit segmentation network."""
    try:
        factory = _ARCHITECTURES[cfg.architecture.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown architecture '{cfg.architecture}'. "
            f"Available: {sorted(_ARCHITECTURES)}"
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
