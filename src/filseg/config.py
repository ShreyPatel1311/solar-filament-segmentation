"""Typed experiment configuration loaded from YAML.

A config file is the single source of truth for a run: the CLI only chooses
which YAML to load and which fields to override, so a Kaggle run and a local
run of the same commit + config are identical.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    image_size: int = 1024
    val_fraction: float = 0.15
    split_seed: int = 1311
    num_workers: int = 2
    limbo_margin: float = 0.98  # fraction of the solar radius kept in the disk mask


@dataclass
class ModelConfig:
    architecture: str = "unet"
    encoder: str = "resnet34"
    encoder_weights: str | None = "imagenet"
    in_channels: int = 1


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-4
    amp: bool = True
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    grad_clip: float = 1.0
    seed: int = 1311


@dataclass
class PostprocessConfig:
    threshold: float = 0.5
    min_area: int = 120           # drop specks below this many pixels
    closing_radius: int = 3       # bridge fragmented filament bodies
    dilate_radius: int = 0        # optional final growth of each instance
    max_instances: int = 100


@dataclass
class Config:
    name: str = "baseline"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls, payload: dict[str, Any]):
    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    for key, value in payload.items():
        if key not in known:
            raise KeyError(f"Unknown config key '{key}' for {cls.__name__}")
        ftype = known[key].type
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[key] = _build(ftype, value)
        elif isinstance(value, dict) and key in {"data", "model", "train", "postprocess"}:
            kwargs[key] = _build(known[key].default_factory(), value)  # type: ignore[misc]
        else:
            kwargs[key] = value
    return cls(**kwargs)


_SECTIONS = {
    "data": DataConfig,
    "model": ModelConfig,
    "train": TrainConfig,
    "postprocess": PostprocessConfig,
}


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config, then apply ``section.field=value`` overrides."""
    payload: dict[str, Any] = {}
    if path is not None:
        payload = yaml.safe_load(Path(path).read_text()) or {}

    cfg = Config(name=payload.get("name", "baseline"))
    for section, cls in _SECTIONS.items():
        setattr(cfg, section, _build(cls, payload.get(section, {}) or {}))

    return _apply_overrides(cfg, overrides)


def _apply_overrides(cfg: Config, overrides: dict[str, Any] | None) -> Config:
    for dotted, value in (overrides or {}).items():
        section, _, key = dotted.partition(".")
        if not key:
            setattr(cfg, section, value)
            continue
        target = getattr(cfg, section)
        if not hasattr(target, key):
            raise KeyError(f"Unknown override '{dotted}'")
        current = getattr(target, key)
        setattr(target, key, type(current)(value) if current is not None else value)
    return cfg


def config_from_dict(payload: dict[str, Any], overrides: dict[str, Any] | None = None) -> Config:
    """Rebuild a config from a checkpoint's stored dict, then apply overrides."""
    cfg = Config(name=payload.get("name", "baseline"))
    for section, cls in _SECTIONS.items():
        setattr(cfg, section, _build(cls, payload.get(section, {}) or {}))
    return _apply_overrides(cfg, overrides)


def parse_overrides(items: list[str] | None) -> dict[str, Any]:
    """Turn ``["train.epochs=5", "model.encoder=resnet50"]`` into a dict."""
    out: dict[str, Any] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Override '{item}' is not of the form section.key=value")
        out[key.strip()] = yaml.safe_load(value)
    return out
