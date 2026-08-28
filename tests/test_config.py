"""Config loading and CLI overrides."""

from filseg.config import load_config, parse_overrides


def test_defaults_and_overrides(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("name: demo\ntrain:\n  epochs: 3\n")

    cfg = load_config(path, parse_overrides(["train.batch_size=8", "model.encoder=resnet50"]))
    assert cfg.name == "demo"
    assert cfg.train.epochs == 3
    assert cfg.train.batch_size == 8
    assert cfg.model.encoder == "resnet50"


def test_load_checkpoint_does_not_request_pretrained_encoder_weights(tmp_path):
    """A checkpoint's own weights overwrite the encoder immediately after
    construction, so rebuilding with the training-time encoder_weights (e.g.
    'imagenet') would download pretrained weights only to discard them a line
    later. load_checkpoint must force encoder_weights=None regardless of what
    the checkpoint's stored config says.
    """
    from unittest.mock import patch

    from filseg.config import Config
    from filseg.models import build as build_mod
    from filseg.models.build import build_model, load_checkpoint, save_checkpoint

    cfg = Config()
    cfg.model.architecture = "unet"
    cfg.model.encoder = "resnet18"
    cfg.model.encoder_weights = "imagenet"  # what training actually used

    no_pretrain_cfg = Config()
    no_pretrain_cfg.model.architecture = "unet"
    no_pretrain_cfg.model.encoder = "resnet18"
    no_pretrain_cfg.model.encoder_weights = None
    model = build_model(no_pretrain_cfg.model)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, cfg)  # saved config still says "imagenet"

    calls = []
    original = build_mod._ARCHITECTURES["unet"]

    def spy(*args, **kwargs):
        calls.append(kwargs.get("encoder_weights"))
        return original(*args, **kwargs)

    with patch.dict(build_mod._ARCHITECTURES, {"unet": spy}):
        _, stored_cfg = load_checkpoint(path, device="cpu")

    assert stored_cfg["model"]["encoder_weights"] == "imagenet"  # unchanged on disk
    assert calls == [None]  # but never requested on reload
