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
