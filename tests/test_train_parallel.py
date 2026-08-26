"""The parallel launcher's pre-flight checks and GPU assignment."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_parallel  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_parallel.py"


def _write(tmp_path: Path, name: str) -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(f"name: {name}\ntrain: {{epochs: 1}}\n")
    return path


def test_config_name_reads_the_name_field(tmp_path):
    assert train_parallel.config_name(_write(tmp_path, "alpha")) == "alpha"


def test_config_name_falls_back_to_the_filename(tmp_path):
    path = tmp_path / "no_name.yaml"
    path.write_text("train: {epochs: 1}\n")
    assert train_parallel.config_name(path) == "no_name"


def test_configs_sharing_a_name_are_rejected_before_launching(tmp_path):
    """Two runs with the same name: would overwrite each other's checkpoints,
    silently, hours into a session. Fail immediately instead."""
    first = tmp_path / "a.yaml"
    second = tmp_path / "b.yaml"
    for path in (first, second):
        path.write_text("name: same\n")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(first), str(second)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "overwrite each other" in result.stdout + result.stderr


def test_missing_config_is_rejected_before_launching(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path / "nope.yaml")],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "No such config" in result.stdout + result.stderr
