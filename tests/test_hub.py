"""HF Hub sync: always the same filename, exercised with the network mocked out."""

from unittest.mock import MagicMock

from filseg import hub


def test_upload_checkpoint_overwrites_a_single_filename(monkeypatch, tmp_path):
    checkpoint = tmp_path / "unet_r34_best.pt"
    checkpoint.write_bytes(b"weights")

    fake_api = MagicMock()
    fake_api.upload_file.return_value = "https://huggingface.co/me/repo/blob/main/best_model.pt"
    fake_cls = MagicMock(return_value=fake_api)
    monkeypatch.setattr("huggingface_hub.HfApi", fake_cls)

    url = hub.upload_checkpoint(checkpoint, "me/repo", token="tok")

    fake_cls.assert_called_once_with(token="tok")
    fake_api.create_repo.assert_called_once_with(
        "me/repo", repo_type="model", exist_ok=True, private=False
    )
    _, kwargs = fake_api.upload_file.call_args
    assert kwargs["path_in_repo"] == hub.DEFAULT_FILENAME
    assert kwargs["repo_id"] == "me/repo"
    assert url.endswith("best_model.pt")


def test_upload_checkpoint_reads_token_from_environment(monkeypatch, tmp_path):
    checkpoint = tmp_path / "x.pt"
    checkpoint.write_bytes(b"weights")
    monkeypatch.setenv("HF_TOKEN", "env-token")

    fake_api = MagicMock()
    fake_cls = MagicMock(return_value=fake_api)
    monkeypatch.setattr("huggingface_hub.HfApi", fake_cls)

    hub.upload_checkpoint(checkpoint, "me/repo")

    fake_cls.assert_called_once_with(token="env-token")


def test_download_checkpoint_returns_local_path(monkeypatch, tmp_path):
    destination = tmp_path / "best_model.pt"
    destination.write_bytes(b"weights")

    fake_download = MagicMock(return_value=str(destination))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)

    path = hub.download_checkpoint("me/repo", local_dir=tmp_path)

    assert path == destination
    _, kwargs = fake_download.call_args
    assert kwargs["repo_id"] == "me/repo"
    assert kwargs["filename"] == hub.DEFAULT_FILENAME
