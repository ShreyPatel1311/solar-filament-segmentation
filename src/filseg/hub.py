"""Optional Hugging Face Hub sync for the training checkpoint.

Training always uploads to the same filename in the target repo, so the repo
holds exactly one live checkpoint: a new training run's best overwrites the
previous one (Hub keeps that history in git, but nothing accumulates in the
repo's file tree the way per-run filenames would).
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_FILENAME = "best_model.pt"


def _token(token: str | None) -> str | None:
    return token or os.environ.get("HF_TOKEN")


def upload_checkpoint(local_path: str | Path, repo_id: str,
                      filename: str = DEFAULT_FILENAME, token: str | None = None,
                      private: bool = False) -> str:
    """Push a checkpoint to ``repo_id/filename``, overwriting whatever is there.

    Returns the resulting file's URL.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=_token(token))
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=private)
    return api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"best checkpoint ({Path(local_path).name})",
    )


def download_checkpoint(repo_id: str, filename: str = DEFAULT_FILENAME,
                        local_dir: str | Path | None = None,
                        token: str | None = None) -> Path:
    """Fetch ``repo_id/filename`` into ``local_dir``, returning the local path."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo_id, filename=filename, repo_type="model",
        local_dir=str(local_dir) if local_dir else None, token=_token(token),
    )
    return Path(path)
