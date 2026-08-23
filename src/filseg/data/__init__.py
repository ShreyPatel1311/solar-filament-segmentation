"""Dataset, annotations and augmentation.

Submodules are exposed lazily: reading annotations or RLE should not require the
augmentation stack, which matters when scoring a submission in a bare kernel.
"""

_EXPORTS = {
    "FilamentDataset": "filseg.data.dataset",
    "build_dataloaders": "filseg.data.dataset",
    "load_split": "filseg.data.dataset",
    "MagfiloAnnotations": "filseg.data.coco",
    "decode_rle": "filseg.data.coco",
    "encode_rle": "filseg.data.coco",
    "train_transforms": "filseg.data.transforms",
    "val_transforms": "filseg.data.transforms",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        import importlib

        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
