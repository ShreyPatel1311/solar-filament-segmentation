"""Training objectives, metrics and the training loop.

``Trainer`` is exposed lazily so that importing the metrics (which need only
torch) does not pull in the model zoo.
"""

from filseg.engine.losses import DiceBCELoss
from filseg.engine.metrics import PanopticQuality, dice_score, iou_score

__all__ = ["DiceBCELoss", "PanopticQuality", "Trainer", "dice_score", "iou_score"]


def __getattr__(name: str):
    if name == "Trainer":
        from filseg.engine.trainer import Trainer

        return Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
