"""Training loop.

Deliberately plain: one file, no framework, checkpoints written after every
improvement so a Kaggle session that hits its time limit still leaves a usable
model behind in ``/kaggle/working``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from filseg.config import Config
from filseg.engine.losses import DiceBCELoss
from filseg.engine.metrics import dice_score, iou_score
from filseg.models.build import save_checkpoint
from filseg.utils.logging import get_logger

logger = get_logger(__name__)


class Trainer:
    """Fit a segmentation model and track the best validation Dice."""

    def __init__(self, model: torch.nn.Module, cfg: Config, output_dir: Path,
                 device: str | torch.device | None = None):
        self.cfg = cfg
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = DiceBCELoss(cfg.train.bce_weight, cfg.train.dice_weight)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.train.epochs
        )
        self.use_amp = cfg.train.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)
        self.history: list[dict[str, float]] = []

    #: Non-finite training losses tolerated within one epoch before fit() raises.
    #: A run that has genuinely diverged (unstable from-scratch init, learning
    #: rate too high for the architecture, fp16 overflow) produces far more
    #: than this in its first few batches -- raising early turns a wasted
    #: multi-hour Kaggle session into a few seconds of wasted compute instead.
    #: Occasional isolated non-finite batches in an otherwise healthy run are
    #: just skipped and don't count toward other epochs.
    NON_FINITE_PATIENCE = 10

    def _step(self) -> None:
        if self.cfg.train.grad_clip:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        accumulation_steps = max(1, self.cfg.train.accumulation_steps)
        total, batches, non_finite = 0.0, 0, 0
        pending = False
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader):
            images = batch["image"].to(self.device, non_blocking=True)
            targets = batch["mask"].to(self.device, non_blocking=True)

            with torch.autocast(self.device.type, enabled=self.use_amp):
                loss = self.criterion(self.model(images), targets)

            if not torch.isfinite(loss):
                non_finite += 1
                if non_finite > self.NON_FINITE_PATIENCE:
                    raise FloatingPointError(
                        f"{non_finite} non-finite training losses in this epoch -- the "
                        "model has diverged. For the improved U-Nets (dilation-122436, "
                        "u-4floor, ...) this usually means model.norm=false (no "
                        "BatchNorm) is unstable at this train.lr, especially with "
                        "train.amp=true (fp16 overflows at ~65504). Try model.norm=true "
                        "and/or a lower train.lr before restarting."
                    )
                continue  # skip this batch: no backward, no optimizer step

            # Averaging the loss before backward (rather than the gradients
            # after) keeps a step's gradient magnitude the same regardless of
            # accumulation_steps, so lr doesn't need retuning when it changes.
            self.scaler.scale(loss / accumulation_steps).backward()
            pending = True

            if (step + 1) % accumulation_steps == 0:
                self._step()
                pending = False

            total += float(loss.detach())
            batches += 1

        if pending:  # flush a final partial accumulation group
            self._step()

        if non_finite:
            logger.warning("skipped %d non-finite training batch(es) this epoch",
                           non_finite)
        return total / max(batches, 1)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        loss_sum, dice_sum, iou_sum, n = 0.0, 0.0, 0.0, 0
        threshold = self.cfg.postprocess.threshold
        for batch in loader:
            images = batch["image"].to(self.device, non_blocking=True)
            targets = batch["mask"].to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            predictions = (torch.sigmoid(logits.float()) >= threshold).float()
            batch_size = images.size(0)
            loss_sum += float(loss) * batch_size
            dice_sum += float(dice_score(predictions, targets).sum())
            iou_sum += float(iou_score(predictions, targets).sum())
            n += batch_size

        n = max(n, 1)
        return {"val_loss": loss_sum / n, "val_dice": dice_sum / n, "val_iou": iou_sum / n}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict[str, float]:
        best_dice = -1.0
        best_path = self.output_dir / f"{self.cfg.name}_best.pt"

        for epoch in range(1, self.cfg.train.epochs + 1):
            started = time.time()
            train_loss = self._train_epoch(train_loader)
            metrics = self._validate(val_loader)
            self.scheduler.step()

            record = {"epoch": epoch, "train_loss": train_loss, **metrics,
                      "lr": self.optimizer.param_groups[0]["lr"],
                      "seconds": time.time() - started}
            self.history.append(record)
            logger.info(
                "epoch %02d/%d | train %.4f | val %.4f | dice %.4f | iou %.4f | %.0fs",
                epoch, self.cfg.train.epochs, train_loss, metrics["val_loss"],
                metrics["val_dice"], metrics["val_iou"], record["seconds"],
            )

            if metrics["val_dice"] > best_dice:
                best_dice = metrics["val_dice"]
                save_checkpoint(best_path, self.model, self.cfg,
                                extra={"epoch": epoch, "metrics": metrics})
                logger.info("new best val dice %.4f -> %s", best_dice, best_path)

        save_checkpoint(self.output_dir / f"{self.cfg.name}_last.pt", self.model, self.cfg,
                        extra={"epoch": self.cfg.train.epochs})
        (self.output_dir / f"{self.cfg.name}_history.json").write_text(
            json.dumps(self.history, indent=2)
        )
        return {"best_val_dice": best_dice, "checkpoint": str(best_path)}
