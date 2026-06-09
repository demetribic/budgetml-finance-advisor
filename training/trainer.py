"""
training/trainer.py — Shared training infrastructure for all transformer models.

BaseTrainer implements the common loop (AMP, gradient clipping, early stopping,
checkpoint saving).  Task-specific subclasses only override compute_loss, so
the loop code lives in exactly one place.

Usage
-----
    trainer = ForecastTrainer(model, optimizer, scheduler, scaler,
                              device, save_path=cfg.save_dir / "forecast_transformer.pt",
                              patience=cfg.training.patience,
                              grad_clip=cfg.training.grad_clip)
    trainer.fit(train_loader, val_loader, epochs=cfg.training.forecast_epochs)
    trainer.load_best()
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader


class BaseTrainer(ABC):
    """
    Common training loop for all transformer models.

    Parameters
    ----------
    model      : nn.Module
    optimizer  : torch.optim.Optimizer
    scheduler  : LR scheduler with a .step() method
    scaler     : GradScaler (AMP; disabled automatically on CPU)
    device     : torch.device
    save_path  : Path   where the best checkpoint is written
    patience   : int    early-stopping patience in epochs without improvement
    grad_clip  : float  max gradient norm for clipping
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        scaler: GradScaler,
        device: torch.device,
        save_path: Path,
        patience: int = 5,
        grad_clip: float = 1.0,
    ):
        self.model     = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler    = scaler
        self.device    = device
        self.save_path = save_path
        self.patience  = patience
        self.grad_clip = grad_clip
        self._use_amp  = device.type == "cuda"

    # ── Abstract interface ─────────────────────────────────────────────────────

    @abstractmethod
    def compute_loss(self, model_out, batch: tuple) -> Tensor:
        """
        Compute a scalar loss for one batch.

        Parameters
        ----------
        model_out : model's forward() output (Tensor or dict depending on task)
        batch     : tuple of tensors from the DataLoader; batch[0] is always x
        """

    # ── Core loop ─────────────────────────────────────────────────────────────

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss, total_n = 0.0, 0
        for batch in loader:
            batch = tuple(t.to(self.device, non_blocking=True) for t in batch)
            self.optimizer.zero_grad()
            with autocast(device_type=self.device.type, enabled=self._use_amp):
                out  = self.model(batch[0])
                loss = self.compute_loss(out, batch)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            n = len(batch[0])
            total_loss += loss.item() * n
            total_n    += n
        return total_loss / max(total_n, 1)

    def val_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss, total_n = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                batch = tuple(t.to(self.device, non_blocking=True) for t in batch)
                with autocast(device_type=self.device.type, enabled=self._use_amp):
                    out  = self.model(batch[0])
                    loss = self.compute_loss(out, batch)
                n = len(batch[0])
                total_loss += loss.item() * n
                total_n    += n
        return total_loss / max(total_n, 1)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> None:
        """Run the training loop with early stopping; saves the best checkpoint."""
        best_val     = float("inf")
        patience_ctr = 0

        for epoch in range(1, epochs + 1):
            t0         = time.time()
            train_loss = self.train_epoch(train_loader)
            self.scheduler.step()
            val_loss   = self.val_epoch(val_loader)
            elapsed    = time.time() - t0

            print(
                f"Epoch {epoch:3d}/{epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"lr={self.scheduler.get_last_lr()[0]:.2e}  {elapsed:.1f}s"
            )

            if val_loss < best_val:
                best_val = val_loss
                torch.save(self.model.state_dict(), self.save_path)
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

    def load_best(self) -> None:
        """Load the best saved checkpoint back into the model."""
        self.model.load_state_dict(
            torch.load(self.save_path, map_location=self.device)
        )


# ── Task-specific trainers ─────────────────────────────────────────────────────

class ForecastTrainer(BaseTrainer):
    """
    batch = (x, y_spend)
    Loss  : Huber (robust to one-off large spend months).
    """

    def compute_loss(self, model_out: Tensor, batch: tuple) -> Tensor:
        return F.huber_loss(model_out, batch[1], delta=50.0)


class AnomalyTrainer(BaseTrainer):
    """
    batch = (x, y_label)
    Loss  : MSE reconstruction on normal samples  +  lambda_cls × cross-entropy.

    The reconstruction loss is computed only on windows labelled normal (y==0)
    so the autoencoder learns the normal manifold; anomalies produce high error.

    After fit() completes, call calibrate_and_save() to compute and persist the
    optimal anomaly threshold on a normal-transaction validation set.
    """

    def compute_loss(self, model_out: dict, batch: tuple) -> Tensor:
        xb, yb      = batch[0], batch[1]
        normal_mask = (yb == 0)

        if normal_mask.any():
            recon_loss = F.mse_loss(model_out["recon"][normal_mask], xb[normal_mask])
        else:
            recon_loss = torch.tensor(0.0, device=xb.device)

        cls_loss = F.cross_entropy(model_out["anomaly_logits"], yb.long())
        return recon_loss + self.model.lambda_cls * cls_loss

    def calibrate_and_save(
        self,
        val_loader,
        percentile: float = 95.0,
        threshold_path: "Path | None" = None,
    ) -> float:
        """
        Compute calibrated threshold on val_loader (normal transactions),
        save it alongside the model checkpoint, and return the value.

        Parameters
        ----------
        val_loader      : DataLoader yielding (x, y) batches, ideally normal-only
        percentile      : which reconstruction-error percentile to use (default 95)
        threshold_path  : Path to write threshold dict; defaults to
                          <save_path parent>/anomaly_threshold.pt

        Returns
        -------
        float : the calibrated threshold
        """
        from pathlib import Path as _Path
        threshold = self.model.calibrate_threshold(
            val_loader, percentile=percentile, device=self.device
        )
        dest = threshold_path or self.save_path.parent / "anomaly_threshold.pt"
        torch.save({"threshold": threshold, "percentile": percentile}, dest)
        print(f"  Calibrated threshold ({percentile:.0f}th pct): {threshold:.6f}  → {dest}")
        return threshold


class BulkBuyTrainer(BaseTrainer):
    """
    batch = (x, y_bulk, y_cat, y_savings)
    Loss  : delegated to BulkBuyRecommendationTransformer.loss()
            (BCE + category cross-entropy + savings regression).
    """

    def compute_loss(self, model_out: dict, batch: tuple) -> Tensor:
        _, bulk_lbl, cat_lbl, sav_lbl = batch
        return self.model.loss(model_out, bulk_lbl, cat_lbl, sav_lbl)


# ── Shared setup helpers ───────────────────────────────────────────────────────

def select_device() -> torch.device:
    """Choose CUDA if available and compute capability >= 7.5, else CPU."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    major, minor = torch.cuda.get_device_capability(0)
    if major * 10 + minor < 75:
        print(
            f"  GPU compute capability {major}.{minor} (sm_{major}{minor}) is not "
            f"supported by this PyTorch build (requires sm_75+). Falling back to CPU."
        )
        return torch.device("cpu")
    return torch.device("cuda")


def make_optimizer_and_scheduler(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    epochs: int,
    lr_eta_min_factor: float,
) -> tuple:
    """AdamW + CosineAnnealingLR. Returns (optimizer, scheduler, scaler)."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * lr_eta_min_factor
    )
    device = next(model.parameters()).device
    scaler = GradScaler(device=device.type, enabled=device.type == "cuda")
    return optimizer, scheduler, scaler
