"""
spending_forecast.py — Transformer that forecasts next-month spending per category.

Task
----
Given a 60-day transaction sequence, predict total spend in each of the
NUM_CATEGORIES spending categories for the next 30 days.

Architecture
------------
  TransactionTransformer (base encoder — own instance or shared backbone)
  → [CLS] token  (d_model,)
  → regression head: Linear → GELU → Dropout → Linear → Softplus → (NUM_CATEGORIES,)

Loss: Huber loss (robust to outlier spend months).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from models.transformers.base import TransactionTransformer

NUM_CATEGORIES = 9   # matches preprocessor.CATEGORIES


class SpendingForecastTransformer(nn.Module):
    """
    Predicts monthly per-category spending from a transaction sequence.

    Parameters
    ----------
    feature_dim     : int   transaction feature dimension (from preprocessor)
    num_categories  : int   number of spending categories to forecast
    d_model         : int   transformer hidden size
    nhead           : int   attention heads
    num_layers      : int   encoder depth
    dim_feedforward : int   FFN inner size
    dropout         : float dropout rate
    encoder         : TransactionTransformer | None
        Optional shared backbone. If None (default), instantiates its own encoder.
        Pass an existing TransactionTransformer to share weights across tasks.
    """

    def __init__(
        self,
        feature_dim:     int   = 11,
        num_categories:  int   = NUM_CATEGORIES,
        d_model:         int   = 128,
        nhead:           int   = 4,
        num_layers:      int   = 3,
        dim_feedforward: int   = 256,
        dropout:         float = 0.1,
        encoder:         TransactionTransformer | None = None,
    ):
        super().__init__()
        self.num_categories = num_categories
        self.dropout_p = dropout

        if encoder is not None:
            self.encoder = encoder
            d_model = encoder.d_model
        else:
            self.encoder = TransactionTransformer(
                feature_dim=feature_dim,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )

        # Regression head: CLS → per-category spend estimate
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_categories),
            nn.Softplus(),   # ensures non-negative spend predictions
        )

        # Coverage adjustment scalar for calibrate_uncertainty()
        self._coverage_adjustment: float = 1.0

        self._init_weights()

    def _init_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, feature_dim)

        Returns
        -------
        pred : (batch, num_categories)   predicted spend per category (non-negative)
        """
        cls = self.encoder.get_cls(x)   # (batch, d_model)
        return self.head(cls)           # (batch, num_categories)

    def mc_predict(
        self,
        x: Tensor,
        n_samples: int = 30,
    ) -> dict[str, Tensor]:
        """
        Monte Carlo Dropout inference — quantifies epistemic uncertainty.

        Runs n_samples forward passes with dropout ENABLED (model.train() mode)
        and returns the empirical distribution of predictions.

        Parameters
        ----------
        x         : (batch, seq_len, feature_dim)
        n_samples : int   number of stochastic forward passes (default 30)

        Returns
        -------
        dict with keys:
          "mean"  : (batch, num_categories) — point estimate (mean of samples)
          "std"   : (batch, num_categories) — epistemic uncertainty (std of samples)
          "lower" : (batch, num_categories) — 10th percentile (adjusted for calibration)
          "upper" : (batch, num_categories) — 90th percentile (adjusted for calibration)
        """
        self.eval()
        # Set only dropout layers to train mode for MC sampling
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()
        try:
            with torch.no_grad():
                samples = torch.stack(
                    [self.forward(x) for _ in range(n_samples)], dim=0
                )   # (n_samples, batch, num_categories)
        finally:
            self.eval()   # always restore eval mode, even on exception

        mean = samples.mean(dim=0)
        std  = samples.std(dim=0)

        # Raw 10th/90th percentiles, then scale half-width by coverage adjustment
        raw_lower = samples.quantile(0.10, dim=0)
        raw_upper = samples.quantile(0.90, dim=0)
        half_width = (raw_upper - raw_lower) / 2.0
        adj = self._coverage_adjustment
        lower = mean - adj * half_width
        upper = mean + adj * half_width

        return {
            "mean":  mean,
            "std":   std,
            "lower": lower.clamp(min=0.0),   # spend can't be negative
            "upper": upper,
        }

    def calibrate_uncertainty(
        self,
        val_loader,
        n_samples: int = 30,
        target_coverage: float = 0.80,
        device: torch.device | None = None,
    ) -> float:
        """
        Measure empirical coverage of the 80% prediction interval on a validation
        set, then store a scalar that scales the intervals to hit target_coverage.

        Parameters
        ----------
        val_loader      : DataLoader yielding (x, y) batches
        n_samples       : MC Dropout samples per batch
        target_coverage : desired coverage (default 0.80 = 80% CI)
        device          : device to run inference on

        Returns
        -------
        float : the coverage_adjustment scalar that was stored
        """
        all_lower, all_upper, all_true = [], [], []

        for xb, yb in val_loader:
            if device:
                xb, yb = xb.to(device), yb.to(device)
            out = self.mc_predict(xb, n_samples=n_samples)
            all_lower.append(out["lower"].cpu())
            all_upper.append(out["upper"].cpu())
            all_true.append(yb.cpu())

        lower = torch.cat(all_lower, dim=0)
        upper = torch.cat(all_upper, dim=0)
        true  = torch.cat(all_true,  dim=0)

        # Empirical coverage: fraction of true values inside [lower, upper]
        in_interval = ((true >= lower) & (true <= upper)).float()
        empirical_coverage = float(in_interval.mean())

        # Scale adjustment: if coverage too low, widen intervals; if too high, narrow.
        # Clamped to [0.5, 2.0] so intervals never collapse or balloon unreasonably.
        if empirical_coverage > 0:
            self._coverage_adjustment = float(
                min(2.0, max(0.5, target_coverage / empirical_coverage))
            )
        else:
            self._coverage_adjustment = 1.0

        return self._coverage_adjustment

    @staticmethod
    def loss(pred: Tensor, target: Tensor) -> Tensor:
        """Huber loss — less sensitive to one-off large spend months."""
        return nn.functional.huber_loss(pred, target, delta=50.0)

    def predict(self, x: Tensor) -> Tensor:
        """Inference wrapper — returns point-estimate predictions without gradient."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
