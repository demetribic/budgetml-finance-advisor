"""
models/transformers/contrastive_anomaly.py — SimCLR-style contrastive anomaly detector.

Normal spending patterns form tight clusters in projection space.
Anomalous sequences fall far from all normal clusters (high min-distance score).

Training procedure:
1. Take a batch of NORMAL transaction windows (is_anomaly == False).
2. Create two independently augmented views of each window.
3. Train encoder + projection head with NT-Xent loss to pull views together.
4. After training, build a memory bank of all normal sequence embeddings.
5. At inference, anomaly score = minimum cosine distance to any memory bank entry.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from models.transformers.base import TransactionTransformer


class ContrastiveAnomalyDetector(nn.Module):
    """
    SimCLR-style contrastive anomaly detector for transaction sequences.

    Parameters
    ----------
    encoder        : TransactionTransformer   shared or dedicated backbone
    projection_dim : int   projection head output dimension (default 64)
    temperature    : float NT-Xent loss temperature (default 0.07)
    """

    def __init__(
        self,
        encoder:        TransactionTransformer,
        projection_dim: int   = 64,
        temperature:    float = 0.07,
    ):
        super().__init__()
        self.encoder     = encoder
        self.temperature = temperature
        d_model          = encoder.d_model

        # Non-linear projection head (SimCLR style: 2-layer MLP + L2-norm)
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, projection_dim),
        )

        # Memory bank: filled by fit_memory_bank() after training
        self._memory_bank: Tensor | None = None
        self._memory_bank_threshold: float | None = None

    # ── Augmentations ─────────────────────────────────────────────────────────

    def augment(self, x: Tensor) -> Tensor:
        """
        Apply a random combination of augmentations to a batch of sequences.
        All augmentations operate on the feature tensor — no raw data needed.

        Augmentations applied:
        - AmountJitter:  Gaussian noise N(0, 0.05) on amount_norm (index 0)
        - TemporalShift: ±1 day shift on cyclical features (indices 3–8)
        - DropoutMask:   5% of time steps zeroed out
        - CategorySwap:  5% chance each category replaced with a neighbour

        Parameters
        ----------
        x : (batch, seq_len, feature_dim)

        Returns
        -------
        Tensor : augmented x with same shape
        """
        x = x.clone()
        batch, seq_len, feat_dim = x.shape

        # AmountJitter: add noise to amount_norm (index 0)
        noise = torch.randn(batch, seq_len, device=x.device) * 0.05
        x[:, :, 0] = x[:, :, 0] + noise

        # TemporalShift: shift cyclical features by a random fraction (±1 day)
        # Cyclical features: indices 3–8 (dow_sin, dow_cos, dom_sin, dom_cos, m_sin, m_cos)
        delta = torch.randint(-1, 2, (batch,), device=x.device).float() / 7.0
        for b in range(batch):
            if delta[b] != 0:
                shifted3 = (torch.asin(x[b, :, 3].clamp(-1, 1)) + delta[b] * 3.14159).clamp(-1.5707963, 1.5707963)
                x[b, :, 3] = torch.sin(shifted3)
                shifted4 = (torch.acos(x[b, :, 4].clamp(-1, 1)) + delta[b] * 3.14159).clamp(0, 3.14159265)
                x[b, :, 4] = torch.cos(shifted4)

        # DropoutMask: randomly zero 5% of time steps
        drop_mask = torch.rand(batch, seq_len, device=x.device) < 0.05
        x[drop_mask] = 0.0

        # CategorySwap: with prob 0.05, replace cat_id (index 1) with a nearby category
        cat_swap_mask = torch.rand(batch, seq_len, device=x.device) < 0.05
        swap_cats     = torch.randint(0, 9, (batch, seq_len), device=x.device).float()
        x[:, :, 1]   = torch.where(cat_swap_mask, swap_cats, x[:, :, 1])

        return x

    # ── Forward ───────────────────────────────────────────────────────────────

    def project(self, x: Tensor) -> Tensor:
        """
        Encode x → CLS → projection head → L2-normalized embedding.

        Parameters
        ----------
        x : (batch, seq_len, feature_dim)

        Returns
        -------
        Tensor : (batch, projection_dim) — unit-length vectors
        """
        cls  = self.encoder.get_cls(x)            # (B, d_model)
        proj = self.projection_head(cls)           # (B, projection_dim)
        return F.normalize(proj, dim=-1)

    def nt_xent_loss(self, z1: Tensor, z2: Tensor) -> Tensor:
        """
        NT-Xent (normalized temperature-scaled cross-entropy) loss.

        z1, z2 : (batch, projection_dim) — two augmented views of the same batch.
        Positive pairs: (z1[i], z2[i]) for each i.
        Negative pairs: all other combinations within the batch.

        Returns scalar loss.
        """
        batch = z1.size(0)
        # Concatenate both views: (2B, D)
        z = torch.cat([z1, z2], dim=0)
        # Compute logits in FP32 for numeric stability under AMP/FP16.
        sim = (z.float() @ z.float().T) / self.temperature

        # Remove self-similarity from numerator
        mask = torch.eye(2 * batch, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, torch.finfo(sim.dtype).min)

        # Labels: z1[i] is positive with z2[i] (offset by batch)
        labels = torch.arange(batch, device=z.device)
        labels = torch.cat([labels + batch, labels])   # (2B,)

        return F.cross_entropy(sim, labels)

    # ── Memory bank ───────────────────────────────────────────────────────────

    def fit_memory_bank(
        self,
        normal_loader: DataLoader,
        device: torch.device | None = None,
        percentile: float = 95.0,
    ) -> None:
        """
        Compute and store projection embeddings for all normal sequences.

        Also calibrates the anomaly threshold as the `percentile`-th percentile
        of minimum within-bank distances (self-distances on the normal set).

        Parameters
        ----------
        normal_loader : DataLoader yielding (x, y) or (x,) batches of NORMAL sequences
        device        : inference device
        percentile    : float — percentile of normal self-distances to use as threshold
        """
        self.eval()
        all_proj: list[Tensor] = []

        with torch.no_grad():
            for batch in normal_loader:
                xb = batch[0] if isinstance(batch, (list, tuple)) else batch
                if device:
                    xb = xb.to(device)
                proj = self.project(xb)
                all_proj.append(proj.cpu())

        if not all_proj:
            return

        bank = torch.cat(all_proj, dim=0)   # (N, projection_dim)
        self._memory_bank = bank

        # Calibrate threshold: min cosine distance to nearest neighbour for each normal
        # sample. Computed in chunks to avoid O(N²) peak memory allocation.
        chunk = 512
        n = len(bank)
        min_dists = torch.full((n,), float("inf"))
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            sims_chunk = bank[start:end] @ bank.T       # (chunk, N)
            # Exclude self-similarity on the diagonal of this chunk
            for local_i in range(end - start):
                global_i = start + local_i
                sims_chunk[local_i, global_i] = -2.0    # force below any real similarity
            max_sims_chunk = sims_chunk.max(dim=-1).values
            min_dists[start:end] = 1.0 - max_sims_chunk
        self._memory_bank_threshold = float(
            torch.quantile(min_dists, percentile / 100.0)
        )

    def anomaly_score(self, x: Tensor) -> Tensor:
        """
        Compute anomaly score for each sequence as the minimum cosine distance
        to any entry in the normal memory bank.

        Parameters
        ----------
        x : (batch, seq_len, feature_dim)

        Returns
        -------
        Tensor : (batch,) — higher = more anomalous
        """
        if self._memory_bank is None:
            raise RuntimeError("Call fit_memory_bank() before anomaly_score()")

        proj = self.project(x)                              # (B, D), unit-length
        bank = self._memory_bank.to(proj.device)            # (N, D), unit-length
        sims = proj @ bank.T                               # (B, N) cosine similarities
        max_sim   = sims.max(dim=-1).values                # (B,) nearest-neighbour sim
        dist      = 1.0 - max_sim                          # cosine distance (0=identical)
        return dist                                        # (B,)

    def predict(
        self,
        x:         Tensor,
        threshold: float | None = None,
    ) -> dict[str, Tensor]:
        """
        Inference: return anomaly scores and binary flags.

        Parameters
        ----------
        x         : (batch, seq_len, feature_dim)
        threshold : float | None — override; if None, uses _memory_bank_threshold

        Returns
        -------
        dict:
          "anomaly_score" (batch,)  float
          "is_anomaly"    (batch,)  bool
          "percentile"    (batch,)  float — each score's percentile in the memory bank
        """
        self.eval()
        bank = self._memory_bank  # may be None if fit_memory_bank() not called

        with torch.no_grad():
            proj = self.project(x)                             # encode once, reuse below

            if bank is not None:
                bank = bank.to(proj.device)
                sims = proj @ bank.T                           # (B, N)
                scores = 1.0 - sims.max(dim=-1).values        # cosine distance
            else:
                sims   = None
                scores = torch.full((x.size(0),), 0.5, device=x.device)

        if threshold is None:
            threshold = self._memory_bank_threshold or 0.5

        is_anomaly = scores > threshold

        # Percentile of each score relative to memory bank distances
        if bank is not None and sims is not None:
            dists = 1.0 - sims.max(dim=-1).values
            bank_self_sims = bank @ bank.T
            eye = torch.eye(len(bank), dtype=torch.bool, device=bank.device)
            bank_self_sims.masked_fill_(eye, 2.0)
            bank_dists = 1.0 - bank_self_sims.max(dim=-1).values   # (N,)
            percentiles = torch.tensor(
                [float((dists[i].item() > bank_dists).float().mean()) for i in range(len(dists))],
                device=scores.device,
            )
        else:
            percentiles = torch.full_like(scores, 0.5)

        return {
            "anomaly_score": scores,
            "is_anomaly":    is_anomaly,
            "percentile":    percentiles,
        }

    def save(self, path: Path | str) -> None:
        """Save model + memory bank."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict":   self.state_dict(),
            "memory_bank":  self._memory_bank,
            "threshold":    self._memory_bank_threshold,
            "projection_dim": self.projection_head[-1].out_features,
            "temperature":  self.temperature,
        }, str(path))

    @classmethod
    def load(
        cls,
        path: Path | str,
        encoder: TransactionTransformer,
    ) -> "ContrastiveAnomalyDetector":
        """Load a saved checkpoint."""
        ckpt  = torch.load(str(path), map_location="cpu", weights_only=False)
        model = cls(
            encoder=encoder,
            projection_dim=ckpt["projection_dim"],
            temperature=ckpt["temperature"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model._memory_bank = ckpt.get("memory_bank")
        model._memory_bank_threshold = ckpt.get("threshold")
        return model

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
