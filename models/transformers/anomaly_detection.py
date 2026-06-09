"""
anomaly_detection.py — Transformer autoencoder for transaction anomaly detection.

Task
----
Detect unusual spending patterns in a user's transaction history.
Outputs:
  1. anomaly_score  — reconstruction error (higher = more anomalous)
  2. anomaly_class  — 0=normal, 1=anomaly (binary classification)

Architecture
------------
  Encoder : TransactionTransformer → [CLS] → bottleneck (d_model // 4)
  Decoder : TransformerDecoder reconstructs sequence from latent vector
            (cross-attention from position queries to latent — preserves temporal order)
  Classifier head : [CLS] → 2-class (normal / anomaly)

Training
--------
  - Reconstruction loss (MSE) on normal transactions
  - Classification loss (cross-entropy) on binary anomaly labels
  - Combined loss = recon_loss + lambda * cls_loss
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from models.transformers.base import TransactionTransformer, Time2Vec


class AnomalyDetectionTransformer(nn.Module):
    """
    Transformer autoencoder that detects and classifies transaction anomalies.

    Parameters
    ----------
    feature_dim     : int   transaction feature dimension
    seq_len         : int   input sequence length (needed for decoder)
    d_model         : int   transformer hidden size
    nhead           : int   attention heads
    num_layers      : int   encoder depth
    dim_feedforward : int   FFN size
    bottleneck_dim  : int   compressed latent dimension
    dropout         : float
    lambda_cls      : float weight on classification loss vs reconstruction loss
    encoder         : TransactionTransformer | None
        Optional shared backbone. If None, instantiates its own encoder.
    """

    NUM_CLASSES = 2   # 0=normal, 1=anomaly (binary labels from PersonaLedger)

    def __init__(
        self,
        feature_dim:         int   = 11,
        seq_len:             int   = 60,
        d_model:             int   = 128,
        nhead:               int   = 4,
        num_layers:          int   = 3,
        dim_feedforward:     int   = 256,
        bottleneck_dim:      int   = 32,
        dropout:             float = 0.1,
        lambda_cls:          float = 0.5,
        encoder:             TransactionTransformer | None = None,
    ):
        super().__init__()
        self.feature_dim    = feature_dim
        self.seq_len        = seq_len
        self.bottleneck_dim = bottleneck_dim
        self.lambda_cls     = lambda_cls

        # ── Encoder ───────────────────────────────────────────────────────────
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
        self.d_model = d_model

        # Bottleneck: compress [CLS] to a small latent vector
        self.bottleneck = nn.Sequential(
            nn.Linear(d_model, bottleneck_dim),
            nn.Tanh(),
        )

        # ── Transformer Decoder: reconstruct the full sequence from latent ───
        # Expand latent → d_model for use as memory (key/value) in cross-attention
        self.decoder_proj = nn.Linear(bottleneck_dim, d_model)

        # Time2Vec position encoding for decoder queries
        t2v_k = max(2, d_model // 4)
        self.decoder_pos = Time2Vec(k=t2v_k)
        self.decoder_pos_proj = nn.Linear(t2v_k, d_model)

        # Stack of TransformerDecoderLayer: queries = position embeddings,
        # keys/values = latent memory vector.  Preserves temporal structure.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Project each decoded token back to feature space
        self.output_proj = nn.Linear(d_model, feature_dim)

        # ── Anomaly classifier head ────────────────────────────────────────
        # Uses [CLS] (not bottleneck) for richer representation
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, self.NUM_CLASSES),
        )

        # Calibrated threshold — set by calibrate_threshold(), used in predict()
        self._calibrated_threshold: float | None = None

        self._init_weights()

    def _init_weights(self):
        for module in [self.bottleneck, self.decoder_proj,
                       self.decoder_pos_proj, self.output_proj, self.classifier]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                for m in module.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """
        Parameters
        ----------
        x : (batch, seq_len, feature_dim)

        Returns
        -------
        dict with keys:
          "recon"           (batch, seq_len, feature_dim)  reconstructed sequence
          "anomaly_score"   (batch,)                       MSE reconstruction error
          "anomaly_logits"  (batch, NUM_CLASSES)           raw class logits
          "latent"          (batch, bottleneck_dim)        compressed representation
        """
        batch = x.size(0)

        # Encode
        hidden = self.encoder(x)                    # (B, seq_len+1, d_model)
        cls = hidden[:, 0, :]                       # (B, d_model)

        # Bottleneck
        latent = self.bottleneck(cls)               # (B, bottleneck_dim)

        # Expand latent → memory for decoder cross-attention: (B, 1, d_model)
        memory = self.decoder_proj(latent).unsqueeze(1)   # (B, 1, d_model)

        # Build decoder queries from Time2Vec position encoding: (1, seq_len, d_model)
        pos_enc = self.decoder_pos(self.seq_len, x.device)   # (1, T, t2v_k)
        queries = self.decoder_pos_proj(pos_enc)              # (1, T, d_model)
        queries = queries.expand(batch, -1, -1)               # (B, T, d_model)

        # Decode: queries cross-attend to latent memory — reconstructs temporal sequence
        decoded = self.decoder(tgt=queries, memory=memory)    # (B, T, d_model)

        # Project each token to feature space
        recon = self.output_proj(decoded)           # (B, T, feature_dim)

        # Per-sample reconstruction error (anomaly score)
        anomaly_score = F.mse_loss(recon, x, reduction="none").mean(dim=[1, 2])

        # Classify anomaly from CLS
        anomaly_logits = self.classifier(cls)       # (B, NUM_CLASSES)

        return {
            "recon":          recon,
            "anomaly_score":  anomaly_score,
            "anomaly_logits": anomaly_logits,
            "latent":         latent,
        }

    def loss(
        self,
        x: Tensor,
        out: dict[str, Tensor],
        labels: Tensor | None = None,
    ) -> Tensor:
        """
        Combined reconstruction + classification loss.

        Parameters
        ----------
        x      : original input (batch, seq_len, feature_dim)
        out    : output dict from forward()
        labels : (batch,) long tensor — 0=normal, 1=anomaly
                 If None, only reconstruction loss is used.
        """
        recon_loss = F.mse_loss(out["recon"], x)

        if labels is not None:
            cls_loss = F.cross_entropy(out["anomaly_logits"], labels)
            return recon_loss + self.lambda_cls * cls_loss

        return recon_loss

    def calibrate_threshold(
        self,
        val_loader: DataLoader,
        percentile: float = 95.0,
        device: torch.device | None = None,
    ) -> float:
        """
        Compute the anomaly threshold on a held-out validation set of normal
        transactions. The threshold is the `percentile`-th percentile of
        reconstruction errors across all normal samples in val_loader.

        Call this after training; the result is stored as
        `self._calibrated_threshold` and used automatically in predict().

        Parameters
        ----------
        val_loader  : DataLoader yielding (x, y) batches
                      Ideally filtered to normal-only (y==0) samples.
        percentile  : float   which percentile to use as threshold (default 95)
        device      : torch.device or None

        Returns
        -------
        float : the calibrated threshold value
        """
        self.eval()
        all_scores: list[float] = []

        with torch.no_grad():
            for batch in val_loader:
                xb = batch[0]
                if device:
                    xb = xb.to(device)
                out = self.forward(xb)
                scores = out["anomaly_score"].cpu().numpy()
                all_scores.extend(scores.tolist())

        if not all_scores:
            raise ValueError(
                "calibrate_threshold: val_loader yielded no batches. "
                "Ensure val_loader is not empty before calibrating."
            )

        threshold = float(np.percentile(all_scores, percentile))
        self._calibrated_threshold = threshold
        return threshold

    def per_user_score(
        self,
        current_score: float,
        user_history: np.ndarray,
    ) -> float:
        """
        Compute a z-score of `current_score` relative to a user's own
        historical reconstruction error distribution.

        A z-score ≥ 3 indicates the transaction is highly anomalous for
        this user, even if it would not trip a global threshold.

        Parameters
        ----------
        current_score : float   reconstruction error for the new window
        user_history  : np.ndarray   array of past reconstruction errors for
                        this user (typically stored in application layer)

        Returns
        -------
        float : z-score — positive = above user's own mean
        """
        if len(user_history) < 2:
            return float("nan")  # distinguish "no data" from "normal" (z=0)
        u_mean = float(user_history.mean())
        u_std  = float(user_history.std())
        if u_std < 1e-8:
            return 0.0
        return (current_score - u_mean) / u_std

    def predict(self, x: Tensor, threshold: float | None = None) -> dict[str, Tensor]:
        """
        Inference: returns anomaly scores, flags, and predicted classes.

        Uses the calibrated threshold (from calibrate_threshold()) if available
        and no explicit threshold is passed.

        Parameters
        ----------
        x         : (batch, seq_len, feature_dim)
        threshold : float, optional
            Override threshold. If None, uses self._calibrated_threshold if set,
            else falls back to 95th-percentile of the batch (backwards compat).

        Returns
        -------
        dict:
          "anomaly_score"  (batch,)   float
          "is_anomaly"     (batch,)   bool
          "anomaly_class"  (batch,)   int  (0 or 1)
          "class_probs"    (batch, NUM_CLASSES) float
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(x)

        scores = out["anomaly_score"]

        if threshold is not None:
            effective_threshold = threshold
        elif self._calibrated_threshold is not None:
            effective_threshold = self._calibrated_threshold
        else:
            # Fallback: batch-level 95th percentile (original behaviour)
            p95 = float(torch.quantile(scores, 0.95))
            effective_threshold = max(p95, float(scores.min()) + 1e-6)

        is_anomaly   = scores > effective_threshold
        class_probs  = F.softmax(out["anomaly_logits"], dim=-1)
        anomaly_class = class_probs.argmax(dim=-1)

        return {
            "anomaly_score":  scores,
            "is_anomaly":     is_anomaly,
            "anomaly_class":  anomaly_class,
            "class_probs":    class_probs,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
