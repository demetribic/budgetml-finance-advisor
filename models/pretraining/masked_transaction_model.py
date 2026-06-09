"""
models/pretraining/masked_transaction_model.py — BERT-style masked pretraining
for transaction sequences.

Overview
--------
Wraps a TransactionTransformer encoder with two pretraining heads:
  - Amount head    : predicts masked transaction amount (regression)
  - Category head  : predicts masked transaction category (classification)

15% of time steps are randomly masked and replaced with a learned [MASK] embedding.
Only masked positions contribute to the loss, so the encoder is forced to learn
contextual representations from surrounding transactions.

After pretraining, the encoder weights can be loaded into any task-specific model
as a warm-start initialization via save_pretrained_encoder / load_pretrained_encoder.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.transformers.base import TransactionTransformer


class MaskedTransactionModel(nn.Module):
    """
    BERT-style masked transaction model for self-supervised pretraining.

    Parameters
    ----------
    encoder       : TransactionTransformer   the shared encoder backbone
    mask_prob     : float   fraction of time steps to mask (default 0.15)
    num_categories: int     number of spending categories (default 9)
    """

    def __init__(
        self,
        encoder:        TransactionTransformer,
        mask_prob:      float = 0.15,
        num_categories: int   = 9,
    ):
        super().__init__()
        self.encoder       = encoder
        self.mask_prob     = mask_prob
        self.num_categories = num_categories
        d_model = encoder.d_model

        # Learnable [MASK] token — replaces masked time steps before input projection
        # Shape: (1, 1, feature_dim). We store it and expand at runtime.
        # We pull feature_dim from the encoder's input projection layer.
        feature_dim = self._infer_feature_dim()
        self.mask_embedding = nn.Parameter(torch.zeros(1, 1, feature_dim))
        nn.init.trunc_normal_(self.mask_embedding, std=0.02)

        # Amount prediction head: d_model → 1 (regression)
        # target is amount_norm (global z-score, can be negative) — no activation
        self.amount_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        # Category prediction head: d_model → num_categories (classification)
        self.category_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_categories),
        )

        self._init_weights()

    def _infer_feature_dim(self) -> int:
        """Read feature_dim from the encoder's HybridInputProjection."""
        # HybridInputProjection stores cont indices; total = cont + cat + merch
        proj = self.encoder.input_proj
        # sum of continuous features + 2 categorical indices
        n_cont = len(proj._CONT_INDICES)   # 9
        return n_cont + 2                  # 11

    def _init_weights(self):
        for module in [self.amount_head, self.category_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """
        Apply random masking then run encoder + both prediction heads.

        Parameters
        ----------
        x : (batch, seq_len, feature_dim)

        Returns
        -------
        dict:
          "masked_x"        (batch, seq_len, feature_dim)  — input after masking
          "mask"            (batch, seq_len)                — True = masked position
          "amount_pred"     (batch, seq_len, 1)             — amount predictions
          "category_logits" (batch, seq_len, num_categories) — cat predictions
          "hidden"          (batch, seq_len+1, d_model)     — full encoder output
        """
        batch, seq_len, feature_dim = x.shape

        # Sample mask: True at positions to mask
        mask = torch.rand(batch, seq_len, device=x.device) < self.mask_prob
        # Ensure at least one position is masked per sequence
        for b in range(batch):
            if not mask[b].any():
                idx = torch.randint(seq_len, (1,)).item()
                mask[b, idx] = True

        # Apply masking: replace masked time steps with mask_embedding
        mask_expand = mask.unsqueeze(-1).expand_as(x)            # (B, T, F)
        mask_emb    = self.mask_embedding.expand(batch, seq_len, -1)
        masked_x    = torch.where(mask_expand, mask_emb, x)

        # Encode
        hidden = self.encoder(masked_x)                          # (B, T+1, d_model)
        # Slice off CLS token → per-token hidden states
        token_hidden = hidden[:, 1:, :]                          # (B, T, d_model)

        # Predictions at every position (loss masks to masked-only positions)
        amount_pred     = self.amount_head(token_hidden)         # (B, T, 1)
        category_logits = self.category_head(token_hidden)      # (B, T, num_categories)

        return {
            "masked_x":        masked_x,
            "mask":            mask,
            "amount_pred":     amount_pred,
            "category_logits": category_logits,
            "hidden":          hidden,
        }

    def loss(self, out: dict[str, Tensor], x_original: Tensor) -> Tensor:
        """
        Compute masked prediction loss at masked positions only.

        Parameters
        ----------
        out        : forward() output dict
        x_original : (batch, seq_len, feature_dim) — unmasked input

        Returns
        -------
        Combined scalar loss.
        """
        mask = out["mask"]                          # (B, T) bool

        # Ground truth targets
        # Amount: index 0 in feature vector is amount_norm
        amount_true = x_original[:, :, 0].unsqueeze(-1)    # (B, T, 1)
        # Category: index 1 is cat_id (integer stored as float); clamp to valid range
        cat_true    = x_original[:, :, 1].long().clamp(0, self.num_categories - 1)  # (B, T)

        # Apply mask — flatten masked positions
        amount_pred_masked = out["amount_pred"][mask]        # (M, 1)
        amount_true_masked = amount_true[mask]               # (M, 1)
        cat_logits_masked  = out["category_logits"][mask]   # (M, num_categories)
        cat_true_masked    = cat_true[mask]                  # (M,)

        if amount_pred_masked.numel() == 0:
            return torch.tensor(0.0, device=amount_true.device, requires_grad=True)

        amount_loss = F.huber_loss(amount_pred_masked, amount_true_masked, delta=1.0)
        cat_loss    = F.cross_entropy(cat_logits_masked, cat_true_masked)

        return amount_loss + 0.5 * cat_loss

    def save_pretrained_encoder(self, path: Path | str) -> None:
        """Save only the encoder weights for use as a task-model warm start."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.encoder.state_dict(), str(path))

    @classmethod
    def load_pretrained_encoder(
        cls,
        path: Path | str,
        **encoder_kwargs,
    ) -> TransactionTransformer:
        """
        Load a pretrained encoder checkpoint and return a new TransactionTransformer.

        Parameters
        ----------
        path           : path to checkpoint saved by save_pretrained_encoder()
        encoder_kwargs : constructor kwargs forwarded to TransactionTransformer

        Returns
        -------
        TransactionTransformer with pretrained weights loaded.
        """
        encoder = TransactionTransformer(**encoder_kwargs)
        state   = torch.load(str(path), map_location="cpu")
        encoder.load_state_dict(state)
        return encoder

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
