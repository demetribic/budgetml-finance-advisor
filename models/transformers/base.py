"""
base.py — Shared TransactionTransformer base class.

Architecture
------------
  Input  : (batch, seq_len, feature_dim)
  ↓ HybridInputProjection  (embeddings for cat/merch + linear for continuous)
  ↓ Time2Vec temporal encoding
  ↓ N × TransformerEncoderLayer (multi-head self-attention + FFN)
  ↓ output per-token hidden states  (batch, seq_len+1, d_model)
  ↓ [CLS] token = hidden[:, 0, :]  — used by task heads

Feature vector layout (after Subagent 1-A):
  Index 0  : amount_norm          (continuous)
  Index 1  : cat_id               (categorical integer 0–8  → nn.Embedding)
  Index 2  : merch_id             (categorical integer 0–511 → nn.Embedding)
  Indices 3–8  : cyclical time features (continuous)
  Index 9  : user_amount_zscore   (continuous)
  Index 10 : description_hash     (continuous)

A learnable [CLS] token is prepended so heads can read a global summary.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor


# ── Time2Vec ──────────────────────────────────────────────────────────────────

class Time2Vec(nn.Module):
    """
    Learnable temporal encoding (Kazemi et al., 2019).

    For each position index τ (integer 0, 1, …, seq_len):
        t2v(τ)[0]   = w_0 · τ + b_0             (linear — captures trend)
        t2v(τ)[i>0] = sin(w_i · τ + b_i)        (periodic — captures cycles)

    Output shape: (batch, seq_len, k) — broadcast-added to the projected input.

    Parameters
    ----------
    k : int
        Number of output components. Index 0 is linear; 1..k-1 are sinusoidal.
    """

    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.w = nn.Parameter(torch.randn(k))
        self.b = nn.Parameter(torch.randn(k))

    def forward(self, seq_len: int, device: torch.device) -> Tensor:
        """
        Returns temporal encoding for positions 0..seq_len-1.

        Returns
        -------
        Tensor : (1, seq_len, k) — ready to broadcast over batch
        """
        tau = torch.arange(seq_len, dtype=torch.float32, device=device)  # (T,)
        # (T, k)
        v = tau.unsqueeze(-1) * self.w.unsqueeze(0) + self.b.unsqueeze(0)
        v = torch.cat([v[:, :1], torch.sin(v[:, 1:])], dim=-1)  # linear at 0, sin elsewhere
        return v.unsqueeze(0)             # (1, T, k)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── HybridInputProjection ─────────────────────────────────────────────────────

class HybridInputProjection(nn.Module):
    """
    Input projection that handles mixed continuous + categorical features.

    Splits the feature vector into:
      - Continuous features  (indices 0, 3–10) → Linear
      - cat_id   (index 1) → nn.Embedding(9, cat_emb_dim)
      - merch_id (index 2) → nn.Embedding(512, merch_emb_dim)

    All three are concatenated → Linear(concat_dim, d_model) → LayerNorm.

    Parameters
    ----------
    feature_dim   : int   total feature vector width (expected 11)
    d_model       : int   output dimension
    num_categories: int   category vocab size (default 9)
    num_merchants : int   merchant vocab size (default 512)
    cat_emb_dim   : int   category embedding size (default 16)
    merch_emb_dim : int   merchant embedding size (default 32)
    """

    # Feature vector index constants
    CAT_IDX   = 1
    MERCH_IDX = 2
    # All other indices are continuous
    _CONT_INDICES = [0, 3, 4, 5, 6, 7, 8, 9, 10]

    def __init__(
        self,
        feature_dim:    int = 11,
        d_model:        int = 128,
        num_categories: int = 9,
        num_merchants:  int = 512,
        cat_emb_dim:    int = 16,
        merch_emb_dim:  int = 32,
        pretrained_merchant_emb: torch.Tensor | None = None,
    ):
        """
        Parameters
        ----------
        pretrained_merchant_emb : (num_merchants, merch_emb_dim) Tensor, optional
            If provided, initializes the merchant embedding table from Merchant2Vec
            pretrained weights instead of random initialization. The tensor must
            match num_merchants × merch_emb_dim exactly.
        """
        super().__init__()
        self.cat_emb_dim   = cat_emb_dim
        self.merch_emb_dim = merch_emb_dim
        cont_dim = len(self._CONT_INDICES)   # 9

        self.cat_embedding  = nn.Embedding(num_categories, cat_emb_dim,   padding_idx=None)
        self.merch_embedding = nn.Embedding(num_merchants,  merch_emb_dim, padding_idx=None)

        concat_dim = cont_dim + cat_emb_dim + merch_emb_dim  # 9 + 16 + 32 = 57
        self.proj = nn.Sequential(
            nn.Linear(concat_dim, d_model),
            nn.LayerNorm(d_model),
        )

        nn.init.trunc_normal_(self.cat_embedding.weight, std=0.02)

        if pretrained_merchant_emb is not None:
            assert pretrained_merchant_emb.shape == (num_merchants, merch_emb_dim), (
                f"pretrained_merchant_emb shape {pretrained_merchant_emb.shape} "
                f"must be ({num_merchants}, {merch_emb_dim})"
            )
            with torch.no_grad():
                self.merch_embedding.weight.copy_(pretrained_merchant_emb)
        else:
            nn.init.trunc_normal_(self.merch_embedding.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, feature_dim)

        Returns
        -------
        Tensor : (batch, seq_len, d_model)
        """
        # Continuous features
        cont_indices = torch.tensor(self._CONT_INDICES, device=x.device)
        cont = x[:, :, cont_indices]                     # (B, T, 9)

        # Categorical — cast to long for embedding lookup; clamp to valid range
        cat_idx   = x[:, :, self.CAT_IDX].long().clamp(0, self.cat_embedding.num_embeddings - 1)
        merch_idx = x[:, :, self.MERCH_IDX].long().clamp(0, self.merch_embedding.num_embeddings - 1)

        cat_emb   = self.cat_embedding(cat_idx)          # (B, T, cat_emb_dim)
        merch_emb = self.merch_embedding(merch_idx)      # (B, T, merch_emb_dim)

        combined = torch.cat([cont, cat_emb, merch_emb], dim=-1)  # (B, T, 57)
        return self.proj(combined)                       # (B, T, d_model)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── SinusoidalPositionalEncoding (kept for reference) ─────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    Kept for reference; new code defaults to Time2Vec.
    """

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)            # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """x : (batch, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ── TransactionTransformer ────────────────────────────────────────────────────

class TransactionTransformer(nn.Module):
    """
    Base transformer encoder over transaction sequences.

    Parameters
    ----------
    feature_dim     : int   number of input features per time step (default 11)
    d_model         : int   transformer hidden size (must be divisible by nhead)
    nhead           : int   number of attention heads
    num_layers      : int   number of TransformerEncoderLayer stacks
    dim_feedforward : int   FFN inner dimension
    dropout         : float dropout rate
    max_seq_len     : int   maximum sequence length for positional encoding
    num_categories  : int   category vocab size for HybridInputProjection
    num_merchants   : int   merchant vocab size for HybridInputProjection
    """

    def __init__(
        self,
        feature_dim:     int   = 11,
        d_model:         int   = 128,
        nhead:           int   = 4,
        num_layers:      int   = 3,
        dim_feedforward: int   = 256,
        dropout:         float = 0.1,
        max_seq_len:     int   = 512,
        num_categories:  int   = 9,
        num_merchants:   int   = 512,
        pretrained_merchant_emb: torch.Tensor | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout_p = dropout

        # Learnable [CLS] token prepended to every sequence
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Hybrid input projection (handles embeddings + continuous)
        self.input_proj = HybridInputProjection(
            feature_dim=feature_dim,
            d_model=d_model,
            num_categories=num_categories,
            num_merchants=num_merchants,
            pretrained_merchant_emb=pretrained_merchant_emb,
        )

        # Time2Vec: k = d_model // 4 components
        t2v_k = max(2, d_model // 4)
        self.time2vec = Time2Vec(k=t2v_k)
        # Project Time2Vec output to d_model for addition
        self.t2v_proj = nn.Linear(t2v_k, d_model)

        self.pos_dropout = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,        # Pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,   # norm_first=True incompatible with nested tensors
        )

    def forward(self, x: Tensor, src_key_padding_mask: Tensor | None = None) -> Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, feature_dim)
        src_key_padding_mask : (batch, seq_len+1) bool mask — True = ignore

        Returns
        -------
        hidden : (batch, seq_len+1, d_model)
            hidden[:, 0, :] is the [CLS] representation.
        """
        assert x.shape[-1] == 11, f"Expected feature_dim=11, got {x.shape[-1]}"
        batch, seq_len, _ = x.shape

        # Project features via hybrid layer
        x_proj = self.input_proj(x)                     # (B, T, d_model)

        # Time2Vec encoding for positions 0..seq_len-1
        t2v = self.time2vec(seq_len, x.device)          # (1, T, k)
        t2v = self.t2v_proj(t2v)                        # (1, T, d_model)
        x_proj = x_proj + t2v                           # broadcast over batch

        # Prepend CLS token
        cls = self.cls_token.expand(batch, -1, -1)      # (B, 1, d_model)
        x_proj = torch.cat([cls, x_proj], dim=1)        # (B, T+1, d_model)

        x_proj = self.pos_dropout(x_proj)

        # Transformer encoder
        hidden = self.encoder(x_proj, src_key_padding_mask=src_key_padding_mask)
        return hidden

    def get_cls(self, x: Tensor) -> Tensor:
        """Return only the [CLS] token representation. Shape: (batch, d_model)."""
        return self.forward(x)[:, 0, :]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── MultiTaskTransactionTransformer ──────────────────────────────────────────

class MultiTaskTransactionTransformer(nn.Module):
    """
    Single shared encoder with multiple named task heads.

    Enables joint training with a combined loss. Each task registers its head
    via register_head(), then calls forward_task() for task-specific inference
    or forward_all() to run every head in one shot.

    Parameters
    ----------
    encoder : TransactionTransformer
        The shared backbone (constructed externally so hyperparams are explicit).

    Usage
    -----
    backbone = TransactionTransformer(feature_dim=11, d_model=128)
    multi = MultiTaskTransactionTransformer(backbone)
    multi.register_head("forecast", ForecastHead(128, 9))
    multi.register_head("anomaly",  AnomalyHead(128, 2))

    out = multi.forward_task("forecast", x)   # (batch, 9)
    all_outs = multi.forward_all(x)           # {"forecast": …, "anomaly": …}
    """

    def __init__(self, encoder: TransactionTransformer):
        super().__init__()
        self.encoder = encoder
        self.heads: nn.ModuleDict = nn.ModuleDict()

    def register_head(self, name: str, head: nn.Module) -> None:
        """Attach a task head under `name`."""
        self.heads[name] = head

    def forward_task(self, name: str, x: Tensor) -> Tensor | dict:
        """
        Run the shared encoder + a single named head.

        Parameters
        ----------
        name : str   head name previously registered with register_head()
        x    : (batch, seq_len, feature_dim)

        Returns
        -------
        Head output (type depends on the head — Tensor or dict).
        """
        if name not in self.heads:
            raise KeyError(f"No head registered under '{name}'. "
                           f"Available: {list(self.heads.keys())}")
        hidden = self.encoder(x)   # (B, T+1, d_model)
        return self.heads[name](hidden)

    def forward_all(self, x: Tensor) -> dict[str, Tensor | dict]:
        """
        Run the shared encoder once, then all registered heads.

        Returns
        -------
        dict mapping head name → head output
        """
        hidden = self.encoder(x)   # (B, T+1, d_model) — encoder runs once
        return {name: head(hidden) for name, head in self.heads.items()}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
