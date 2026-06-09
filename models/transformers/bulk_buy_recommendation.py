"""
bulk_buy_recommendation.py — Transformer that identifies items worth buying in bulk.

Task
----
Given a user's transaction window, classify whether a frequently-purchased item
(grocery, household supply, subscription) would save money if bought in bulk.
Also outputs: which category / merchant to target.

Architecture
------------
  TransactionTransformer (shared base or own instance)
  → [CLS] token
  → Binary classification head  (bulk-buy yes/no)
  → Category head               (which of NUM_CATEGORIES to target)
  → Estimated monthly savings   (regression, in dollars)

Recommendation logic
--------------------
The model scores each (user, merchant) pair over the window.
High-scoring pairs are passed to the decision engine with:
  - purchase_frequency  (times per 30 days)
  - avg_unit_price
  - estimated_bulk_savings  (predicted by regression head)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.transformers.base import TransactionTransformer

NUM_CATEGORIES = 9


class BulkBuyRecommendationTransformer(nn.Module):
    """
    Identifies which merchants/categories a user should consider buying in bulk.

    Outputs per input window
    ------------------------
    - bulk_buy_logit    : (batch,)             raw binary score
    - category_logits   : (batch, NUM_CAT)     which category to target
    - savings_estimate  : (batch,)             estimated $/month savings

    Parameters
    ----------
    feature_dim     : int
    num_categories  : int
    d_model         : int
    nhead           : int
    num_layers      : int
    dim_feedforward : int
    dropout         : float
    lambda_cat      : float   weight on category classification loss
    lambda_savings  : float   weight on savings regression loss
    encoder         : TransactionTransformer | None
        Optional shared backbone. If None (default), instantiates its own encoder.
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
        lambda_cat:      float = 0.3,
        lambda_savings:  float = 0.2,
        encoder:         TransactionTransformer | None = None,
    ):
        super().__init__()
        self.num_categories = num_categories
        self.lambda_cat     = lambda_cat
        self.lambda_savings = lambda_savings

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

        hidden_half = d_model // 2

        # Binary head: should user consider bulk buying at all?
        self.bulk_head = nn.Sequential(
            nn.Linear(d_model, hidden_half),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_half, 1),
        )

        # Category head: which category to target for bulk buying
        self.category_head = nn.Sequential(
            nn.Linear(d_model, hidden_half),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_half, num_categories),
        )

        # Savings regression head: estimated $/month savings from bulk buying
        self.savings_head = nn.Sequential(
            nn.Linear(d_model, hidden_half),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_half, 1),
            nn.Softplus(),   # savings must be non-negative
        )

        self._init_weights()

    def _init_weights(self):
        for module in [self.bulk_head, self.category_head, self.savings_head]:
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
        dict:
          "bulk_logit"      (batch,)            raw binary score (sigmoid for prob)
          "category_logits" (batch, num_cat)    category targeting scores
          "savings_estimate" (batch,)           predicted $/month savings
        """
        cls = self.encoder.get_cls(x)          # (batch, d_model)

        bulk_logit      = self.bulk_head(cls).squeeze(-1)           # (B,)
        category_logits = self.category_head(cls)                   # (B, C)
        savings         = self.savings_head(cls).squeeze(-1)        # (B,)

        return {
            "bulk_logit":       bulk_logit,
            "category_logits":  category_logits,
            "savings_estimate": savings,
        }

    def loss(
        self,
        out: dict[str, Tensor],
        bulk_labels:     Tensor,
        category_labels: Tensor | None = None,
        savings_targets: Tensor | None = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        out              : forward() output dict
        bulk_labels      : (batch,) float 0/1 — is this a bulk-buy user?
        category_labels  : (batch,) long — target category index
        savings_targets  : (batch,) float — actual savings amount
        """
        bce = F.binary_cross_entropy_with_logits(
            out["bulk_logit"], bulk_labels.float()
        )
        total = bce

        if category_labels is not None:
            cat_loss = F.cross_entropy(out["category_logits"], category_labels)
            total = total + self.lambda_cat * cat_loss

        if savings_targets is not None:
            sav_loss = F.huber_loss(out["savings_estimate"], savings_targets.float())
            total = total + self.lambda_savings * sav_loss

        return total

    def predict(self, x: Tensor, threshold: float = 0.5) -> dict[str, Tensor]:
        """
        Inference pass.

        Returns
        -------
        dict:
          "bulk_prob"         (batch,)   probability of bulk-buy recommendation
          "recommend"         (batch,)   bool — above threshold
          "target_category"   (batch,)   int  — category index to target
          "category_probs"    (batch, C) float
          "savings_estimate"  (batch,)   float $/month
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(x)

        bulk_prob    = torch.sigmoid(out["bulk_logit"])
        cat_probs    = F.softmax(out["category_logits"], dim=-1)
        target_cat   = cat_probs.argmax(dim=-1)

        return {
            "bulk_prob":        bulk_prob,
            "recommend":        bulk_prob >= threshold,
            "target_category":  target_cat,
            "category_probs":   cat_probs,
            "savings_estimate": out["savings_estimate"],
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
