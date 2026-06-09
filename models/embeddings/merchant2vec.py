"""
models/embeddings/merchant2vec.py — Skip-gram Merchant2Vec embeddings.

Treats each user's ordered transaction history as a sequence of merchant "words"
and trains a Word2Vec-style skip-gram model with negative sampling. The resulting
32-dim embeddings capture merchant co-occurrence patterns (e.g., grocery stores
cluster together, coffee shops cluster together, etc.).

After training, the embedding matrix is used to initialize the merchant
nn.Embedding table in HybridInputProjection instead of random initialization.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Merchant2Vec(nn.Module):
    """
    Skip-gram model for merchant embeddings trained with negative sampling.

    Parameters
    ----------
    vocab_size : int   number of merchants in vocabulary (max_merchants from preprocessor)
    emb_dim    : int   embedding dimension — must match HybridInputProjection.merch_emb_dim
    """

    def __init__(self, vocab_size: int, emb_dim: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim    = emb_dim

        # Center (target) and context embeddings — standard skip-gram split
        self.center_emb  = nn.Embedding(vocab_size, emb_dim, sparse=True)
        self.context_emb = nn.Embedding(vocab_size, emb_dim, sparse=True)

        nn.init.uniform_(self.center_emb.weight,  -0.5 / emb_dim, 0.5 / emb_dim)
        nn.init.zeros_(self.context_emb.weight)

    def forward(
        self,
        center:    Tensor,   # (batch,)        center merchant indices
        context:   Tensor,   # (batch,)        positive context indices
        negatives: Tensor,   # (batch, k)      negative sample indices
    ) -> Tensor:
        """
        Negative sampling loss for one batch.

        Returns scalar loss (mean over batch).
        """
        center_v  = self.center_emb(center)                # (B, D)
        context_v = self.context_emb(context)              # (B, D)
        neg_v     = self.context_emb(negatives)            # (B, k, D)

        # Positive score
        pos_score = (center_v * context_v).sum(dim=-1)    # (B,)
        pos_loss  = F.logsigmoid(pos_score)

        # Negative scores
        neg_score = torch.bmm(neg_v, center_v.unsqueeze(-1)).squeeze(-1)  # (B, k)
        neg_loss  = F.logsigmoid(-neg_score).sum(dim=-1)  # (B,)

        return -(pos_loss + neg_loss).mean()

    def get_embeddings(self) -> Tensor:
        """Return embedding matrix (vocab_size, emb_dim) for weight transfer."""
        return self.center_emb.weight.detach()

    def most_similar(
        self,
        merchant_id: int,
        top_k:       int = 10,
    ) -> list[tuple[int, float]]:
        """
        Return top_k most similar merchant indices by cosine similarity.

        Returns
        -------
        list of (merchant_id, cosine_similarity) tuples sorted descending
        """
        embs   = F.normalize(self.center_emb.weight, dim=-1)  # (V, D)
        query  = embs[merchant_id].unsqueeze(0)                # (1, D)
        sims   = (embs @ query.T).squeeze(-1)                  # (V,)
        sims[merchant_id] = -2.0   # exclude self
        top    = sims.topk(top_k)
        return [(int(idx), float(sim)) for idx, sim in zip(top.indices, top.values)]

    def save(self, path: Path | str) -> None:
        """Save model state dict."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "vocab_size":  self.vocab_size,
            "emb_dim":     self.emb_dim,
        }, str(path))

    @classmethod
    def load(cls, path: Path | str) -> "Merchant2Vec":
        """Load a saved Merchant2Vec checkpoint."""
        ckpt = torch.load(str(path), map_location="cpu")
        model = cls(vocab_size=ckpt["vocab_size"], emb_dim=ckpt["emb_dim"])
        model.load_state_dict(ckpt["state_dict"])
        return model

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
