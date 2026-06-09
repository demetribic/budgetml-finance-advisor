"""
models/user_vae.py — Variational Autoencoder for user spending representations.

Encodes a user's 60-day transaction sequence into a 64-dim latent user embedding
via a beta-VAE. The latent space enables:
  - Cohort comparison: find users with similar financial behavior
  - Financial archetype classification: k-means cluster labels
  - Cold-start bootstrapping: infer likely future behavior from similar users
  - Financial drift detection: track how a user's embedding moves over time
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.transformers.base import TransactionTransformer


class UserSpendingVAE(nn.Module):
    """
    VAE that encodes a user's transaction sequence into a 64-dim latent user embedding.

    Encoder : TransactionTransformer → [CLS] → mu (64,) + log_var (64,)
    Decoder : Linear(64) → GELU → Linear(64, num_categories × seq_len)
              reconstructs daily category spend (not raw transactions)
    Loss    : ELBO = reconstruction (MSE on category spend) + beta × KL divergence

    Parameters
    ----------
    encoder        : TransactionTransformer   shared or dedicated backbone
    num_categories : int   number of spending categories (default 9)
    seq_len        : int   transaction window length (default 60)
    beta           : float weight on KL divergence term (beta-VAE; default 1.0)
    """

    LATENT_DIM = 64

    def __init__(
        self,
        encoder:        TransactionTransformer,
        num_categories: int   = 9,
        seq_len:        int   = 60,
        beta:           float = 1.0,
    ):
        super().__init__()
        self.encoder        = encoder
        self.num_categories = num_categories
        self.seq_len        = seq_len
        self.beta           = beta
        d_model             = encoder.d_model
        latent              = self.LATENT_DIM

        # Encoder heads: CLS → mu and log_var
        self.mu_head      = nn.Linear(d_model, latent)
        self.log_var_head = nn.Linear(d_model, latent)

        # Decoder: latent → daily category spend (seq_len × num_categories)
        decode_dim = num_categories * seq_len
        self.decoder = nn.Sequential(
            nn.Linear(latent, latent * 2),
            nn.GELU(),
            nn.Linear(latent * 2, decode_dim),
            nn.Softplus(),   # spend is non-negative
        )

        self._init_weights()

    def _init_weights(self):
        for m in [self.mu_head, self.log_var_head]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        for m in self.decoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Encode a transaction sequence to VAE latent parameters.

        Parameters
        ----------
        x : (batch, seq_len, feature_dim)

        Returns
        -------
        mu      : (batch, LATENT_DIM)
        log_var : (batch, LATENT_DIM)
        """
        cls = self.encoder.get_cls(x)         # (B, d_model)
        mu      = self.mu_head(cls)            # (B, latent)
        log_var = self.log_var_head(cls)       # (B, latent)
        return mu, log_var

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Reparameterization trick: z = mu + eps * std.
        During eval(), returns mu directly (deterministic embedding).
        """
        if not self.training:
            return mu
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode latent vector to reconstructed daily category spend.

        Parameters
        ----------
        z : (batch, LATENT_DIM)

        Returns
        -------
        Tensor : (batch, seq_len, num_categories)
        """
        flat = self.decoder(z)                          # (B, seq_len × num_categories)
        return flat.view(-1, self.seq_len, self.num_categories)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """
        Full VAE forward pass.

        Returns
        -------
        dict:
          "z"       (batch, LATENT_DIM)                      — sampled latent
          "mu"      (batch, LATENT_DIM)                      — posterior mean
          "log_var" (batch, LATENT_DIM)                      — posterior log variance
          "recon"   (batch, seq_len, num_categories)         — reconstructed spend
        """
        mu, log_var = self.encode(x)
        z    = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return {"z": z, "mu": mu, "log_var": log_var, "recon": recon}

    def loss(self, out: dict[str, Tensor], target_category_spend: Tensor) -> Tensor:
        """
        ELBO loss = reconstruction MSE + beta × KL divergence.

        Parameters
        ----------
        out                   : forward() output dict
        target_category_spend : (batch, seq_len, num_categories) ground-truth daily spend

        Returns
        -------
        scalar loss tensor
        """
        recon_loss = F.mse_loss(out["recon"], target_category_spend)

        # KL divergence: D_KL(N(mu, var) || N(0, 1))
        # = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl = -0.5 * (1 + out["log_var"] - out["mu"].pow(2) - out["log_var"].exp())
        kl_loss = kl.sum(dim=-1).mean()   # mean over batch

        return recon_loss + self.beta * kl_loss

    def get_user_embedding(self, x: Tensor) -> Tensor:
        """
        Inference: return the deterministic latent embedding (mu).

        Parameters
        ----------
        x : (batch, seq_len, feature_dim)

        Returns
        -------
        Tensor : (batch, LATENT_DIM)
        """
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu

    def nearest_cohort(
        self,
        z:                 Tensor,
        cohort_embeddings: Tensor,
        top_k:             int = 5,
    ) -> tuple[Tensor, Tensor]:
        """
        Return the top_k nearest users in a cohort embedding matrix.

        Parameters
        ----------
        z                 : (batch, LATENT_DIM) or (LATENT_DIM,)
        cohort_embeddings : (N_users, LATENT_DIM)
        top_k             : int

        Returns
        -------
        indices    : (batch, top_k) — cohort user indices
        similarities : (batch, top_k) — cosine similarities
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        z_norm     = F.normalize(z, dim=-1)                     # (B, D)
        coh_norm   = F.normalize(cohort_embeddings, dim=-1)     # (N, D)
        sims       = z_norm @ coh_norm.T                        # (B, N)
        top        = sims.topk(top_k, dim=-1)
        return top.indices, top.values

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
