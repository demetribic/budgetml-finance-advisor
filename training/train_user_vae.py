"""
training/train_user_vae.py — Train UserSpendingVAE and compute cohort embeddings.

Steps
-----
1. Load data and preprocessor
2. Build (X, y_daily_cat_spend) pairs where X is the 60-day feature window
   and y is the daily category spend matrix for the same window
3. Train the VAE (ELBO loss)
4. Save model → models/saved/user_vae.pt
5. Compute embeddings for all training users → models/saved/user_cohort_embeddings.pt
6. Fit k-means (k=8) on cohort embeddings, save cluster centers + labels
   → models/saved/user_archetypes.pt

Usage
-----
    python training/train_user_vae.py --source personaledger --epochs 20
    python training/train_user_vae.py --source synthetic --epochs 1  # smoke test
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
import joblib

from config import Settings
from data.loader import load_transactions
from data.preprocessor import TransactionPreprocessor, CATEGORIES
from data.pipeline import DataPipeline
from models.transformers.base import TransactionTransformer
from models.user_vae import UserSpendingVAE
from training.trainer import select_device


# ── Build target: daily category spend matrix ─────────────────────────────────

def _build_daily_cat_spend_windows(
    df,
    prep: TransactionPreprocessor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each user, build (X, y) pairs where:
      X : (seq_len, feature_dim) transaction feature window
      y : (seq_len, num_categories) daily category spend for the same window

    Returns (X_all, y_all) with shapes (N, seq_len, feature_dim) and
    (N, seq_len, num_categories).
    """
    import pandas as pd

    seq_len = prep.seq_len
    X_list, y_list = [], []

    for uid, udf in df.groupby("user_id"):
        daily = prep._build_daily_series(udf)
        n = len(daily)
        if n < seq_len:
            continue

        # Feature matrix
        feat = prep._encode_df_fast(daily, user_id=uid)   # (n, feat_dim)

        # Daily category spend
        udf_copy = udf.copy()
        udf_copy["date"] = pd.to_datetime(udf_copy["date"]).dt.normalize()
        cat_spend = (
            udf_copy.groupby(["date", "category"])["amount"]
            .sum()
            .unstack(fill_value=0.0)
            .reindex(columns=CATEGORIES, fill_value=0.0)
        )
        cat_aligned = (
            cat_spend.reindex(daily["date"], fill_value=0.0)
            .fillna(0.0)
            .values
            .astype(np.float32)
        )   # (n, num_categories)

        n_windows = n - seq_len + 1
        for i in range(n_windows):
            X_list.append(feat[i: i + seq_len])
            y_list.append(cat_aligned[i: i + seq_len])

    if not X_list:
        fd = prep.feature_dim
        return (
            torch.zeros(0, prep.seq_len, fd),
            torch.zeros(0, prep.seq_len, len(CATEGORIES)),
        )

    X = torch.from_numpy(np.stack(X_list).astype(np.float32))
    y = torch.from_numpy(np.stack(y_list).astype(np.float32))
    return X, y


# ── k-means (pure torch, no sklearn dependency for GPU compat) ────────────────

def _kmeans(embeddings: torch.Tensor, k: int, n_iter: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simple Lloyd's k-means on CPU.

    Returns
    -------
    centers : (k, D)
    labels  : (N,) int64
    """
    N, D = embeddings.shape
    perm    = torch.randperm(N)[:k]
    centers = embeddings[perm].clone()

    for _ in range(n_iter):
        dists  = torch.cdist(embeddings, centers)       # (N, k)
        labels = dists.argmin(dim=-1)                   # (N,)
        new_centers = torch.zeros_like(centers)
        for j in range(k):
            mask = (labels == j)
            if mask.any():
                new_centers[j] = embeddings[mask].mean(dim=0)
            else:
                new_centers[j] = centers[j]
        if (new_centers - centers).norm() < 1e-6:
            break
        centers = new_centers

    return centers, labels


_ARCHETYPE_NAMES = [
    "Balanced Saver",
    "Dining-Heavy",
    "Subscription Heavy",
    "Transport Commuter",
    "Healthcare Focused",
    "Entertainment Enthusiast",
    "Grocery Optimizer",
    "Impulse Spender",
]


def train(args) -> None:
    cfg = Settings.load()
    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    device = select_device()
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading data (source={args.source}) …")
    pl_cfg = cfg.dataset("personaledger")
    df = load_transactions(
        source=args.source,
        personaledger_config=pl_cfg.default_hf_config,
        personaledger_max_rows=args.max_rows or pl_cfg.default_max_rows,
    )
    print(f"  {len(df):,} transactions, {df['user_id'].nunique()} users")

    # ── Preprocessor ──────────────────────────────────────────────────────────
    prep_path = cfg.save_dir / "preprocessor.pkl"
    if prep_path.exists():
        prep = joblib.load(prep_path)
    else:
        pcfg = cfg.preprocessor
        prep = TransactionPreprocessor(
            seq_len=pcfg.seq_len,
            forecast_horizon=pcfg.forecast_horizon,
            max_merchants=pcfg.max_merchants,
        )
        all_users = df["user_id"].unique()
        np.random.default_rng(42).shuffle(all_users)
        prep.fit(df[df["user_id"].isin(all_users[:int(0.8 * len(all_users))])])
        joblib.dump(prep, prep_path)

    # ── Build windows ─────────────────────────────────────────────────────────
    all_users = df["user_id"].unique()
    np.random.default_rng(42).shuffle(all_users)
    n_tr = int(0.8 * len(all_users))
    n_va = int(0.1 * len(all_users))
    train_users = set(all_users[:n_tr])
    val_users   = set(all_users[n_tr: n_tr + n_va])

    print("Building VAE training windows …")
    X_tr, y_tr = _build_daily_cat_spend_windows(df[df["user_id"].isin(train_users)], prep)
    X_va, y_va = _build_daily_cat_spend_windows(df[df["user_id"].isin(val_users)],   prep)
    print(f"  Windows: train={len(X_tr)}  val={len(X_va)}")

    pin = device.type == "cuda"
    nw  = min(4, os.cpu_count() or 1)
    bs  = args.batch_size or cfg.training.batch_size
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=bs,
                              shuffle=True,  num_workers=nw, pin_memory=pin)
    val_loader   = DataLoader(TensorDataset(X_va, y_va), batch_size=bs,
                              shuffle=False, num_workers=nw, pin_memory=pin)

    # ── Model ─────────────────────────────────────────────────────────────────
    mcfg   = cfg.model("forecast")
    epochs = args.epochs or 20

    encoder = TransactionTransformer(
        feature_dim=prep.feature_dim,
        d_model=mcfg.d_model,
        nhead=mcfg.nhead,
        num_layers=mcfg.num_layers,
    ).to(device)

    model = UserSpendingVAE(
        encoder=encoder,
        num_categories=len(CATEGORIES),
        seq_len=prep.seq_len,
        beta=args.beta or 1.0,
    ).to(device)
    print(f"  Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr or cfg.training.lr, weight_decay=1e-2
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=(args.lr or cfg.training.lr) * 0.01
    )
    use_amp = device.type == "cuda"
    scaler  = GradScaler(device=device.type, enabled=use_amp)

    # ── Training loop ──────────────────────────────────────────────────────────
    save_path  = cfg.save_dir / "user_vae.pt"
    best_val   = float("inf")
    patience   = cfg.training.patience
    patience_ctr = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_n = 0.0, 0
        t0 = time.time()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            with autocast(device_type=device.type, enabled=use_amp):
                out  = model(xb)
                loss = model.loss(out, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(xb)
            total_n    += len(xb)
        scheduler.step()
        train_loss = total_loss / max(total_n, 1)

        model.eval()
        val_total, val_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out  = model(xb)
                loss = model.loss(out, yb)
                val_total += loss.item() * len(xb)
                val_n     += len(xb)
        val_loss = val_total / max(val_n, 1)
        elapsed  = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  {elapsed:.1f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "state_dict":    model.state_dict(),
                "feature_dim":   prep.feature_dim,
                "seq_len":       prep.seq_len,
                "num_categories": len(CATEGORIES),
                "d_model":       mcfg.d_model,
                "nhead":         mcfg.nhead,
                "num_layers":    mcfg.num_layers,
                "beta":          model.beta,
            }, save_path)
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nSaved → {save_path}")

    # ── Compute cohort embeddings for all training users ───────────────────────
    print("Computing cohort embeddings …")
    model.eval()
    all_embs = []
    with torch.no_grad():
        for (xb, _) in DataLoader(TensorDataset(X_tr, y_tr), batch_size=bs * 4):
            emb = model.get_user_embedding(xb.to(device))
            all_embs.append(emb.cpu())
    cohort_embs = torch.cat(all_embs, dim=0)   # (N_windows, 64)
    print(f"  Cohort embeddings shape: {cohort_embs.shape}")

    cohort_path = cfg.save_dir / "user_cohort_embeddings.pt"
    torch.save(cohort_embs, cohort_path)
    print(f"  Saved → {cohort_path}")

    # ── k-means clustering → financial archetypes ─────────────────────────────
    print("Running k-means (k=8) for financial archetypes …")
    k = min(8, len(cohort_embs))
    centers, labels = _kmeans(cohort_embs, k=k)
    archetype_names = _ARCHETYPE_NAMES[:k]

    archetype_path = cfg.save_dir / "user_archetypes.pt"
    torch.save({
        "centers":  centers,
        "labels":   labels,
        "names":    archetype_names,
        "k":        k,
    }, archetype_path)
    print(f"  Archetype distribution: { {n: int((labels == i).sum()) for i, n in enumerate(archetype_names)} }")
    print(f"  Saved → {archetype_path}")

    # ── Per-cluster spending statistics for peer comparison ───────────────────
    print("Computing cohort spend statistics …")
    # y_tr shape: (N_windows, seq_len, num_categories) — daily category spend
    # Approximate 30-day monthly spend: sum over the 60-day window / 2
    window_monthly = y_tr.sum(dim=1) / 2.0          # (N_windows, num_categories)
    cohort_stats: dict = {}
    for c in range(k):
        mask = (labels == c).numpy()
        if not mask.any():
            continue
        cluster_spend = window_monthly[mask].numpy()  # (n_c, num_categories)
        cohort_stats[c] = {}
        for j, cat in enumerate(CATEGORIES):
            vals = cluster_spend[:, j]
            cohort_stats[c][cat] = {
                "mean":   float(vals.mean()),
                "median": float(np.median(vals)),
                "p25":    float(np.percentile(vals, 25)),
                "p75":    float(np.percentile(vals, 75)),
            }

    stats_path = cfg.save_dir / "cohort_spend_stats.pt"
    torch.save(cohort_stats, stats_path)
    print(f"  Saved cohort spend stats ({k} clusters) → {stats_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Train UserSpendingVAE")
    p.add_argument("--source",     default="auto",
                   choices=["auto", "personaledger", "moneyvis", "synthetic", "db", "db+auto"])
    p.add_argument("--max-rows",   type=int, default=None)
    p.add_argument("--epochs",     type=int, default=None,
                   help="Training epochs (default 20)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--beta",       type=float, default=None,
                   help="Beta-VAE KL weight (default 1.0)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
