"""
training/train_contrastive_anomaly.py — Train ContrastiveAnomalyDetector.

Trains on NORMAL transaction windows only (is_anomaly == False) using NT-Xent
contrastive loss. After training, builds a memory bank from the validation set
for use in inference-time nearest-neighbour anomaly scoring.

Output
------
    models/saved/contrastive_anomaly.pt  — model + memory bank + threshold

Usage
-----
    python training/train_contrastive_anomaly.py --source personaledger --epochs 20
    python training/train_contrastive_anomaly.py --source synthetic --epochs 1
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
from data.preprocessor import TransactionPreprocessor
from data.pipeline import DataPipeline
from models.transformers.base import TransactionTransformer
from models.transformers.contrastive_anomaly import ContrastiveAnomalyDetector
from training.trainer import select_device


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
    print(f"  {len(df):,} rows | anomaly rate: {df['is_anomaly'].mean():.2%}")

    # ── Preprocessor ──────────────────────────────────────────────────────────
    prep_path = cfg.save_dir / "preprocessor.pkl"
    if prep_path.exists():
        prep = joblib.load(prep_path)
    else:
        pcfg = cfg.preprocessor
        prep = TransactionPreprocessor(seq_len=pcfg.seq_len)
        all_users = df["user_id"].unique()
        np.random.default_rng(42).shuffle(all_users)
        prep.fit(df[df["user_id"].isin(all_users[:int(0.8 * len(all_users))])])
        joblib.dump(prep, prep_path)

    # ── Build anomaly windows (we keep only NORMAL samples for training) ───────
    pipeline = DataPipeline(prep, cfg)
    print("Building anomaly windows …")
    X, y = pipeline.build_anomaly(df)
    print(f"  Total windows: {len(X)} | anomaly_rate: {y.float().mean():.2%}")

    # Train on normals only; validate on a mix (to compute calibration threshold)
    normal_mask = (y == 0)
    X_normal    = X[normal_mask]
    X_anom      = X[~normal_mask]

    # 80/10/10 split on normal samples
    n  = len(X_normal)
    n_val  = max(1, int(0.1 * n))
    n_test = max(1, int(0.1 * n))
    n_train = n - n_val - n_test

    idx = torch.randperm(n)
    X_tr  = X_normal[idx[:n_train]]
    X_va  = X_normal[idx[n_train: n_train + n_val]]

    print(f"  Training on {len(X_tr)} normal sequences | val={len(X_va)}")

    pin = device.type == "cuda"
    nw  = min(4, os.cpu_count() or 1)
    bs  = args.batch_size or cfg.training.batch_size

    train_loader = DataLoader(TensorDataset(X_tr), batch_size=bs,
                              shuffle=True,  num_workers=nw, pin_memory=pin)
    val_loader   = DataLoader(TensorDataset(X_va), batch_size=bs * 2,
                              shuffle=False, num_workers=nw, pin_memory=pin)

    # ── Model ─────────────────────────────────────────────────────────────────
    mcfg   = cfg.model("anomaly")
    epochs = args.epochs or 20

    encoder = TransactionTransformer(
        feature_dim=prep.feature_dim,
        d_model=mcfg.d_model,
        nhead=mcfg.nhead,
        num_layers=mcfg.num_layers,
    ).to(device)

    model = ContrastiveAnomalyDetector(
        encoder=encoder,
        projection_dim=64,
        temperature=0.07,
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
    save_path = cfg.save_dir / "contrastive_anomaly.pt"
    best_loss = float("inf")
    patience  = cfg.training.patience
    patience_ctr = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_n = 0.0, 0
        t0 = time.time()

        for (xb,) in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            with autocast(device_type=device.type, enabled=use_amp):
                # Create two independently augmented views
                view1 = model.augment(xb)
                view2 = model.augment(xb)
                z1    = model.project(view1)
                z2    = model.project(view2)
                loss  = model.nt_xent_loss(z1, z2)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(xb)
            total_n    += len(xb)

        scheduler.step()
        train_loss = total_loss / max(total_n, 1)

        # Validation: contrastive loss on val normals
        model.eval()
        val_total, val_n = 0.0, 0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                z1 = model.project(model.augment(xb))
                z2 = model.project(model.augment(xb))
                loss = model.nt_xent_loss(z1, z2)
                val_total += loss.item() * len(xb)
                val_n     += len(xb)
        val_loss = val_total / max(val_n, 1)
        elapsed  = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  {elapsed:.1f}s"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)   # temp save
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best weights
    model.load_state_dict(torch.load(save_path, map_location=device))

    # ── Build memory bank from validation normals ──────────────────────────────
    print("Building memory bank from validation normals …")
    model.fit_memory_bank(val_loader, device=device, percentile=95.0)
    print(f"  Memory bank: {model._memory_bank.shape}  "
          f"threshold: {model._memory_bank_threshold:.6f}")

    model.save(save_path)
    print(f"\nSaved → {save_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Train ContrastiveAnomalyDetector")
    p.add_argument("--source",     default="auto",
                   choices=["auto", "personaledger", "moneyvis", "synthetic", "db", "db+auto"])
    p.add_argument("--max-rows",   type=int, default=None)
    p.add_argument("--epochs",     type=int, default=None,
                   help="Training epochs (default 20)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr",         type=float, default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
