"""
training/pretrain_masked.py — Self-supervised masked transaction pretraining.

Trains MaskedTransactionModel on raw transaction sequences (no task labels needed).
The learned encoder weights are saved for use as a warm-start in task fine-tuning.

Usage
-----
    python training/pretrain_masked.py --source personaledger --epochs 20
    python training/pretrain_masked.py --source synthetic --epochs 1  # smoke test

Output
------
    models/saved/pretrained_encoder.pt   — encoder state dict for warm-start
"""

from __future__ import annotations

import os
import argparse
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
from models.pretraining.masked_transaction_model import MaskedTransactionModel
from training.trainer import select_device, make_optimizer_and_scheduler


def pretrain(args) -> None:
    cfg = Settings.load()
    cfg.save_dir.mkdir(parents=True, exist_ok=True)

    device = select_device()
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading data (source={args.source}) …")
    df = load_transactions(
        source=args.source,
        personaledger_config=cfg.dataset("personaledger").default_hf_config,
        personaledger_max_rows=args.max_rows or cfg.dataset("personaledger").default_max_rows,
    )
    print(f"  {len(df):,} transactions, {df['user_id'].nunique()} users")

    # ── Preprocessor ──────────────────────────────────────────────────────────
    prep_path = cfg.save_dir / "preprocessor.pkl"
    if prep_path.exists():
        print("Loading existing preprocessor …")
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
        n_train = int(0.8 * len(all_users))
        prep.fit(df[df["user_id"].isin(all_users[:n_train])])
        joblib.dump(prep, prep_path)
        print(f"  Fitted preprocessor → {prep_path}")

    # ── Build windowed tensors (raw sequences — no labels needed) ─────────────
    all_users = df["user_id"].unique()
    np.random.default_rng(42).shuffle(all_users)
    n_tr = int(0.8 * len(all_users))
    n_va = int(0.1 * len(all_users))
    train_users = set(all_users[:n_tr])
    val_users   = set(all_users[n_tr: n_tr + n_va])

    pipeline = DataPipeline(prep, cfg)
    print("Building windowed tensors (forecast windows reused for pretraining) …")
    # Reuse forecast window builder — we only need X (ignore labels)
    X_tr, _ = pipeline.build_forecast(df[df["user_id"].isin(train_users)])
    X_va, _ = pipeline.build_forecast(df[df["user_id"].isin(val_users)])
    print(f"  Windows: train={len(X_tr)}  val={len(X_va)}")

    pin = device.type == "cuda"
    nw  = min(4, os.cpu_count() or 1)
    bs  = args.batch_size or cfg.training.batch_size
    train_loader = DataLoader(TensorDataset(X_tr), batch_size=bs,
                              shuffle=True,  num_workers=nw, pin_memory=pin)
    val_loader   = DataLoader(TensorDataset(X_va), batch_size=bs,
                              shuffle=False, num_workers=nw, pin_memory=pin)

    # ── Model ─────────────────────────────────────────────────────────────────
    mcfg   = cfg.model("forecast")   # reuse architecture hyperparams
    epochs = args.epochs or 20

    encoder = TransactionTransformer(
        feature_dim=prep.feature_dim,
        d_model=mcfg.d_model,
        nhead=mcfg.nhead,
        num_layers=mcfg.num_layers,
    ).to(device)

    model = MaskedTransactionModel(
        encoder=encoder,
        mask_prob=0.15,
        num_categories=9,
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

    # ── Pretraining loop ──────────────────────────────────────────────────────
    best_val  = float("inf")
    save_path = cfg.save_dir / "pretrained_encoder.pt"
    patience  = cfg.training.patience
    patience_ctr = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss, total_n = 0.0, 0
        t0 = time.time()
        for (xb,) in train_loader:
            xb = xb.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast(device_type=device.type, enabled=use_amp):
                out  = model(xb)
                loss = model.loss(out, xb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(xb)
            total_n    += len(xb)
        train_loss = total_loss / max(total_n, 1)
        scheduler.step()

        # Validate
        model.eval()
        val_total, val_n = 0.0, 0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                with autocast(device_type=device.type, enabled=use_amp):
                    out  = model(xb)
                    loss = model.loss(out, xb)
                val_total += loss.item() * len(xb)
                val_n     += len(xb)
        val_loss = val_total / max(val_n, 1)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  {elapsed:.1f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            model.save_pretrained_encoder(save_path)
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nPretrained encoder saved → {save_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Masked transaction pretraining")
    p.add_argument("--source",     default="auto",
                   choices=["auto", "personaledger", "moneyvis", "synthetic", "db", "db+auto"])
    p.add_argument("--max-rows",   type=int, default=None)
    p.add_argument("--epochs",     type=int, default=None,
                   help="Pretraining epochs (default 20)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr",         type=float, default=None)
    return p.parse_args()


if __name__ == "__main__":
    pretrain(parse_args())
