"""
training/train_forecast.py — Train SpendingForecastTransformer.

Steps
-----
1. Load settings from config/settings.yaml
2. Load data and build windowed tensors via DataPipeline
3. Train with ForecastTrainer (Huber loss, AMP, early stopping)
4. Save model  → models/saved/forecast_transformer.pt
5. Save preprocessor → models/saved/preprocessor.pkl
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import Settings
from data.loader import load_transactions
from data.preprocessor import TransactionPreprocessor
from data.pipeline import DataPipeline
from models.transformers.spending_forecast import SpendingForecastTransformer
from training.trainer import ForecastTrainer, select_device, make_optimizer_and_scheduler


def train(args):
    cfg = Settings.load()
    cfg.save_dir.mkdir(parents=True, exist_ok=True)

    # Apply CLI overrides
    if args.seq_len is not None:
        cfg.preprocessor.seq_len = args.seq_len
    if args.horizon is not None:
        cfg.preprocessor.forecast_horizon = args.horizon
    if args.d_model is not None:
        cfg.models["forecast"].d_model = args.d_model
    if args.nhead is not None:
        cfg.models["forecast"].nhead = args.nhead
    if args.num_layers is not None:
        cfg.models["forecast"].num_layers = args.num_layers
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.lr is not None:
        cfg.training.lr = args.lr
    if args.patience is not None:
        cfg.training.patience = args.patience

    device = select_device()
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading data (source={args.source}) …")
    df = load_transactions(
        source=args.source,
        personaledger_config=args.pl_config or cfg.dataset("personaledger").default_hf_config,
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

    # ── Build windowed tensors (user-based split to avoid leakage) ────────────
    all_users = df["user_id"].unique()
    np.random.default_rng(42).shuffle(all_users)
    n_tr = int(0.8 * len(all_users))
    n_va = int(0.1 * len(all_users))
    train_users = set(all_users[:n_tr])
    val_users   = set(all_users[n_tr: n_tr + n_va])
    test_users  = set(all_users[n_tr + n_va:])

    pipeline = DataPipeline(prep, cfg)
    print("Building windowed tensors …")
    X_tr, y_tr = pipeline.build_forecast(df[df["user_id"].isin(train_users)])
    X_va, y_va = pipeline.build_forecast(df[df["user_id"].isin(val_users)])
    X_te, y_te = pipeline.build_forecast(df[df["user_id"].isin(test_users)])
    print(f"  Windows: train={len(X_tr)}  val={len(X_va)}  test={len(X_te)}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    pin = device.type == "cuda"
    nw  = min(4, os.cpu_count() or 1)
    tcfg = cfg.training
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=tcfg.batch_size,
                              shuffle=True,  num_workers=nw, pin_memory=pin)
    val_loader   = DataLoader(TensorDataset(X_va, y_va), batch_size=tcfg.batch_size,
                              shuffle=False, num_workers=nw, pin_memory=pin)

    # ── Model ─────────────────────────────────────────────────────────────────
    mcfg = cfg.model("forecast")
    epochs = args.epochs or tcfg.forecast_epochs

    # Optionally load pretrained merchant embeddings
    pretrained_merch_emb = None
    if args.pretrained_merchant_emb:
        from models.embeddings.merchant2vec import Merchant2Vec
        m2v = Merchant2Vec.load(args.pretrained_merchant_emb)
        pretrained_merch_emb = m2v.get_embeddings()
        print(f"  Loaded Merchant2Vec embeddings from {args.pretrained_merchant_emb}")

    # Optionally warm-start from pretrained encoder
    pretrained_enc = None
    if args.pretrained_encoder:
        from models.pretraining.masked_transaction_model import MaskedTransactionModel
        pretrained_enc = MaskedTransactionModel.load_pretrained_encoder(
            args.pretrained_encoder,
            feature_dim=prep.feature_dim,
            d_model=mcfg.d_model,
            nhead=mcfg.nhead,
            num_layers=mcfg.num_layers,
        )
        print(f"  Loaded pretrained encoder from {args.pretrained_encoder}")

    # If using a pretrained encoder, merchant emb is already baked in — skip double-load
    if pretrained_enc is None and pretrained_merch_emb is not None:
        from models.transformers.base import TransactionTransformer
        pretrained_enc = TransactionTransformer(
            feature_dim=prep.feature_dim,
            d_model=mcfg.d_model,
            nhead=mcfg.nhead,
            num_layers=mcfg.num_layers,
            pretrained_merchant_emb=pretrained_merch_emb,
        )

    model = SpendingForecastTransformer(
        feature_dim=prep.feature_dim,
        d_model=mcfg.d_model,
        nhead=mcfg.nhead,
        num_layers=mcfg.num_layers,
        encoder=pretrained_enc,
    ).to(device)
    print(f"  Parameters: {model.count_parameters():,}")

    optimizer, scheduler, scaler = make_optimizer_and_scheduler(
        model, tcfg.lr, tcfg.weight_decay, epochs, tcfg.lr_eta_min_factor
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    save_path = cfg.save_dir / "forecast_transformer.pt"
    trainer = ForecastTrainer(
        model, optimizer, scheduler, scaler, device,
        save_path=save_path,
        patience=tcfg.patience,
        grad_clip=tcfg.grad_clip,
    )
    trainer.fit(train_loader, val_loader, epochs=epochs)
    trainer.load_best()

    # ── Test evaluation ───────────────────────────────────────────────────────
    model.eval()
    test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=tcfg.batch_size,
                             shuffle=False)
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds.append(model(xb.to(device)).cpu().numpy())
            trues.append(yb.numpy())

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    print(f"\nTest MAE:  ${mae:.2f}")
    print(f"Test RMSE: ${rmse:.2f}")

    joblib.dump(prep, prep_path)
    print(f"Saved → {save_path}")
    print(f"Saved → {prep_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Train SpendingForecastTransformer")
    p.add_argument("--source",    default="auto",
                   choices=["auto", "personaledger", "moneyvis", "synthetic", "db", "db+auto"])
    p.add_argument("--pl-config", default=None,
                   help="PersonaLedger HF config (default from settings.yaml)")
    p.add_argument("--max-rows",  type=int, default=None,
                   help="Max rows to load (default from settings.yaml)")
    p.add_argument("--epochs",    type=int, default=None,
                   help="Training epochs (default from settings.yaml)")
    # Preprocessor overrides
    p.add_argument("--seq-len",   type=int, default=None,
                   help="Sequence window length in days (default from settings.yaml)")
    p.add_argument("--horizon",   type=int, default=None,
                   help="Forecast horizon in days (default from settings.yaml)")
    # Model architecture overrides
    p.add_argument("--d-model",   type=int, default=None,
                   help="Transformer d_model (default from settings.yaml)")
    p.add_argument("--nhead",     type=int, default=None,
                   help="Transformer nhead (default from settings.yaml)")
    p.add_argument("--num-layers", type=int, default=None,
                   help="Transformer num_layers (default from settings.yaml)")
    # Training hyperparameter overrides
    p.add_argument("--batch-size", type=int, default=None,
                   help="Batch size (default from settings.yaml)")
    p.add_argument("--lr",        type=float, default=None,
                   help="Learning rate (default from settings.yaml)")
    p.add_argument("--patience",  type=int, default=None,
                   help="Early-stopping patience (default from settings.yaml)")
    # Pretrained merchant embeddings (Merchant2Vec)
    p.add_argument("--pretrained-merchant-emb", default=None, metavar="PATH",
                   help="Path to merchant2vec.pt. Initializes merchant embedding table.")
    # Pretrained encoder warm-start
    p.add_argument("--pretrained-encoder", default=None, metavar="PATH",
                   help="Path to pretrained_encoder.pt from pretrain_masked.py. "
                        "Initializes the transformer encoder before task fine-tuning.")
    # Ensemble training: train N independent models with different seeds
    p.add_argument("--ensemble",  type=int, default=None, metavar="N",
                   help=(
                       "Train N independent models (different random seeds) and save "
                       "as forecast_transformer_0.pt … forecast_transformer_N-1.pt. "
                       "The API will detect these and use ensemble averaging at inference."
                   ))
    return p.parse_args()


def train_ensemble(args, n: int) -> None:
    """
    Train `n` independent SpendingForecastTransformer models with different random
    seeds and save them as forecast_transformer_0.pt … forecast_transformer_{n-1}.pt.

    The ModelRegistry in api/app.py will detect these files and use ensemble
    averaging at inference time.
    """
    print(f"\n=== Ensemble training: {n} models ===\n")

    # Fit the preprocessor once with a stable seed before the ensemble loop so
    # all members share an identical train/val split and vocabulary.
    cfg = Settings.load()
    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    if args.seq_len is not None:    cfg.preprocessor.seq_len = args.seq_len
    if args.horizon is not None:    cfg.preprocessor.forecast_horizon = args.horizon

    df = load_transactions(
        source=args.source,
        personaledger_config=args.pl_config or cfg.dataset("personaledger").default_hf_config,
        personaledger_max_rows=args.max_rows or cfg.dataset("personaledger").default_max_rows,
    )

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
        all_u = df["user_id"].unique()
        np.random.default_rng(42).shuffle(all_u)
        prep.fit(df[df["user_id"].isin(all_u[: int(0.8 * len(all_u))])])
        joblib.dump(prep, prep_path)

    # Stable user split shared across all ensemble members
    all_users_stable = df["user_id"].unique()
    np.random.default_rng(42).shuffle(all_users_stable)

    for seed in range(n):
        print(f"\n--- Ensemble member {seed}/{n - 1} (seed={seed}) ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        cfg = Settings.load()
        cfg.save_dir.mkdir(parents=True, exist_ok=True)

        # Apply same CLI overrides as single-model training
        if args.seq_len is not None:    cfg.preprocessor.seq_len = args.seq_len
        if args.horizon is not None:    cfg.preprocessor.forecast_horizon = args.horizon
        if args.d_model is not None:    cfg.models["forecast"].d_model = args.d_model
        if args.nhead is not None:      cfg.models["forecast"].nhead = args.nhead
        if args.num_layers is not None: cfg.models["forecast"].num_layers = args.num_layers
        if args.batch_size is not None: cfg.training.batch_size = args.batch_size
        if args.lr is not None:         cfg.training.lr = args.lr
        if args.patience is not None:   cfg.training.patience = args.patience

        device = select_device()

        all_users = all_users_stable.copy()  # copy to avoid in-place aliasing across members
        rng = np.random.default_rng(seed)
        rng.shuffle(all_users)
        n_tr = int(0.8 * len(all_users))
        n_va = int(0.1 * len(all_users))
        train_users = set(all_users[:n_tr])
        val_users   = set(all_users[n_tr: n_tr + n_va])

        pipeline = DataPipeline(prep, cfg)
        X_tr, y_tr = pipeline.build_forecast(df[df["user_id"].isin(train_users)])
        X_va, y_va = pipeline.build_forecast(df[df["user_id"].isin(val_users)])

        pin = device.type == "cuda"
        nw  = min(4, os.cpu_count() or 1)
        tcfg = cfg.training
        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=tcfg.batch_size,
                                  shuffle=True,  num_workers=nw, pin_memory=pin)
        val_loader   = DataLoader(TensorDataset(X_va, y_va), batch_size=tcfg.batch_size,
                                  shuffle=False, num_workers=nw, pin_memory=pin)

        mcfg = cfg.model("forecast")
        epochs = args.epochs or tcfg.forecast_epochs
        model = SpendingForecastTransformer(
            feature_dim=prep.feature_dim,
            d_model=mcfg.d_model,
            nhead=mcfg.nhead,
            num_layers=mcfg.num_layers,
        ).to(device)

        optimizer, scheduler, scaler = make_optimizer_and_scheduler(
            model, tcfg.lr, tcfg.weight_decay, epochs, tcfg.lr_eta_min_factor
        )

        member_path = cfg.save_dir / f"forecast_transformer_{seed}.pt"
        trainer = ForecastTrainer(
            model, optimizer, scheduler, scaler, device,
            save_path=member_path,
            patience=tcfg.patience,
            grad_clip=tcfg.grad_clip,
        )
        trainer.fit(train_loader, val_loader, epochs=epochs)
        print(f"  Saved ensemble member → {member_path}")

    print(f"\nEnsemble training complete. {n} models saved to {cfg.save_dir}/")


if __name__ == "__main__":
    args = parse_args()
    if args.ensemble is not None and args.ensemble > 1:
        train_ensemble(args, args.ensemble)
    else:
        train(args)
