"""
training/train_anomaly.py — Train AnomalyDetectionTransformer.

Steps
-----
1. Load settings from config/settings.yaml
2. Load data (PersonaLedger identity_theft_3months has is_fraud labels)
3. Build anomaly windows via DataPipeline (auto-augments if labels absent)
4. Train with AnomalyTrainer (recon + classification loss, AMP, early stopping)
5. Compute reconstruction-score threshold on val normals (mean + 2σ)
6. Save model     → models/saved/anomaly_transformer.pt
7. Save threshold → models/saved/anomaly_threshold.pt
"""

from __future__ import annotations

import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score
import joblib

from config import Settings
from data.loader import load_transactions
from data.preprocessor import TransactionPreprocessor
from data.pipeline import DataPipeline
from models.transformers.anomaly_detection import AnomalyDetectionTransformer
from training.trainer import AnomalyTrainer, select_device, make_optimizer_and_scheduler


def train(args):
    cfg = Settings.load()
    cfg.save_dir.mkdir(parents=True, exist_ok=True)

    # Apply CLI overrides
    if args.seq_len is not None:
        cfg.preprocessor.seq_len = args.seq_len
    if args.d_model is not None:
        cfg.models["anomaly"].d_model = args.d_model
    if args.nhead is not None:
        cfg.models["anomaly"].nhead = args.nhead
    if args.num_layers is not None:
        cfg.models["anomaly"].num_layers = args.num_layers
    if args.lambda_cls is not None:
        cfg.models["anomaly"].lambda_cls = args.lambda_cls
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.lr is not None:
        cfg.training.lr = args.lr
    if args.patience is not None:
        cfg.training.patience = args.patience

    device = select_device()
    print(f"Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    # identity_theft_3months is the PersonaLedger config with is_fraud labels.
    pl_cfg = cfg.dataset("personaledger")
    print(f"Loading data (source={args.source}) …")
    df = load_transactions(
        source=args.source,
        personaledger_config=pl_cfg.default_hf_config,
        personaledger_max_rows=args.max_rows or pl_cfg.default_max_rows,
    )
    print(f"  {len(df):,} rows | anomaly rate: {df['is_anomaly'].mean():.2%}")

    # ── Preprocessor ──────────────────────────────────────────────────────────
    # Split users FIRST to prevent preprocessor leakage into test users.
    all_users = df["user_id"].unique()
    rng_split = np.random.default_rng(42)
    rng_split.shuffle(all_users)
    n_tr_users  = int(0.8 * len(all_users))
    n_val_users = int(0.1 * len(all_users))
    train_users = all_users[:n_tr_users]
    val_users   = all_users[n_tr_users: n_tr_users + n_val_users]
    test_users  = all_users[n_tr_users + n_val_users:]

    prep_path = cfg.save_dir / "preprocessor.pkl"
    if prep_path.exists():
        print("Loading existing preprocessor …")
        prep = joblib.load(prep_path)
    else:
        pcfg = cfg.preprocessor
        prep = TransactionPreprocessor(seq_len=pcfg.seq_len)
        prep.fit(df[df["user_id"].isin(train_users)])
        joblib.dump(prep, prep_path)

    # ── Build windows (user-split to avoid leakage) ───────────────────────────
    pipeline = DataPipeline(prep, cfg)
    print("Building anomaly windows …")
    X_tr, y_tr = pipeline.build_anomaly(df[df["user_id"].isin(train_users)])
    X_va, y_va = pipeline.build_anomaly(df[df["user_id"].isin(val_users)])
    X_te, y_te = pipeline.build_anomaly(df[df["user_id"].isin(test_users)])
    print(f"  Windows: train={len(X_tr)}  val={len(X_va)}  test={len(X_te)}")
    print(f"  Anomaly rate — train={y_tr.float().mean():.2%}  val={y_va.float().mean():.2%}  test={y_te.float().mean():.2%}")

    train_ds = TensorDataset(X_tr, y_tr)
    val_ds   = TensorDataset(X_va, y_va)
    test_ds  = TensorDataset(X_te, y_te)

    tcfg = cfg.training
    pin  = device.type == "cuda"
    nw   = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size,
                              shuffle=True,  num_workers=nw, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=tcfg.batch_size,
                              shuffle=False, num_workers=nw, pin_memory=pin)

    # ── Model ─────────────────────────────────────────────────────────────────
    mcfg   = cfg.model("anomaly")
    epochs = args.epochs or tcfg.epochs

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

    model  = AnomalyDetectionTransformer(
        feature_dim=prep.feature_dim,
        seq_len=prep.seq_len,
        d_model=mcfg.d_model,
        nhead=mcfg.nhead,
        num_layers=mcfg.num_layers,
        lambda_cls=mcfg.lambda_cls,
        encoder=pretrained_enc,
    ).to(device)
    print(f"  Parameters: {model.count_parameters():,}")

    optimizer, scheduler, scaler = make_optimizer_and_scheduler(
        model, tcfg.lr, tcfg.weight_decay, epochs, tcfg.lr_eta_min_factor
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    save_path = cfg.save_dir / "anomaly_transformer.pt"
    trainer = AnomalyTrainer(
        model, optimizer, scheduler, scaler, device,
        save_path=save_path,
        patience=tcfg.patience,
        grad_clip=tcfg.grad_clip,
    )
    trainer.fit(train_loader, val_loader, epochs=epochs)
    trainer.load_best()

    # ── Test evaluation ───────────────────────────────────────────────────────
    model.eval()
    test_loader = DataLoader(test_ds, batch_size=tcfg.batch_size, shuffle=False)
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb.to(device))
            cls_pred = out["anomaly_logits"].argmax(dim=-1).cpu().numpy()
            all_preds.append(cls_pred)
            all_true.append(yb.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    print(f"\nTest  F1={f1:.3f}  Precision={prec:.3f}  Recall={rec:.3f}")

    # ── Compute reconstruction-score threshold on val normals ─────────────────
    # mean + 2σ on normal samples → threshold used by API for real-time scoring
    val_loader_thresh = DataLoader(val_ds, batch_size=tcfg.batch_size, shuffle=False)
    normal_scores = []
    with torch.no_grad():
        for xb, yb in val_loader_thresh:
            out    = model(xb.to(device))
            scores = out["anomaly_score"].cpu().numpy()
            normal_scores.append(scores[yb.numpy() == 0])
    normal_scores = np.concatenate(normal_scores)
    threshold = float(normal_scores.mean() + 2.0 * normal_scores.std())

    thr_path = cfg.save_dir / "anomaly_threshold.pt"
    torch.save({"threshold": threshold}, thr_path)
    print(f"Anomaly threshold (mean+2σ on val normals): {threshold:.4f}")
    print(f"Saved → {save_path}")
    print(f"Saved → {thr_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Train AnomalyDetectionTransformer")
    p.add_argument("--source",     default="auto",
                   choices=["auto", "personaledger", "moneyvis", "synthetic", "db", "db+auto"])
    p.add_argument("--max-rows",   type=int, default=None,
                   help="Max rows to load (default from settings.yaml)")
    p.add_argument("--epochs",     type=int, default=None,
                   help="Training epochs (default from settings.yaml)")
    # Preprocessor overrides
    p.add_argument("--seq-len",    type=int, default=None,
                   help="Sequence window length in days (default from settings.yaml)")
    # Model architecture overrides
    p.add_argument("--d-model",    type=int, default=None,
                   help="Transformer d_model (default from settings.yaml)")
    p.add_argument("--nhead",      type=int, default=None,
                   help="Transformer nhead (default from settings.yaml)")
    p.add_argument("--num-layers", type=int, default=None,
                   help="Transformer num_layers (default from settings.yaml)")
    p.add_argument("--lambda-cls", type=float, default=None,
                   help="Classification loss weight (default from settings.yaml)")
    # Training hyperparameter overrides
    p.add_argument("--batch-size", type=int, default=None,
                   help="Batch size (default from settings.yaml)")
    p.add_argument("--lr",         type=float, default=None,
                   help="Learning rate (default from settings.yaml)")
    p.add_argument("--patience",   type=int, default=None,
                   help="Early-stopping patience (default from settings.yaml)")
    p.add_argument("--pretrained-encoder", default=None, metavar="PATH",
                   help="Path to pretrained_encoder.pt from pretrain_masked.py.")
    p.add_argument("--pretrained-merchant-emb", default=None, metavar="PATH",
                   help="Path to merchant2vec.pt. Initializes merchant embedding table.")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
