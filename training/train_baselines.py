"""
training/train_baselines.py — Train all baseline models and save results.

Trains
------
  GaussianNBForecaster         → models/saved/nb_forecaster.pkl
  GaussianNBAnomalyDetector    → models/saved/nb_anomaly.pkl
  XGBForecaster                → models/saved/xgb_forecaster.pkl
  LGBMAnomalyDetector          → models/saved/lgbm_anomaly.pkl
  XGBBulkBuyClassifier         → models/saved/xgb_bulkbuy.pkl
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import joblib

from config import Settings
from data.loader import load_transactions
from data.preprocessor import TransactionPreprocessor
from data.pipeline import DataPipeline
from models.baselines.naive_bayes import GaussianNBForecaster, GaussianNBAnomalyDetector
from models.baselines.gradient_boosting import XGBForecaster, LGBMAnomalyDetector, XGBBulkBuyClassifier


def train(args):
    cfg = Settings.load()
    cfg.save_dir.mkdir(parents=True, exist_ok=True)

    pl_cfg = cfg.dataset("personaledger")
    print(f"Loading data (source={args.source}) …")
    df = load_transactions(
        source=args.source,
        personaledger_max_rows=args.max_rows or pl_cfg.default_max_rows,
    )
    print(f"  {len(df):,} rows | anomaly: {df['is_anomaly'].mean():.2%}")

    # ── Shared preprocessor ────────────────────────────────────────────────────
    prep_path = cfg.save_dir / "preprocessor.pkl"
    if prep_path.exists():
        prep = joblib.load(prep_path)
        print("Loaded existing preprocessor.")
    else:
        pcfg = cfg.preprocessor
        prep = TransactionPreprocessor(seq_len=pcfg.seq_len)
        all_users = df["user_id"].unique()
        rng = np.random.default_rng(42)
        rng.shuffle(all_users)
        prep.fit(df[df["user_id"].isin(all_users[:int(0.8 * len(all_users))])])
        joblib.dump(prep, prep_path)

    pipeline = DataPipeline(prep, cfg)

    # ── Monthly features for forecasting baselines ─────────────────────────────
    print("Building monthly feature matrix …")
    X_monthly, y_monthly = prep.make_monthly_features(df)
    n = len(X_monthly)
    split = int(0.8 * n)
    X_tr, X_te = X_monthly[:split], X_monthly[split:]
    y_tr, y_te = y_monthly[:split], y_monthly[split:]
    print(f"  Monthly samples: {n}  (train={split}, test={n - split})")

    # ── Windowed features for anomaly / bulk-buy baselines ─────────────────────
    print("Building windowed tensors …")
    X_anom, y_anom = pipeline.build_anomaly(df)
    # DataPipeline.build_bulkbuy_binary auto-mixes synthetic when labels are absent
    X_bulk, y_bulk = pipeline.build_bulkbuy_binary(df)

    X_anom_np = X_anom.numpy()
    y_anom_np = y_anom.numpy()
    X_bulk_np = X_bulk.numpy()
    y_bulk_np = y_bulk.numpy()

    # Shuffle so positives aren't clustered in one split (users are grouped by ID)
    rng = np.random.RandomState(42)
    idx_a = rng.permutation(len(X_anom_np))
    X_anom_np, y_anom_np = X_anom_np[idx_a], y_anom_np[idx_a]
    idx_b = rng.permutation(len(X_bulk_np))
    X_bulk_np, y_bulk_np = X_bulk_np[idx_b], y_bulk_np[idx_b]

    n_a = len(X_anom_np); sp_a = int(0.8 * n_a)
    n_b = len(X_bulk_np); sp_b = int(0.8 * n_b)

    results = {}

    # ── 1. Naive Bayes Forecaster ──────────────────────────────────────────────
    print("\n--- Naive Bayes Forecaster ---")
    nb_fc = GaussianNBForecaster()
    nb_fc.fit(X_tr, y_tr)
    nb_fc.save(str(cfg.save_dir / "nb_forecaster.pkl"))
    m = nb_fc.evaluate(X_te, y_te)
    print(f"  Overall MAE: ${m['overall_mae']:.2f}")
    results["nb_forecaster"] = {"mae": m["overall_mae"]}

    # ── 2. Naive Bayes Anomaly Detector ───────────────────────────────────────
    print("\n--- Naive Bayes Anomaly Detector ---")
    nb_ad = GaussianNBAnomalyDetector()
    nb_ad.fit(X_anom_np[:sp_a], y_anom_np[:sp_a])
    nb_ad.save(str(cfg.save_dir / "nb_anomaly.pkl"))
    m = nb_ad.evaluate(X_anom_np[sp_a:], y_anom_np[sp_a:])
    print(f"  F1={m['f1']:.3f}  Precision={m['precision']:.3f}  Recall={m['recall']:.3f}")
    results["nb_anomaly"] = {"f1": m["f1"], "precision": m["precision"], "recall": m["recall"]}

    # ── 3. XGBoost Forecaster ─────────────────────────────────────────────────
    print("\n--- XGBoost Forecaster ---")
    xgb_fc = XGBForecaster()
    xgb_fc.fit(X_tr, y_tr)
    xgb_fc.save(str(cfg.save_dir / "xgb_forecaster.pkl"))
    m = xgb_fc.evaluate(X_te, y_te)
    print(f"  Overall MAE: ${m['overall_mae']:.2f}  RMSE: ${m['overall_rmse']:.2f}")
    results["xgb_forecaster"] = {"mae": m["overall_mae"], "rmse": m["overall_rmse"]}

    # ── 4. LightGBM Anomaly Detector ──────────────────────────────────────────
    print("\n--- LightGBM Anomaly Detector ---")
    lgbm_ad = LGBMAnomalyDetector()
    lgbm_ad.fit(X_anom_np[:sp_a], y_anom_np[:sp_a])
    lgbm_ad.save(str(cfg.save_dir / "lgbm_anomaly.pkl"))
    m = lgbm_ad.evaluate(X_anom_np[sp_a:], y_anom_np[sp_a:])
    print(f"  F1={m['f1']:.3f}  Precision={m['precision']:.3f}  Recall={m['recall']:.3f}")
    results["lgbm_anomaly"] = {"f1": m["f1"], "precision": m["precision"], "recall": m["recall"]}

    # ── 5. XGBoost Bulk-Buy Classifier ────────────────────────────────────────
    print("\n--- XGBoost Bulk-Buy Classifier ---")
    xgb_bb = XGBBulkBuyClassifier()
    xgb_bb.fit(X_bulk_np[:sp_b], y_bulk_np[:sp_b])
    xgb_bb.save(str(cfg.save_dir / "xgb_bulkbuy.pkl"))
    m = xgb_bb.evaluate(X_bulk_np[sp_b:], y_bulk_np[sp_b:])
    print(f"  F1={m['f1']:.3f}  Precision={m['precision']:.3f}  Recall={m['recall']:.3f}")
    results["xgb_bulkbuy"] = {"f1": m["f1"], "precision": m["precision"], "recall": m["recall"]}

    out_path = cfg.save_dir / "baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBaseline results saved → {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Train all baseline models")
    p.add_argument("--source",   default="auto",
                   choices=["auto", "personaledger", "moneyvis", "synthetic", "db", "db+auto"])
    p.add_argument("--max-rows", type=int, default=None,
                   help="Max rows to load (default from settings.yaml)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
