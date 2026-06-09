"""
data/pipeline.py — Centralised dataset building for each ML task.

DataPipeline wraps a fitted TransactionPreprocessor and builds task-specific
(X, y) tensors.  Crucially, it checks whether the DataFrame actually contains
positive labels for each task and automatically mixes in synthetic data when
it does not — so no training script ever needs to re-implement that logic.

    # Before: scattered across train_baselines.py, train_bulk_buy.py, evaluate.py
    df_synth = generate_synthetic_data(num_users=200)
    df_src = pd.concat([df, df_synth], ignore_index=True)

    # After: one call, same behaviour everywhere
    pipeline = DataPipeline(prep, cfg)
    X, y = pipeline.build_bulkbuy_binary(df)

Public API
----------
    build_forecast(df)          → (X, y)                 spending forecast
    build_anomaly(df)           → (X, y)                 anomaly detection
    build_bulkbuy_binary(df)    → (X, y)                 baselines + eval
    build_bulkbuy_multitask(df) → (X, y_bulk, y_cat, y_savings)  transformer
"""

from __future__ import annotations

import pandas as pd
import torch
from torch import Tensor

from data.preprocessor import TransactionPreprocessor, CATEGORIES
from config import Settings


class DataPipeline:
    """
    Centralised dataset builder.

    Parameters
    ----------
    prep : TransactionPreprocessor
        A fitted preprocessor instance.
    cfg : Settings
        Project-wide settings (loaded from config/settings.yaml).
    """

    def __init__(self, prep: TransactionPreprocessor, cfg: Settings):
        self.prep = prep
        self.cfg = cfg

    # ── Public build methods ───────────────────────────────────────────────────

    def build_forecast(self, df: pd.DataFrame) -> tuple[Tensor, Tensor]:
        """(X, y) windows for spending forecast — no augmentation needed."""
        return self.prep.make_forecast_windows(df)

    def build_anomaly(self, df: pd.DataFrame) -> tuple[Tensor, Tensor]:
        """
        (X, y) anomaly-detection windows.
        Augments with synthetic data if the DataFrame has no anomaly labels.
        """
        aug = self.cfg.synthetic_augmentation
        if aug.anomaly_num_users > 0 and not df["is_anomaly"].any():
            df = self._augment(df, aug.anomaly_num_users)
        return self.prep.make_anomaly_windows(df)

    def build_bulkbuy_binary(self, df: pd.DataFrame) -> tuple[Tensor, Tensor]:
        """
        (X, y) bulk-buy windows with a single binary label per window.
        Used by baseline models and evaluation.
        Augments with synthetic data if the DataFrame has no bulk-buy labels.
        """
        df = self._ensure_bulkbuy_labels(df)
        return self.prep.make_bulkbuy_windows(df)

    def build_bulkbuy_multitask(
        self, df: pd.DataFrame
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        (X, y_bulk, y_cat, y_savings) for BulkBuyRecommendationTransformer.

        Uses actual is_bulk_buy labels when available, otherwise falls back to
        frequency-based inference.  Augments with synthetic data if needed.
        """
        df = self._ensure_bulkbuy_labels(df)
        return _build_bulkbuy_multitask(df, self.prep)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _ensure_bulkbuy_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append synthetic data if df has no positive bulk-buy labels."""
        aug = self.cfg.synthetic_augmentation
        if aug.bulk_buy_num_users > 0 and not df["is_bulk_buy"].any():
            df = self._augment(df, aug.bulk_buy_num_users)
        return df

    def _augment(self, df: pd.DataFrame, num_users: int) -> pd.DataFrame:
        from data.synthetic import generate_synthetic_data
        df_synth = generate_synthetic_data(num_users=num_users)
        return pd.concat([df, df_synth], ignore_index=True)


# ── Bulk-buy multi-task helpers ────────────────────────────────────────────────
# (Moved from training/train_bulk_buy.py so all callers share one implementation)

def _infer_bulk_buy_label(df_window: pd.DataFrame) -> tuple[int, int, float]:
    """
    For datasets without is_bulk_buy labels, infer bulk-buy potential from
    purchase frequency: flag if any merchant appears >= 3x in the window.

    Returns (bulk_label, category_idx, estimated_monthly_savings).
    """
    merchant_counts = df_window["merchant"].value_counts()
    if merchant_counts.empty or merchant_counts.iloc[0] < 3:
        return 0, len(CATEGORIES) - 1, 0.0

    top_merchant = merchant_counts.index[0]
    top_cat = df_window[df_window["merchant"] == top_merchant]["category"].mode()
    cat_name = top_cat.iloc[0] if not top_cat.empty else "other"
    cat_idx = CATEGORIES.index(cat_name) if cat_name in CATEGORIES else len(CATEGORIES) - 1

    avg_price = float(df_window[df_window["merchant"] == top_merchant]["amount"].mean())
    freq = float(merchant_counts.iloc[0])
    days_in_window = max((df_window["date"].max() - df_window["date"].min()).days + 1, 1)
    monthly_freq = freq * (30.0 / days_in_window)
    savings = avg_price * monthly_freq * 0.20   # 20% bulk-discount estimate

    return 1, cat_idx, savings


def _build_bulkbuy_multitask(
    df: pd.DataFrame, prep: TransactionPreprocessor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Build (X, y_bulk, y_cat, y_savings) tensors for multi-task bulk-buy training.

    Uses actual is_bulk_buy labels when the column is present and has positives;
    otherwise falls back to frequency-based inference per window.
    """
    has_label = "is_bulk_buy" in df.columns and df["is_bulk_buy"].any()
    X_win, y_bulk, y_cat, y_sav = [], [], [], []

    for uid, udf in df.groupby("user_id"):
        daily = prep._build_daily_series(udf)
        n = len(daily)

        for start in range(n - prep.seq_len):
            window = daily.iloc[start: start + prep.seq_len]
            feats = [prep._encode_row(row) for _, row in window.iterrows()]
            X_win.append(feats)

            if has_label:
                bulk_label = int(window["is_bulk_buy"].any())
                cat_name = window["category"].mode().iloc[0]
                cat_idx = (
                    CATEGORIES.index(cat_name)
                    if cat_name in CATEGORIES
                    else len(CATEGORIES) - 1
                )
                bulk_txns = udf[udf["is_bulk_buy"]]
                savings = (
                    float(bulk_txns["amount"].mean() * 0.15)
                    if not bulk_txns.empty
                    else 0.0
                )
            else:
                w_dates = set(window["date"])
                tx_window = udf[udf["date"].isin(w_dates)]
                bulk_label, cat_idx, savings = _infer_bulk_buy_label(tx_window)

            y_bulk.append(bulk_label)
            y_cat.append(cat_idx)
            y_sav.append(savings)

    X  = torch.tensor(X_win, dtype=torch.float32)
    yb = torch.tensor(y_bulk, dtype=torch.float32)
    yc = torch.tensor(y_cat,  dtype=torch.long)
    ys = torch.tensor(y_sav,  dtype=torch.float32)
    return X, yb, yc, ys
