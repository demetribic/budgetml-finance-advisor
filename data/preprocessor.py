"""
preprocessor.py — Encode raw transaction DataFrames into tensors for transformer input.

Feature vector (11 scalars per transaction, after 1-A fixes):
    Index 0  : amount_norm          — global z-score of transaction amount
    Index 1  : cat_id               — category integer (0–8), fed to nn.Embedding in base.py
    Index 2  : merch_id             — merchant integer (0–511), fed to nn.Embedding in base.py
    Index 3  : dow_sin              — sin(2π × day_of_week / 7)
    Index 4  : dow_cos              — cos(2π × day_of_week / 7)
    Index 5  : dom_sin              — sin(2π × day_of_month / 31)
    Index 6  : dom_cos              — cos(2π × day_of_month / 31)
    Index 7  : month_sin            — sin(2π × month / 12)
    Index 8  : month_cos            — cos(2π × month / 12)
    Index 9  : user_amount_zscore   — per-user z-score of amount (0 when user unseen)
    Index 10 : description_hash     — hash(description.lower().strip()) % 256 / 256

Rolling windows of `seq_len` days are slid over each user's history to create
(X, y) pairs for the three tasks:
    - Forecasting : y = total spend per category in the next 30 days
    - Anomaly     : y = is_anomaly label for the last transaction in the window
    - Bulk-buy    : y = is_bulk_buy label
"""

from __future__ import annotations

import hashlib
import math
from typing import Optional
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib


def _desc_hash(desc: str) -> float:
    """Deterministic description hash feature. Uses MD5 so value is stable across sessions."""
    if not desc:
        return 0.0
    return int(hashlib.md5(desc.encode(), usedforsecurity=False).hexdigest(), 16) % 256 / 256.0


CATEGORIES = [
    "dining", "groceries", "subscriptions", "transportation",
    "utilities", "entertainment", "shopping", "healthcare", "other",
]
NUM_CATEGORIES = len(CATEGORIES)


class TransactionPreprocessor:
    """
    Fits on a transaction DataFrame and transforms it into windowed PyTorch tensors.

    Parameters
    ----------
    seq_len : int
        Number of days per input window (default 60).
    forecast_horizon : int
        Days ahead to forecast spend (default 30).
    max_merchants : int
        Vocabulary size for merchant embeddings (default 512).
    """

    def __init__(
        self,
        seq_len: int = 60,
        forecast_horizon: int = 30,
        max_merchants: int = 512,
    ):
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        self.max_merchants = max_merchants

        self.cat_encoder    = LabelEncoder()
        self.merch_encoder  = LabelEncoder()
        self.amount_scaler  = StandardScaler()
        self._fitted        = False
        # Fast-lookup caches (populated in fit())
        self._cat_map:   dict = {}
        self._cat_set:   set  = set()
        self._merch_map: dict = {}
        self._merch_set: set  = set()
        self._amount_mean: float = 0.0
        self._amount_std:  float = 1.0
        # Per-user normalization stats: user_id → (mean, std)
        self._user_amount_stats: dict[str | int, tuple[float, float]] = {}

    def __setstate__(self, state: dict) -> None:
        """Backward-compatible unpickling — fill in any attributes added after save."""
        self.__dict__.update(state)
        if "_user_amount_stats" not in state:
            self._user_amount_stats = {}

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "TransactionPreprocessor":
        """Learn encoders and scalers from the full training dataset."""
        self.cat_encoder.fit(CATEGORIES)

        # Limit merchant vocab; replace rare merchants with <UNK>
        top_merchants = (
            df["merchant"].value_counts()
            .head(self.max_merchants - 1)
            .index.tolist()
        )
        self.merch_encoder.fit(top_merchants + ["<UNK>"])

        amounts = df["amount"].values.astype(float)
        self.amount_scaler.fit(amounts.reshape(-1, 1))
        self._amount_mean = float(self.amount_scaler.mean_[0])
        self._amount_std  = float(self.amount_scaler.scale_[0])

        # Build O(1) lookup dicts for encoding
        self._cat_set  = set(self.cat_encoder.classes_)
        self._cat_map  = {c: i for i, c in enumerate(self.cat_encoder.classes_)}
        self._merch_set = set(self.merch_encoder.classes_)
        self._merch_map = {m: i for i, m in enumerate(self.merch_encoder.classes_)}

        # Per-user amount stats for z-score normalization
        self._user_amount_stats = {}
        for uid, udf in df.groupby("user_id"):
            user_amounts = udf["amount"].values.astype(float)
            if len(user_amounts) >= 2:
                self._user_amount_stats[uid] = (
                    float(user_amounts.mean()),
                    float(user_amounts.std()) or 1.0,
                )

        self._fitted = True
        return self

    # ── Single transaction → feature vector ──────────────────────────────────

    def _encode_row(self, row: pd.Series, user_id: int | str | None = None) -> list[float]:
        """Return a fixed-length feature vector for one transaction row."""
        # Global amount normalization
        amount_norm = (float(row["amount"]) - self._amount_mean) / max(self._amount_std, 1e-8)

        cat = row["category"] if row["category"] in self._cat_set else "other"
        cat_id = float(self._cat_map[cat])

        merch = row["merchant"] if row["merchant"] in self._merch_set else "<UNK>"
        merch_id = float(self._merch_map[merch])

        ts = row["date"]
        dow   = float(ts.dayofweek)
        dom   = float(ts.day)
        month = float(ts.month)

        dow_sin   = math.sin(2 * math.pi * dow   / 7)
        dow_cos   = math.cos(2 * math.pi * dow   / 7)
        dom_sin   = math.sin(2 * math.pi * dom   / 31)
        dom_cos   = math.cos(2 * math.pi * dom   / 31)
        month_sin = math.sin(2 * math.pi * month / 12)
        month_cos = math.cos(2 * math.pi * month / 12)

        # Per-user z-score
        uid = user_id if user_id is not None else row.get("user_id", None)
        if uid is not None and uid in self._user_amount_stats:
            u_mean, u_std = self._user_amount_stats[uid]
            user_amount_zscore = (float(row["amount"]) - u_mean) / max(u_std, 1e-8)
        else:
            user_amount_zscore = 0.0

        # Description hash feature (deterministic across sessions)
        desc = str(row.get("description", "") or "").lower().strip()
        desc_hash = _desc_hash(desc)

        return [
            amount_norm, cat_id, merch_id,
            dow_sin, dow_cos, dom_sin, dom_cos, month_sin, month_cos,
            user_amount_zscore, desc_hash,
        ]

    def _encode_df_fast(
        self,
        df: pd.DataFrame,
        user_id: int | str | None = None,
    ) -> np.ndarray:
        """Vectorized encoding of an entire DataFrame → (N, feature_dim) ndarray."""
        amounts = df["amount"].values.astype(float)
        amount_norm = (amounts - self._amount_mean) / max(self._amount_std, 1e-8)

        cats    = df["category"].where(df["category"].isin(self._cat_set), "other")
        cat_ids = cats.map(self._cat_map).fillna(self._cat_map["other"]).values.astype(float)

        merchs    = df["merchant"].where(df["merchant"].isin(self._merch_set), "<UNK>")
        merch_ids = merchs.map(self._merch_map).fillna(self._merch_map["<UNK>"]).values.astype(float)

        dates  = pd.to_datetime(df["date"])
        dows   = dates.dt.dayofweek.values.astype(float)
        doms   = dates.dt.day.values.astype(float)
        months = dates.dt.month.values.astype(float)

        dow_sin   = np.sin(2 * np.pi * dows   / 7)
        dow_cos   = np.cos(2 * np.pi * dows   / 7)
        dom_sin   = np.sin(2 * np.pi * doms   / 31)
        dom_cos   = np.cos(2 * np.pi * doms   / 31)
        month_sin = np.sin(2 * np.pi * months / 12)
        month_cos = np.cos(2 * np.pi * months / 12)

        # Per-user z-score: resolve uid from arg or df column
        uid = user_id
        if uid is None and "user_id" in df.columns:
            uids = df["user_id"].values
            # If all rows share the same user_id, use per-user stats
            if len(np.unique(uids)) == 1:
                uid = uids[0]

        if uid is not None and uid in self._user_amount_stats:
            u_mean, u_std = self._user_amount_stats[uid]
            user_zscores = (amounts - u_mean) / max(u_std, 1e-8)
        else:
            user_zscores = np.zeros(len(amounts), dtype=float)

        # Description hash
        if "description" in df.columns:
            desc_col = df["description"].fillna("").astype(str).str.lower().str.strip()
            desc_hashes = np.array([_desc_hash(d) for d in desc_col], dtype=float)
        else:
            desc_hashes = np.zeros(len(amounts), dtype=float)

        return np.column_stack([
            amount_norm, cat_ids, merch_ids,
            dow_sin, dow_cos, dom_sin, dom_cos, month_sin, month_cos,
            user_zscores, desc_hashes,
        ])

    @property
    def feature_dim(self) -> int:
        """Number of features per transaction step.

        [amount_norm, cat_id, merch_id,
         dow_sin, dow_cos, dom_sin, dom_cos, month_sin, month_cos,
         user_amount_zscore, description_hash]
        """
        return 11

    def save(self, path: str | None = None, *, save_dir: str | None = None) -> None:
        """
        Serialize preprocessor to disk via joblib.

        Parameters
        ----------
        path     : full output path (e.g. 'models/saved/preprocessor.pkl')
        save_dir : directory — saves as <save_dir>/preprocessor.pkl
        """
        import pathlib
        if path is None and save_dir is None:
            raise ValueError("Provide path or save_dir")
        dest = pathlib.Path(path) if path else pathlib.Path(save_dir) / "preprocessor.pkl"
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, str(dest))

    # ── DataFrame → daily aggregated sequences ────────────────────────────────

    def _build_daily_series(self, user_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate per-user transactions to daily rows.
        Returns a DatetimIndex-aligned DataFrame with one row per calendar day,
        filling missing days with zeros.
        """
        user_df = user_df.copy()
        user_df["date"] = pd.to_datetime(user_df["date"]).dt.normalize()

        # Daily aggregations
        agg = user_df.groupby("date").agg(
            total_amount=("amount", "sum"),
            num_txns=("amount", "count"),
            is_anomaly=("is_anomaly", "max"),
            is_bulk_buy=("is_bulk_buy", "max"),
            category=("category", lambda x: x.mode().iloc[0] if len(x) > 0 else "other"),
            merchant=("merchant", lambda x: x.mode().iloc[0] if len(x) > 0 else "<UNK>"),
            description=("description", lambda x: x.iloc[0] if len(x) > 0 else ""),
        ).reset_index()

        # Fill every calendar day in the range
        date_range = pd.date_range(agg["date"].min(), agg["date"].max(), freq="D")
        agg = agg.set_index("date").reindex(date_range).reset_index()
        agg.rename(columns={"index": "date"}, inplace=True)
        agg["total_amount"] = agg["total_amount"].fillna(0.0)
        agg["num_txns"]     = agg["num_txns"].fillna(0).astype(int)
        agg["is_anomaly"]   = agg["is_anomaly"].fillna(0).astype(bool)
        agg["is_bulk_buy"]  = agg["is_bulk_buy"].fillna(0).astype(bool)
        agg["category"]     = agg["category"].fillna("other")
        agg["merchant"]     = agg["merchant"].fillna("<UNK>")
        agg["description"]  = agg["description"].fillna("")
        agg["amount"]       = agg["total_amount"]
        return agg

    # ── Window generation ─────────────────────────────────────────────────────

    def _make_windows_fast(
        self, df: pd.DataFrame, label_col: str, label_dtype: str
    ) -> tuple[Tensor, Tensor]:
        """
        Shared vectorized window builder. Encodes the full daily series at once,
        then uses numpy stride tricks to extract all windows in one shot.

        label_col  : column name in the daily series for labels
        label_dtype: "float32" or "long"
        """
        X_arrays, y_arrays = [], []

        for uid, udf in df.groupby("user_id"):
            daily = self._build_daily_series(udf)
            n = len(daily)
            if n < self.seq_len + 1:
                continue

            # Encode entire series at once (vectorized), passing uid for per-user z-score
            feat_matrix = self._encode_df_fast(daily, user_id=uid)   # (n, feature_dim)

            # Extract all valid windows via stride trick
            n_windows = n - self.seq_len + 1
            windows = np.lib.stride_tricks.sliding_window_view(
                feat_matrix, window_shape=(self.seq_len, feat_matrix.shape[1])
            ).squeeze(axis=1)                            # (n_windows, seq_len, feat_dim)

            labels_raw = daily[label_col].values
            if label_dtype == "float32":
                cumsum = np.concatenate([[0.0], np.cumsum(labels_raw.astype(float))])
                labels = (cumsum[self.seq_len : self.seq_len + n_windows]
                          - cumsum[:n_windows]).astype(np.float32)
            else:
                win_view = np.lib.stride_tricks.sliding_window_view(
                    labels_raw.astype(np.int8), self.seq_len
                )[:n_windows]
                labels = (win_view.max(axis=1) > 0).astype(np.int64)

            X_arrays.append(windows)
            y_arrays.append(labels)

        if not X_arrays:
            empty_x = torch.zeros(0, self.seq_len, self.feature_dim)
            empty_y = torch.zeros(0, dtype=torch.long if label_dtype == "long" else torch.float32)
            return empty_x, empty_y

        X_np = np.concatenate(X_arrays, axis=0).astype(np.float32)
        y_np = np.concatenate(y_arrays, axis=0)

        X = torch.from_numpy(X_np)
        y = torch.from_numpy(y_np)
        if label_dtype == "long":
            y = y.long()
        return X, y

    def make_forecast_windows(
        self, df: pd.DataFrame
    ) -> tuple[Tensor, Tensor]:
        """
        Build (X, y) windows for the spending forecast task.

        X : (N, seq_len, feature_dim)
        y : (N, NUM_CATEGORIES) — total spend per category in the next forecast_horizon days
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before make_forecast_windows()")

        X_arrays, y_arrays = [], []
        horizon = self.forecast_horizon

        for uid, udf in df.groupby("user_id"):
            daily = self._build_daily_series(udf)
            n = len(daily)
            if n < self.seq_len + horizon:
                continue

            feat_matrix = self._encode_df_fast(daily, user_id=uid)   # (n, feature_dim)
            n_windows = n - self.seq_len - horizon + 1

            windows = np.lib.stride_tricks.sliding_window_view(
                feat_matrix, window_shape=(self.seq_len, feat_matrix.shape[1])
            ).squeeze(axis=1)[:n_windows]               # (n_windows, seq_len, feat_dim)

            # Vectorized forecast labels via cumsum
            udf_idx = udf.copy()
            udf_idx["date"] = pd.to_datetime(udf_idx["date"]).dt.normalize()
            daily_cat_spend = (
                udf_idx.groupby(["date", "category"])["amount"]
                .sum()
                .unstack(fill_value=0.0)
                .reindex(columns=CATEGORIES, fill_value=0.0)
            )
            daily_cat_aligned = (
                daily_cat_spend
                .reindex(daily["date"], fill_value=0.0)
                .fillna(0.0)
                .values
            )  # (n, NUM_CATEGORIES)

            cumcat = np.zeros((n + 1, NUM_CATEGORIES), dtype=np.float32)
            cumcat[1:] = np.cumsum(daily_cat_aligned, axis=0)

            i_arr = np.arange(n_windows)
            start_idx = i_arr + self.seq_len
            end_idx   = np.minimum(start_idx + horizon, n)
            y_matrix  = cumcat[end_idx] - cumcat[start_idx]  # (n_windows, NUM_CATEGORIES)

            X_arrays.append(windows.astype(np.float32))
            y_arrays.append(y_matrix)

        if not X_arrays:
            return torch.zeros(0, self.seq_len, self.feature_dim), torch.zeros(0, NUM_CATEGORIES)

        X = torch.from_numpy(np.concatenate(X_arrays, axis=0))
        y = torch.from_numpy(np.concatenate(y_arrays, axis=0))
        return X, y

    def make_anomaly_windows(
        self, df: pd.DataFrame
    ) -> tuple[Tensor, Tensor]:
        """
        X : (N, seq_len, feature_dim)
        y : (N,) long — 1 if any anomaly in window
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before make_anomaly_windows()")
        return self._make_windows_fast(df, "is_anomaly", "long")

    def make_bulkbuy_windows(
        self, df: pd.DataFrame
    ) -> tuple[Tensor, Tensor]:
        """
        X : (N, seq_len, feature_dim)
        y : (N,) long — 1 if any bulk-buy in window
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before make_bulkbuy_windows()")
        return self._make_windows_fast(df, "is_bulk_buy", "long")

    # ── Baseline feature matrix (non-sequential) ──────────────────────────────

    def make_monthly_features(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Aggregate per-user per-month spend into a flat feature matrix for baselines.

        Returns
        -------
        X : (N, NUM_CATEGORIES)  — spend this month per category
        y : (N, NUM_CATEGORIES)  — spend next month per category (forecast target)
        """
        df = df.copy()
        df["year_month"] = df["date"].dt.to_period("M")

        monthly = (
            df.groupby(["user_id", "year_month", "category"])["amount"]
            .sum()
            .unstack(fill_value=0.0)
        )
        # Ensure all categories present
        for cat in CATEGORIES:
            if cat not in monthly.columns:
                monthly[cat] = 0.0
        monthly = monthly[CATEGORIES]

        X_list, y_list = [], []
        for uid in monthly.index.get_level_values("user_id").unique():
            user_months = monthly.xs(uid, level="user_id")
            for i in range(len(user_months) - 1):
                X_list.append(user_months.iloc[i].values.astype(float))
                y_list.append(user_months.iloc[i + 1].values.astype(float))

        return np.array(X_list), np.array(y_list)
