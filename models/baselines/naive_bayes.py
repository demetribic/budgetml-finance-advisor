"""
naive_bayes.py — Naive Bayes baselines for anomaly detection and category classification.

Two models
----------
1. GaussianNBForecaster
   Wraps sklearn GaussianNB to forecast whether next-month spend in each category
   will be "high" (above the user's personal median).  Per-category binary classifiers.

2. GaussianNBAnomalyDetector
   Trains a GaussianNB on per-transaction features; labels transactions as normal/anomaly.
   Used to compare against the transformer autoencoder.
"""

from __future__ import annotations

import numpy as np
import joblib
from pathlib import Path
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report,
)
import pandas as pd


CATEGORIES = [
    "dining", "groceries", "subscriptions", "transportation",
    "utilities", "entertainment", "shopping", "healthcare", "other",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Naive Bayes Spending Forecaster
# ─────────────────────────────────────────────────────────────────────────────

class GaussianNBForecaster:
    """
    Predicts whether next-month spend per category will exceed the
    user's rolling median — a binary "high spend" signal per category.

    Input  X : (N, NUM_CATEGORIES)  — this month's spend per category
    Output y : (N, NUM_CATEGORIES)  — 1 if next month's spend > median, else 0

    Used as a baseline against SpendingForecastTransformer.
    For a fairer comparison we also expose a continuous regression variant
    that just returns the training mean per category (naive mean baseline).
    """

    def __init__(self):
        self.models: list[GaussianNB] = [GaussianNB() for _ in CATEGORIES]
        self.thresholds: list[float] = [0.0] * len(CATEGORIES)
        self.train_means: np.ndarray | None = None
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNBForecaster":
        """
        Parameters
        ----------
        X : (N, NUM_CATEGORIES)   this-month spend per category
        y : (N, NUM_CATEGORIES)   next-month spend per category (continuous)

        Internally binarises y per-category using median thresholds.
        """
        self.train_means = y.mean(axis=0)

        for i, model in enumerate(self.models):
            threshold = float(np.median(y[:, i]))
            self.thresholds[i] = threshold
            y_bin = (y[:, i] > threshold).astype(int)
            model.fit(X, y_bin)

        self._fitted = True
        return self

    def predict_binary(self, X: np.ndarray) -> np.ndarray:
        """Returns (N, NUM_CATEGORIES) binary array: 1 = predicted high spend."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        preds = np.column_stack([m.predict(X) for m in self.models])
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns (N, NUM_CATEGORIES) probability of high spend per category."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        probs = np.column_stack([m.predict_proba(X)[:, 1] for m in self.models])
        return probs

    def predict_spend(self, X: np.ndarray) -> np.ndarray:
        """
        Naive continuous forecast: return per-category training means.
        This is the 'naive mean' baseline — even simpler than NB.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return np.tile(self.train_means, (len(X), 1))

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        """
        Evaluate on held-out data.

        Parameters
        ----------
        y_true : (N, NUM_CATEGORIES) continuous spend values

        Returns
        -------
        dict with per-category F1 and overall MAE (on naive mean prediction).
        """
        y_bin_true = np.column_stack([
            (y_true[:, i] > self.thresholds[i]).astype(int)
            for i in range(len(CATEGORIES))
        ])
        y_bin_pred = self.predict_binary(X)

        metrics: dict = {}
        for i, cat in enumerate(CATEGORIES):
            metrics[cat] = {
                "accuracy":  float(accuracy_score(y_bin_true[:, i], y_bin_pred[:, i])),
                "f1":        float(f1_score(y_bin_true[:, i], y_bin_pred[:, i], zero_division=0)),
                "precision": float(precision_score(y_bin_true[:, i], y_bin_pred[:, i], zero_division=0)),
                "recall":    float(recall_score(y_bin_true[:, i], y_bin_pred[:, i], zero_division=0)),
            }

        # Continuous MAE using naive mean
        y_pred_cont = self.predict_spend(X)
        metrics["overall_mae"] = float(np.mean(np.abs(y_pred_cont - y_true)))
        return metrics

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "GaussianNBForecaster":
        return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Naive Bayes Anomaly Detector
# ─────────────────────────────────────────────────────────────────────────────

class GaussianNBAnomalyDetector:
    """
    Anomaly detection baseline using Gaussian Naive Bayes.

    Features per transaction (matches preprocessor encoding):
        amount_norm, category_id, day_of_week, day_of_month, month, merchant_id

    Labels:
        0 = normal, 1 = anomaly

    Note: NB is a weak anomaly detector since it assumes feature independence,
    but it gives a fast, interpretable baseline.
    """

    def __init__(self):
        self.model   = GaussianNB()
        self.scaler  = StandardScaler()
        self._fitted = False

    def _flatten_windows(self, X: np.ndarray) -> np.ndarray:
        """
        X may be (N, seq_len, feature_dim) windows or (N, feature_dim) flat.
        For NB, we summarise each window as mean + std per feature.
        """
        if X.ndim == 3:
            # Summarise window: mean and std per feature → (N, 2*feature_dim)
            return np.concatenate([X.mean(axis=1), X.std(axis=1)], axis=1)
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNBAnomalyDetector":
        """
        Parameters
        ----------
        X : (N, seq_len, feature_dim) or (N, feature_dim)
        y : (N,) binary labels — 1=anomaly
        """
        X_flat = self._flatten_windows(X)
        X_scaled = self.scaler.fit_transform(X_flat)
        self.model.fit(X_scaled, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns (N,) binary predictions."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        X_flat = self._flatten_windows(X)
        return self.model.predict(self.scaler.transform(X_flat))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns (N, 2) — probability of [normal, anomaly]."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        X_flat = self._flatten_windows(X)
        return self.model.predict_proba(self.scaler.transform(X_flat))

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Returns (N,) probability of being anomalous (0–1)."""
        return self.predict_proba(X)[:, 1]

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        y_pred = self.predict(X)
        return {
            "accuracy":  float(accuracy_score(y_true, y_pred)),
            "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
            "report":    classification_report(y_true, y_pred, zero_division=0),
        }

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "GaussianNBAnomalyDetector":
        return joblib.load(path)
