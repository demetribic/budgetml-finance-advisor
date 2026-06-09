"""
gradient_boosting.py — XGBoost and LightGBM baselines.

Three baseline models
---------------------
1. XGBForecaster         — XGBoost multi-output spending forecast (one model per category)
2. LGBMAnomalyDetector   — LightGBM binary anomaly detection classifier
3. XGBBulkBuyClassifier  — XGBoost bulk-buy recommendation classifier

All expose the same fit/predict/evaluate/save/load interface for easy comparison
with the transformer models in evaluation/evaluate.py.
"""

from __future__ import annotations

import numpy as np
import joblib
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    f1_score, precision_score, recall_score, accuracy_score,
    classification_report,
)

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False


CATEGORIES = [
    "dining", "groceries", "subscriptions", "transportation",
    "utilities", "entertainment", "shopping", "healthcare", "other",
]


def _require_xgb():
    if not _XGB_AVAILABLE:
        raise ImportError("xgboost not installed. Run: pip install xgboost")


def _require_lgbm():
    if not _LGBM_AVAILABLE:
        raise ImportError("lightgbm not installed. Run: pip install lightgbm")


# ─────────────────────────────────────────────────────────────────────────────
# 1. XGBoost Spending Forecaster
# ─────────────────────────────────────────────────────────────────────────────

class XGBForecaster:
    """
    Predicts next-month per-category spend using one XGBoost regressor per category.

    Input  X : (N, NUM_CATEGORIES)  — this month's spend per category
    Output y : (N, NUM_CATEGORIES)  — predicted next-month spend
    """

    def __init__(self, **xgb_kwargs):
        _require_xgb()
        defaults = dict(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )
        defaults.update(xgb_kwargs)
        self.models = [xgb.XGBRegressor(**defaults) for _ in CATEGORIES]
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: list[tuple] | None = None,
        verbose: bool = False,
    ) -> "XGBForecaster":
        """
        Parameters
        ----------
        X : (N, NUM_CATEGORIES)
        y : (N, NUM_CATEGORIES)
        """
        for i, model in enumerate(self.models):
            fit_kwargs: dict = {"verbose": verbose}
            if eval_set is not None:
                fit_kwargs["eval_set"] = [(eval_set[0][0], eval_set[0][1][:, i])]
                fit_kwargs["verbose"] = False
            model.fit(X, y[:, i], **fit_kwargs)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns (N, NUM_CATEGORIES) predicted spend, clipped to >= 0."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        preds = np.column_stack([m.predict(X) for m in self.models])
        return np.clip(preds, 0.0, None)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        y_pred = self.predict(X)
        mae  = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        per_cat: dict = {}
        for i, cat in enumerate(CATEGORIES):
            per_cat[cat] = {
                "mae":  float(mean_absolute_error(y_true[:, i], y_pred[:, i])),
                "rmse": float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))),
            }
        return {"overall_mae": mae, "overall_rmse": rmse, "per_category": per_cat}

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "XGBForecaster":
        return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# 2. LightGBM Anomaly Detector
# ─────────────────────────────────────────────────────────────────────────────

class LGBMAnomalyDetector:
    """
    Binary anomaly classifier using LightGBM.

    Input  X : (N, seq_len, feature_dim) windows or (N, feature_dim) flat
    Output y : (N,) binary — 1=anomaly
    """

    def __init__(self, **lgbm_kwargs):
        _require_lgbm()
        defaults = dict(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        defaults.update(lgbm_kwargs)
        self.model   = lgb.LGBMClassifier(**defaults)
        self._fitted = False

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            return np.concatenate([X.mean(axis=1), X.std(axis=1)], axis=1)
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LGBMAnomalyDetector":
        self.model.fit(self._flatten(X), y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return self.model.predict(self._flatten(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return self.model.predict_proba(self._flatten(X))

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
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
    def load(cls, path: str) -> "LGBMAnomalyDetector":
        return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# 3. XGBoost Bulk-Buy Classifier
# ─────────────────────────────────────────────────────────────────────────────

class XGBBulkBuyClassifier:
    """
    Binary classifier: should this user consider buying a category in bulk?

    Input  X : (N, seq_len, feature_dim) windows or (N, feature_dim) flat
    Output y : (N,) binary — 1 = bulk-buy recommended
    """

    def __init__(self, **xgb_kwargs):
        _require_xgb()
        defaults = dict(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            scale_pos_weight=3.0,    # handles class imbalance
            random_state=42,
            n_jobs=-1,
        )
        defaults.update(xgb_kwargs)
        self.model   = xgb.XGBClassifier(**defaults)
        self._fitted = False

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            return np.concatenate([X.mean(axis=1), X.std(axis=1)], axis=1)
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBBulkBuyClassifier":
        self.model.fit(self._flatten(X), y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return self.model.predict(self._flatten(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return self.model.predict_proba(self._flatten(X))

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
    def load(cls, path: str) -> "XGBBulkBuyClassifier":
        return joblib.load(path)
