"""
tests/test_auto_categorize.py — Tests for auto-categorize dispatch and regression coverage.

Verifies that:
- TF-IDF classifier uses a calibrated confidence_threshold
- SetFit classifier uses a higher calibrated threshold than TF-IDF
- Ollama fallback is called when confidence is below threshold
- High-confidence predictions bypass Ollama

Run with:
    pytest tests/test_auto_categorize.py -v
"""

from __future__ import annotations

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

import api.app as app_module
from api.app import _auto_categorize
from models.embeddings.description_classifier import (
    DescriptionClassifier,
    SetFitDescriptionClassifier,
    combine_transaction_text,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _other_df(n: int = 3) -> pd.DataFrame:
    """DataFrame where all rows start as 'other' — candidates for reclassification."""
    return pd.DataFrame({
        "date":        pd.date_range("2024-01-01", periods=n),
        "amount":      [10.0] * n,
        "category":    ["other"] * n,
        "merchant":    ["Starbucks", "Netflix", "Shell"] [:n],
        "description": ["Starbucks Coffee", "Netflix Monthly", "Shell Gas"][:n],
    })


def _make_mock_clf(conf: float, cat: str = "dining") -> MagicMock:
    """Mock classifier that returns `conf` for `cat`, spreading the remainder evenly.

    Using {cat: conf, "other": 1-conf} would make "other" the winner when conf<0.5,
    so Ollama would never be triggered. Spreading evenly ensures `cat` is the argmax.
    """
    from models.embeddings.description_classifier import CATEGORIES
    clf = MagicMock()
    remainder = (1.0 - conf) / max(1, len(CATEGORIES) - 1)
    proba = {c: remainder for c in CATEGORIES}
    proba[cat] = conf
    clf.predict_proba.return_value = [proba]
    return clf


# ── Threshold class attributes ─────────────────────────────────────────────────

def test_tfidf_classifier_threshold():
    clf = DescriptionClassifier()
    assert clf.confidence_threshold == 0.40


def test_setfit_classifier_threshold():
    clf = SetFitDescriptionClassifier()
    assert clf.confidence_threshold == 0.50


def test_setfit_threshold_higher_than_tfidf():
    assert SetFitDescriptionClassifier().confidence_threshold > DescriptionClassifier().confidence_threshold


# ── Threshold dispatch in _auto_categorize ─────────────────────────────────────

def test_high_confidence_skips_ollama():
    """When classifier is confident, Ollama must not be called."""
    clf = _make_mock_clf(conf=0.90, cat="dining")
    clf.confidence_threshold = 0.40

    with patch.object(app_module.ModelRegistry, "description_clf", clf):
        with patch("api.app._ollama_classify") as mock_ollama:
            result = _auto_categorize(_other_df(1))

    mock_ollama.assert_not_called()
    assert result["category"].iloc[0] == "dining"


def test_low_confidence_calls_ollama():
    """When classifier confidence is below threshold, Ollama must be called."""
    clf = _make_mock_clf(conf=0.30, cat="dining")
    clf.confidence_threshold = 0.40

    with patch.object(app_module.ModelRegistry, "description_clf", clf):
        with patch("api.app._ollama_classify", return_value="groceries") as mock_ollama:
            result = _auto_categorize(_other_df(1))

    mock_ollama.assert_called_once()
    assert result["category"].iloc[0] == "groceries"


def test_setfit_threshold_triggers_ollama_at_0_60():
    """A 0.45 confidence score is below SetFit's 0.50 threshold → Ollama called."""
    clf = _make_mock_clf(conf=0.45, cat="shopping")
    clf.confidence_threshold = SetFitDescriptionClassifier.confidence_threshold  # 0.50

    with patch.object(app_module.ModelRegistry, "description_clf", clf):
        with patch("api.app._ollama_classify", return_value="shopping") as mock_ollama:
            _auto_categorize(_other_df(1))

    mock_ollama.assert_called_once()


def test_tfidf_does_not_trigger_ollama_at_0_60():
    """A 0.60 confidence is above TF-IDF's 0.40 threshold → Ollama NOT called."""
    clf = _make_mock_clf(conf=0.60, cat="shopping")
    clf.confidence_threshold = DescriptionClassifier.confidence_threshold  # 0.40

    with patch.object(app_module.ModelRegistry, "description_clf", clf):
        with patch("api.app._ollama_classify") as mock_ollama:
            _auto_categorize(_other_df(1))

    mock_ollama.assert_not_called()


def test_ollama_none_falls_back_to_other():
    """When both classifier and Ollama are unsure, category stays 'other'."""
    clf = _make_mock_clf(conf=0.20, cat="dining")
    clf.confidence_threshold = 0.40

    with patch.object(app_module.ModelRegistry, "description_clf", clf):
        with patch("api.app._ollama_classify", return_value=None):
            result = _auto_categorize(_other_df(1))

    assert result["category"].iloc[0] == "other"


def test_model_predicting_other_still_calls_ollama():
    """A confident 'other' should still trigger fallback because no reclassification happened."""
    clf = _make_mock_clf(conf=0.92, cat="other")
    clf.confidence_threshold = 0.40

    with patch.object(app_module.ModelRegistry, "description_clf", clf):
        with patch("api.app._ollama_classify", return_value="subscriptions") as mock_ollama:
            result = _auto_categorize(_other_df(1))

    mock_ollama.assert_called_once()
    assert result["category"].iloc[0] == "subscriptions"


def test_explicit_min_confidence_overrides_classifier_attribute():
    """Passing min_confidence explicitly ignores the classifier's own threshold."""
    clf = _make_mock_clf(conf=0.55, cat="dining")
    clf.confidence_threshold = 0.40  # would accept 0.55 normally

    with patch.object(app_module.ModelRegistry, "description_clf", clf):
        with patch("api.app._ollama_classify", return_value="groceries") as mock_ollama:
            # Force a high threshold override → Ollama is called despite clf.threshold=0.40
            result = _auto_categorize(_other_df(1), min_confidence=0.80)

    mock_ollama.assert_called_once()


def test_no_classifier_returns_df_unchanged():
    """With no classifier loaded, _auto_categorize is a no-op."""
    df = _other_df(3)
    with patch.object(app_module.ModelRegistry, "description_clf", None):
        result = _auto_categorize(df)
    assert list(result["category"]) == ["other", "other", "other"]


def test_non_other_rows_not_reclassified():
    """Rows that are not 'other' must never be touched."""
    df = pd.DataFrame({
        "date":        pd.date_range("2024-01-01", periods=3),
        "amount":      [10.0, 20.0, 30.0],
        "category":    ["dining", "groceries", "other"],
        "merchant":    ["Chipotle", "Whole Foods", "Unknown"],
        "description": ["Chipotle", "Whole Foods", "Unknown"],
    })
    clf = _make_mock_clf(conf=0.90, cat="shopping")
    clf.confidence_threshold = 0.40

    with patch.object(app_module.ModelRegistry, "description_clf", clf):
        result = _auto_categorize(df)

    assert result["category"].iloc[0] == "dining"
    assert result["category"].iloc[1] == "groceries"
    # only the "other" row is reclassified
    assert result["category"].iloc[2] == "shopping"


def test_auto_categorize_uses_combined_merchant_and_description():
    clf = _make_mock_clf(conf=0.90, cat="subscriptions")
    clf.confidence_threshold = 0.40
    df = pd.DataFrame({
        "date": ["2024-01-01"],
        "amount": [19.99],
        "category": ["other"],
        "merchant": ["Claude Subscription Anthropic Com"],
        "description": ["Monthly Claude Pro subscription"],
    })

    with patch.object(app_module.ModelRegistry, "description_clf", clf):
        with patch("api.app._ollama_classify") as mock_ollama:
            result = _auto_categorize(df)

    mock_ollama.assert_not_called()
    clf.predict_proba.assert_called_once_with(
        [combine_transaction_text(
            merchant="Claude Subscription Anthropic Com",
            description="Monthly Claude Pro subscription",
        )]
    )
    assert result["category"].iloc[0] == "subscriptions"


def test_description_classifier_regression_fixture():
    fixture = Path("evaluation/fixtures/transaction_category_regression.csv")
    df = pd.read_csv(fixture)
    texts = [
        combine_transaction_text(merchant=str(m), description=str(d))
        for m, d in zip(df["merchant"], df["description"])
    ]
    labels = [str(v).strip().lower() for v in df["category"]]

    clf = DescriptionClassifier().fit()
    preds = clf.predict(texts)
    accuracy = sum(int(pred == label) for pred, label in zip(preds, labels)) / len(labels)

    assert accuracy >= 0.90
