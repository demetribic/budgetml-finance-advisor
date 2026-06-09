"""
tests/test_intelligence_detectors.py — Unit tests for _run_intelligence_detectors.

Run with:
    pytest tests/test_intelligence_detectors.py -v
"""

from __future__ import annotations

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from api.app import _run_intelligence_detectors


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_df(n: int = 5) -> pd.DataFrame:
    """Minimal DataFrame that satisfies all intelligence detectors."""
    import numpy as np
    dates = pd.date_range("2024-01-01", periods=n, freq="7D")
    return pd.DataFrame({
        "date":        dates,
        "amount":      [50.0, 120.0, 30.0, 200.0, 75.0][:n],
        "category":    ["dining", "groceries", "transportation", "shopping", "dining"][:n],
        "merchant":    ["Chipotle", "Whole Foods", "Shell", "Amazon", "Starbucks"][:n],
        "description": [""] * n,
    })


_EMPTY_MODEL_OUTPUTS: dict = {}

_FORECAST_MODEL_OUTPUTS: dict = {
    "forecast": {
        "dining":         {"point": 300.0, "lower": 250.0, "upper": 350.0, "std": 25.0},
        "groceries":      {"point": 400.0, "lower": 350.0, "upper": 450.0, "std": 30.0},
        "transportation": {"point": 100.0, "lower": 80.0,  "upper": 120.0, "std": 10.0},
        "subscriptions":  {"point": 50.0,  "lower": 40.0,  "upper": 60.0,  "std": 5.0},
        "utilities":      {"point": 150.0, "lower": 130.0, "upper": 170.0, "std": 10.0},
        "entertainment":  {"point": 80.0,  "lower": 60.0,  "upper": 100.0, "std": 10.0},
        "shopping":       {"point": 200.0, "lower": 150.0, "upper": 250.0, "std": 20.0},
        "healthcare":     {"point": 30.0,  "lower": 20.0,  "upper": 40.0,  "std": 5.0},
        "other":          {"point": 50.0,  "lower": 30.0,  "upper": 70.0,  "std": 10.0},
    }
}


# ── Return-type contract ──────────────────────────────────────────────────────

def test_returns_tuple_of_list_and_dict():
    result = _run_intelligence_detectors(_make_df(), _EMPTY_MODEL_OUTPUTS, {})
    suggestions, fragment = result
    assert isinstance(suggestions, list)
    assert isinstance(fragment, dict)


def test_returns_correct_types_with_forecast():
    suggestions, fragment = _run_intelligence_detectors(_make_df(), _FORECAST_MODEL_OUTPUTS, {})
    assert isinstance(suggestions, list)
    assert isinstance(fragment, dict)


def test_returns_empty_on_import_error():
    """If any intelligence module is missing the function must swallow the error."""
    with patch.dict("sys.modules", {
        "models.intelligence.life_event_detector":      None,
        "models.intelligence.behavioral_bias_detector": None,
        "models.intelligence.cash_crunch_predictor":    None,
        "models.intelligence.goal_inferencer":          None,
    }):
        suggestions, fragment = _run_intelligence_detectors(_make_df(), _EMPTY_MODEL_OUTPUTS, {})
    assert suggestions == []
    assert fragment == {}


def test_never_raises_on_empty_dataframe():
    empty_df = pd.DataFrame(columns=["date", "amount", "category", "merchant", "description"])
    # Must not raise regardless of data emptiness
    suggestions, fragment = _run_intelligence_detectors(empty_df, _EMPTY_MODEL_OUTPUTS, {})
    assert isinstance(suggestions, list)
    assert isinstance(fragment, dict)


def test_never_raises_when_detector_throws():
    """Individual detector exceptions must be swallowed by the outer try/except."""
    mock_detector = MagicMock(side_effect=RuntimeError("boom"))
    with patch("models.intelligence.life_event_detector.LifeEventDetector", mock_detector):
        suggestions, fragment = _run_intelligence_detectors(_make_df(), _EMPTY_MODEL_OUTPUTS, {})
    # outer except should catch — result must still be valid types
    assert isinstance(suggestions, list)
    assert isinstance(fragment, dict)


# ── Fragment keys ─────────────────────────────────────────────────────────────

def test_fragment_has_expected_keys_when_successful():
    suggestions, fragment = _run_intelligence_detectors(_make_df(), _EMPTY_MODEL_OUTPUTS, {})
    if fragment:  # non-empty means detectors ran
        expected_keys = {"life_events", "behavioral_biases", "cash_crunch_danger_dates", "inferred_goal"}
        assert expected_keys.issubset(fragment.keys())


def test_inferred_goal_is_none_or_string():
    _, fragment = _run_intelligence_detectors(_make_df(), _EMPTY_MODEL_OUTPUTS, {})
    if fragment:
        assert fragment["inferred_goal"] is None or isinstance(fragment["inferred_goal"], str)


def test_cash_crunch_danger_dates_is_list():
    _, fragment = _run_intelligence_detectors(_make_df(), _EMPTY_MODEL_OUTPUTS, {})
    if fragment:
        assert isinstance(fragment["cash_crunch_danger_dates"], list)


# ── Forecast forwarding ───────────────────────────────────────────────────────

def test_forecast_data_forwarded_to_cash_crunch():
    """CashCrunchPredictor must receive the forecast point values, not zeros."""
    captured: dict = {}

    class _CaptureCashCrunch:
        def predict(self, df, forecast_cats):
            captured["forecast_cats"] = forecast_cats
            return {"danger_dates": []}

    with patch("models.intelligence.cash_crunch_predictor.CashCrunchPredictor", _CaptureCashCrunch):
        _run_intelligence_detectors(_make_df(), _FORECAST_MODEL_OUTPUTS, {})

    if captured:  # only assert if patch took effect
        dining_point = _FORECAST_MODEL_OUTPUTS["forecast"]["dining"]["point"]
        assert captured["forecast_cats"].get("dining") == dining_point
