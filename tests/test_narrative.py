"""
tests/test_narrative.py — Integration tests for POST /narrative.

Ollama is mocked so tests run without a running LLM server.

Run with:
    pytest tests/test_narrative.py -v
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app, raise_server_exceptions=False)


# ── Minimal valid request bodies ──────────────────────────────────────────────

_BARE_REQUEST = {
    "suggestions":      [],
    "model_outputs":    {},
    "date_range":       "2024-01-01 – 2024-03-31",
    "num_transactions": 42,
}

_FULL_REQUEST = {
    "suggestions": [
        {"type": "overspending", "category": "dining",  "message": "You spent $600, 20% over budget."},
        {"type": "anomaly_alert","category": "shopping", "message": "Unusual $800 charge at Amazon."},
    ],
    "model_outputs": {
        "forecast": {
            "dining":    {"point": 620.0, "lower": 570.0, "upper": 670.0, "std": 25.0},
            "groceries": {"point": 400.0, "lower": 370.0, "upper": 430.0, "std": 15.0},
        },
        "subscriptions": {"total_monthly": 87.50, "trap_count": 2},
    },
    "date_range":       "2024-01-01 – 2024-03-31",
    "num_transactions": 120,
    "peer_comparison":  {"archetype": "Spender", "message": "You spend 30% more than peers."},
}


# ── Success path ──────────────────────────────────────────────────────────────

def test_narrative_returns_200_and_text():
    with patch("api.app._ollama_generate", return_value="You spent $600 on dining this quarter."):
        resp = client.post("/narrative", json=_BARE_REQUEST)
    assert resp.status_code == 200
    body = resp.json()
    assert "narrative" in body
    assert body["narrative"] == "You spent $600 on dining this quarter."


def test_narrative_full_request_passes_through():
    generated = "Your top spend is dining at $620/mo. Cut two subscriptions to save $30."
    with patch("api.app._ollama_generate", return_value=generated):
        resp = client.post("/narrative", json=_FULL_REQUEST)
    assert resp.status_code == 200
    assert resp.json()["narrative"] == generated


def test_narrative_calls_generate_once():
    """Verify _ollama_generate is called exactly once per request."""
    with patch("api.app._ollama_generate", return_value="ok") as mock_gen:
        client.post("/narrative", json=_BARE_REQUEST)
    mock_gen.assert_called_once()


def test_narrative_prompt_contains_date_range():
    """The prompt forwarded to the LLM must include the date range from the request."""
    captured: list[str] = []

    def _capture(prompt, **kwargs):
        captured.append(prompt)
        return "summary"

    with patch("api.app._ollama_generate", side_effect=_capture):
        client.post("/narrative", json=_FULL_REQUEST)

    assert captured, "generate was not called"
    assert "2024-01-01" in captured[0]


def test_narrative_prompt_includes_subscription_spend():
    captured: list[str] = []

    def _capture(prompt, **kwargs):
        captured.append(prompt)
        return "summary"

    with patch("api.app._ollama_generate", side_effect=_capture):
        client.post("/narrative", json=_FULL_REQUEST)

    assert "87.50" in captured[0]


# ── Failure / fallback path ───────────────────────────────────────────────────

def test_narrative_503_when_no_llm_available():
    with patch("api.app._ollama_generate", return_value=None):
        resp = client.post("/narrative", json=_BARE_REQUEST)
    assert resp.status_code == 503


def test_narrative_503_body_has_detail():
    with patch("api.app._ollama_generate", return_value=None):
        resp = client.post("/narrative", json=_BARE_REQUEST)
    body = resp.json()
    assert "detail" in body


# ── Input validation ──────────────────────────────────────────────────────────

def test_narrative_422_missing_required_fields():
    """Pydantic should reject a body missing 'suggestions' and 'model_outputs'."""
    resp = client.post("/narrative", json={"date_range": "2024-01"})
    assert resp.status_code == 422


def test_narrative_empty_suggestions_still_works():
    with patch("api.app._ollama_generate", return_value="No issues found."):
        resp = client.post("/narrative", json={**_BARE_REQUEST, "suggestions": []})
    assert resp.status_code == 200
