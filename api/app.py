"""
api/app.py — FastAPI service exposing all three models + decision engine.

Endpoints
---------
POST /analyze          Run all models → return unified suggestions
POST /forecast         Spending forecast only
POST /anomaly          Anomaly detection only
POST /recommend        Bulk-buy recommendation only
GET  /health           Liveness check

Run with:
    uvicorn api.app:app --reload --port 8000

Request body (all POST endpoints):
    {
      "transactions": [
        {
          "date": "2024-03-15",
          "amount": 45.20,
          "category": "dining",
          "merchant": "Chipotle",
          "description": "Chipotle - dining"
        },
        ...
      ],
      "budget": {                      // optional, per-category monthly budgets
        "dining": 200.0,
        "groceries": 400.0
      }
    }
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path
from typing import AsyncGenerator
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
import joblib

from fastapi import FastAPI, HTTPException, Query, Security, Depends
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator

from config import Settings
from data.preprocessor import TransactionPreprocessor, NUM_CATEGORIES, CATEGORIES
from models.embeddings.description_classifier import combine_transaction_text
from models.transformers.spending_forecast    import SpendingForecastTransformer
from models.transformers.anomaly_detection    import AnomalyDetectionTransformer
from models.transformers.bulk_buy_recommendation import BulkBuyRecommendationTransformer
from rules.decision_engine import DecisionEngine, Suggestion


SAVE_DIR = Path(__file__).parent.parent / "models" / "saved"
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ANALYSIS_MAX_TRANSACTIONS = int(os.environ.get("BUDGETML_ANALYSIS_TX_CAP", "1000"))

# ── API key authentication ────────────────────────────────────────────────────
# Set BUDGETML_API_KEY env var to enable auth. If unset, auth is disabled.
_API_KEY = os.environ.get("BUDGETML_API_KEY")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def _verify_api_key(api_key: str | None = Security(_api_key_header)):
    if _API_KEY is None:
        return  # auth disabled
    if not hmac.compare_digest(api_key or "", _API_KEY or ""):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# ── Simple in-memory rate limiter ─────────────────────────────────────────────
_RATE_LIMIT = int(os.environ.get("BUDGETML_RATE_LIMIT", "60"))  # requests per minute
_rate_buckets: dict[str, list[float]] = defaultdict(list)
_rate_lock = threading.Lock()


async def _check_rate_limit(api_key: str | None = Security(_api_key_header)):
    client_id = api_key or "anonymous"
    now = time.monotonic()
    with _rate_lock:
        window = [t for t in _rate_buckets[client_id] if now - t < 60]
        if len(window) >= _RATE_LIMIT:
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
        window.append(now)
        _rate_buckets[client_id] = window


# ── Pydantic schemas ──────────────────────────────────────────────────────────

_VALID_CATEGORIES = set(CATEGORIES)


class Transaction(BaseModel):
    date:        str
    amount:      float
    category:    str   = "other"
    merchant:    str   = "Unknown"
    description: str   = ""

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        try:
            pd.to_datetime(v)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid date format: {v!r}. Use YYYY-MM-DD.")
        return v

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Amount must be non-negative.")
        return v

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        if v not in _VALID_CATEGORIES:
            return "other"
        return v


class AnalyzeRequest(BaseModel):
    transactions: list[Transaction] = Field(..., min_length=1, max_length=10000)
    budget:   dict[str, float] | None = None
    user_id:  str | None = None   # if provided, history is persisted in SQLite


class SuggestionOut(BaseModel):
    type:          str
    category:      str
    message:       str
    confidence:    float
    amount_impact: float
    details:       dict
    explanation:   dict | None = None   # filled by SuggestionExplainer if available


class AnalyzeResponse(BaseModel):
    suggestions:      list[SuggestionOut]
    num_transactions: int
    date_range:       str
    model_outputs:    dict
    analysis_context: dict | None = None
    peer_comparison:  dict | None = None
    deals:            list | None = None


# ── Ensemble forecast wrapper ─────────────────────────────────────────────────

class _EnsembleForecast:
    """
    Wraps N independently-trained SpendingForecastTransformer instances.

    mc_predict() runs MC Dropout on each member and pools the sample distributions
    before computing mean/std/quantiles — combines both epistemic uncertainty
    (within-member MC Dropout) and model-level diversity (across members).
    """

    def __init__(self, members: list):
        self.members = members

    def mc_predict(self, x: torch.Tensor, n_samples: int = 30) -> dict[str, torch.Tensor]:
        """Pool MC samples across all ensemble members."""
        all_samples = []
        for m in self.members:
            out = m.mc_predict(x, n_samples=n_samples)
            # out["mean"] shape: (batch, num_categories)
            # reconstruct raw samples via mean ± std isn't lossless, so just
            # use each member's mean as one "super-sample" from the ensemble
            all_samples.append(out["mean"])

        # (num_members, batch, num_categories)
        stacked = torch.stack(all_samples, dim=0)
        mean  = stacked.mean(dim=0)
        std   = stacked.std(dim=0, correction=0)
        lower = stacked.quantile(0.10, dim=0).clamp(min=0.0)
        upper = stacked.quantile(0.90, dim=0)
        return {"mean": mean, "std": std, "lower": lower, "upper": upper}


# ── Model registry (loaded once at startup) ──────────────────────────────────

class ModelRegistry:
    prep:              TransactionPreprocessor | None                         = None
    forecast:          SpendingForecastTransformer | _EnsembleForecast | None = None
    anomaly:           AnomalyDetectionTransformer | None                     = None
    contrastive_anomaly = None   # ContrastiveAnomalyDetector | None
    bulkbuy:           BulkBuyRecommendationTransformer | None                = None
    user_vae           = None   # UserSpendingVAE | None
    cohort_embs        = None   # (N, 64) Tensor | None
    archetypes         = None   # dict with centers/names/labels | None
    cohort_stats       = None   # dict {cluster_idx: {category: {mean,median,p25,p75}}} | None
    description_clf    = None   # DescriptionClassifier | None
    setfit_description_clf = None
    tfidf_description_clf = None
    active_description_backend: str | None = None
    threshold:         float = 0.5
    engine:            DecisionEngine = DecisionEngine()

    @classmethod
    def load(cls):
        cfg = Settings.load()

        # Preprocessor
        prep_path = SAVE_DIR / "preprocessor.pkl"
        if prep_path.exists():
            cls.prep = joblib.load(prep_path)
        else:
            cls.prep = None

        # Forecast transformer — detect ensemble (forecast_transformer_0.pt, _1.pt …)
        # or fall back to single model (forecast_transformer.pt)
        if cls.prep:
            mcfg = cfg.model("forecast")
            ensemble_paths = sorted(SAVE_DIR.glob("forecast_transformer_[0-9]*.pt"))
            if ensemble_paths:
                members = []
                for ep in ensemble_paths:
                    m = SpendingForecastTransformer(
                        feature_dim=cls.prep.feature_dim,
                        d_model=mcfg.d_model,
                        nhead=mcfg.nhead,
                        num_layers=mcfg.num_layers,
                    ).to(DEVICE)
                    m.load_state_dict(torch.load(ep, map_location=DEVICE, weights_only=True))
                    m.eval()
                    members.append(m)
                cls.forecast = _EnsembleForecast(members)
                print(f"[API] Loaded ensemble of {len(members)} forecast models")
            else:
                fc_path = SAVE_DIR / "forecast_transformer.pt"
                if fc_path.exists():
                    m = SpendingForecastTransformer(
                        feature_dim=cls.prep.feature_dim,
                        d_model=mcfg.d_model,
                        nhead=mcfg.nhead,
                        num_layers=mcfg.num_layers,
                    ).to(DEVICE)
                    m.load_state_dict(torch.load(fc_path, map_location=DEVICE, weights_only=True))
                    m.eval()
                    cls.forecast = m

        # Anomaly transformer
        an_path  = SAVE_DIR / "anomaly_transformer.pt"
        thr_path = SAVE_DIR / "anomaly_threshold.pt"
        if an_path.exists() and cls.prep:
            mcfg = cfg.model("anomaly")
            m = AnomalyDetectionTransformer(
                feature_dim=cls.prep.feature_dim,
                seq_len=cls.prep.seq_len,
                d_model=mcfg.d_model,
                nhead=mcfg.nhead,
                num_layers=mcfg.num_layers,
            ).to(DEVICE)
            m.load_state_dict(torch.load(an_path, map_location=DEVICE, weights_only=True))
            m.eval()
            cls.anomaly = m
            if thr_path.exists():
                cls.threshold = float(
                    torch.load(thr_path, map_location="cpu", weights_only=True)["threshold"]
                )

        # Bulk-buy transformer
        bb_path = SAVE_DIR / "bulkbuy_transformer.pt"
        if bb_path.exists() and cls.prep:
            mcfg = cfg.model("bulk_buy")
            m = BulkBuyRecommendationTransformer(
                feature_dim=cls.prep.feature_dim,
                num_categories=NUM_CATEGORIES,
                d_model=mcfg.d_model,
                nhead=mcfg.nhead,
                num_layers=mcfg.num_layers,
            ).to(DEVICE)
            m.load_state_dict(torch.load(bb_path, map_location=DEVICE, weights_only=True))
            m.eval()
            cls.bulkbuy = m

        # Contrastive anomaly model (optional second anomaly detector)
        ca_path = SAVE_DIR / "contrastive_anomaly.pt"
        if ca_path.exists() and cls.prep:
            from models.transformers.base import TransactionTransformer as _TT
            from models.transformers.contrastive_anomaly import ContrastiveAnomalyDetector
            mcfg = cfg.model("anomaly")
            enc  = _TT(
                feature_dim=cls.prep.feature_dim,
                d_model=mcfg.d_model,
                nhead=mcfg.nhead,
                num_layers=mcfg.num_layers,
            ).to(DEVICE)
            cls.contrastive_anomaly = ContrastiveAnomalyDetector.load(ca_path, enc)
            cls.contrastive_anomaly.to(DEVICE)
            cls.contrastive_anomaly.eval()

        # User VAE
        vae_path = SAVE_DIR / "user_vae.pt"
        if vae_path.exists() and cls.prep:
            from models.transformers.base import TransactionTransformer
            from models.user_vae import UserSpendingVAE
            # weights_only=False: checkpoint contains non-tensor metadata (d_model, nhead, etc.)
            ckpt = torch.load(vae_path, map_location=DEVICE, weights_only=False)
            enc  = TransactionTransformer(
                feature_dim=ckpt["feature_dim"],
                d_model=ckpt["d_model"],
                nhead=ckpt["nhead"],
                num_layers=ckpt["num_layers"],
            ).to(DEVICE)
            vae  = UserSpendingVAE(
                encoder=enc,
                num_categories=ckpt["num_categories"],
                seq_len=ckpt["seq_len"],
                beta=ckpt["beta"],
            ).to(DEVICE)
            vae.load_state_dict(ckpt["state_dict"])
            vae.eval()
            cls.user_vae = vae

        cohort_path = SAVE_DIR / "user_cohort_embeddings.pt"
        if cohort_path.exists():
            cls.cohort_embs = torch.load(cohort_path, map_location=DEVICE, weights_only=True)

        archetype_path = SAVE_DIR / "user_archetypes.pt"
        if archetype_path.exists():
            # weights_only=False: archetypes dict contains string names list alongside tensors
            cls.archetypes = torch.load(archetype_path, map_location="cpu", weights_only=False)

        cohort_stats_path = SAVE_DIR / "cohort_spend_stats.pt"
        if cohort_stats_path.exists():
            # weights_only=False: cohort_stats is a nested dict {cluster: {category: {mean, ...}}}
            cls.cohort_stats = torch.load(cohort_stats_path, map_location="cpu", weights_only=False)

        cls.description_clf = None
        cls.setfit_description_clf = None
        cls.tfidf_description_clf = None
        cls.active_description_backend = None

        # SetFit directory takes priority; fall back to legacy TF-IDF .pkl
        setfit_path = SAVE_DIR / "setfit_description_classifier"
        clf_path    = SAVE_DIR / "description_classifier.pkl"
        if setfit_path.exists() and (setfit_path / "setfit_manifest.json").exists():
            try:
                from models.embeddings.description_classifier import SetFitDescriptionClassifier
                cls.setfit_description_clf = SetFitDescriptionClassifier.load(setfit_path)
                print(f"[API] SetFit description classifier loaded (params: {cls.setfit_description_clf.count_parameters():,})")
            except Exception as e:
                print(f"[API] SetFit load failed ({e})")

        if clf_path.exists():
            from models.embeddings.description_classifier import DescriptionClassifier
            cls.tfidf_description_clf = DescriptionClassifier.load(clf_path)
            print(f"[API] TF-IDF description classifier loaded")

        if cls.setfit_description_clf is not None:
            cls.description_clf = cls.setfit_description_clf
            cls.active_description_backend = "setfit"
        elif cls.tfidf_description_clf is not None:
            cls.description_clf = cls.tfidf_description_clf
            cls.active_description_backend = "tfidf"

        print(f"[API] Models loaded: "
              f"forecast={cls.forecast is not None} "
              f"anomaly={cls.anomaly is not None} "
              f"bulkbuy={cls.bulkbuy is not None} "
              f"user_vae={cls.user_vae is not None} "
              f"description_backend={cls.active_description_backend}")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="BudgetML API",
    description="ML-powered budgeting suggestions using transformer models.",
    version="1.0.0",
    dependencies=[Depends(_verify_api_key), Depends(_check_rate_limit)],
    docs_url="/docs",
    redoc_url="/redoc",
)


_STATIC = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    return HTMLResponse(
        content=(_STATIC / "index.html").read_text(),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/static/chart.js", include_in_schema=False)
def serve_chartjs():
    from fastapi.responses import Response
    return Response(
        content=(_STATIC / "chart.umd.min.js").read_bytes(),
        media_type="application/javascript",
    )


@app.on_event("startup")
async def startup_event():
    from api.db import init_db
    init_db()
    print("[API] Startup: initializing model registry")
    # Run synchronous, potentially slow model loading in a thread so the event
    # loop stays responsive.
    await asyncio.to_thread(ModelRegistry.load)
    print("[API] Startup: models ready")

    # Startup must never block on external network probes. Kick these off in
    # background so the site is immediately reachable even when offline.
    async def _prewarm_backends() -> None:
        try:
            await asyncio.to_thread(_check_ollama)
            await asyncio.to_thread(_check_web_search)
        except Exception as e:
            print(f"[API] Startup prewarm skipped ({e})")

    asyncio.create_task(_prewarm_backends())


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {
            "forecast": ModelRegistry.forecast is not None,
            "anomaly":  ModelRegistry.anomaly  is not None,
            "bulkbuy":  ModelRegistry.bulkbuy  is not None,
        },
        "device": str(DEVICE),
    }


def _transactions_to_df(transactions: list[Transaction]) -> pd.DataFrame:
    records = [
        {
            "user_id":    0,
            "date":       pd.to_datetime(t.date),
            "amount":     t.amount,
            "category":   t.category,
            "merchant":   t.merchant,
            "description": t.description,
            "is_anomaly":  False,
            "is_bulk_buy": False,
        }
        for t in transactions
    ]
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _df_to_tensor(df: pd.DataFrame, prep: TransactionPreprocessor) -> torch.Tensor:
    """
    Encode the most recent `seq_len` days as a single-sample batch.

    When fewer than seq_len days of data are available, the window is filled
    by tiling the real data rather than zero-padding. Zero-padding creates an
    artificial "33 silent days then burst" pattern that causes the transformer
    to over-predict because it was trained on uniformly active 60-day windows.
    Tiling preserves the user's actual spending distribution throughout the
    full input window.
    """
    seq_len = prep.seq_len
    daily   = prep._build_daily_series(df)
    n       = len(daily)

    if n < seq_len:
        needed = seq_len - n
        # Tile the real data to fill the gap, adjusting dates backward
        repeat_times = (needed // max(n, 1)) + 2
        tile_df = pd.concat([daily] * repeat_times, ignore_index=True)
        fill_df = tile_df.tail(needed).copy().reset_index(drop=True)
        start_date = daily["date"].min() - pd.Timedelta(days=needed)
        fill_df["date"] = [start_date + pd.Timedelta(days=i) for i in range(needed)]
        daily = pd.concat([fill_df, daily], ignore_index=True)

    window = daily.tail(seq_len)
    feats  = [prep._encode_row(row) for _, row in window.iterrows()]
    return torch.tensor([feats], dtype=torch.float32).to(DEVICE)


_AUTO_CATEGORIZE_MIN_CONF: float = 0.40   # default; overridden by classifier.confidence_threshold


# Always blocked — no description text can identify the underlying spend
_P2P_HARD_BLOCK = frozenset({
    "apple cash", "zelle", "cash app", "cashapp",
    "samsung pay", "square cash",
})
# Soft-blocked — can pass through when the description identifies a real service (e.g. PayPal/Uber Eats)
_P2P_SOFT_BLOCK = frozenset({
    "paypal", "venmo", "google pay", "chime", "current", "wise", "revolut",
})
_P2P_ALWAYS_OTHER = _P2P_HARD_BLOCK | _P2P_SOFT_BLOCK

# Deterministic keyword rules checked before any ML model.
# Patterns are matched against uppercased classifier text.
_KEYWORD_RULES: list[tuple[re.Pattern, str]] = [
    # More-specific patterns first so they shadow less-specific ones
    (re.compile(r"\bUBER\s*EATS\b"),                                                                 "dining"),
    (re.compile(r"\bUBER\b"),                                                                        "transportation"),
    (re.compile(r"\b(ANTHROPIC|CLAUDE)\b"),                                                          "subscriptions"),
    (re.compile(r"\b(INTL\s+FEE|INTERNATIONAL\s+FEE|FOREIGN\s+TRANSACTION|FX\s+FEE|CROSS\s+BORDER\s+FEE|FOREIGN\s+EXCHANGE\s+FEE)\b"), "other"),
    (re.compile(r"\b(SERVICE\s+CHARGE|MONTHLY\s+SERVICE\s+FEE|ACCOUNT\s+FEE|MAINTENANCE\s+FEE|OVERDRAFT\s+FEE|LATE\s+FEE|ANNUAL\s+FEE|CARD\s+FEE)\b"), "other"),
    (re.compile(r"\b(RIU\s+MARKET|RESORT\s+MARKET|HOTEL\s+MARKET|DUTY\s+FREE)\b"),                  "shopping"),
    (re.compile(r"\bLIQUORS?\b"),                                                                    "shopping"),
]

# Tokens strongly associated with subscriptions; if SetFit predicts "subscriptions" but none
# appear in the text, confidence is penalized to prevent false positives like "Eleven Ewing".
_SUBSCRIPTION_TOKENS: frozenset[str] = frozenset({
    "SUBSCRI", "PREMIUM", "PLUS", "PLAN", "MEMBER", "ANNUAL", "MONTHLY",
    "NETFLIX", "SPOTIFY", "HULU", "DISNEY", "HBO", "APPLE", "AMAZON",
    "YOUTUBE", "PARAMOUNT", "PEACOCK", "ADOBE", "MICROSOFT", "DROPBOX",
    "GOOGLE", "ICLOUD", "AUDIBLE", "LINKEDIN", "NYTIMES", "FITNESS",
    "YMCA", "GYM", "EQUINOX", "CRUNCH", "DUOLINGO", "HEADSPACE", "CALM",
    "NOOM", "MASTERCLASS", "COURSERA", "NOTION", "SLACK", "ZOOM",
    "GITHUB", "COPILOT", "CLAUDE", "ANTHROPIC", "OPENAI", "CHATGPT",
})


def _is_p2p_merchant(merchant: str) -> bool:
    m = str(merchant).lower()
    return any(kw in m for kw in _P2P_ALWAYS_OTHER)

def _is_p2p_hard_block(merchant: str) -> bool:
    m = str(merchant).lower()
    return any(kw in m for kw in _P2P_HARD_BLOCK)


def _keyword_classify(text: str) -> str | None:
    """Return a hard-coded category for text matching a known keyword rule, or None."""
    upper = text.upper()
    for pattern, cat in _KEYWORD_RULES:
        if pattern.search(upper):
            return cat
    return None


def _penalize_subscription_fp(text: str, cat: str, conf: float) -> float:
    """Reduce confidence when SetFit predicts 'subscriptions' without subscription keywords present."""
    if cat != "subscriptions":
        return conf
    upper = text.upper()
    tokens = set(re.findall(r"[A-Z]{3,}", upper))
    if not tokens & _SUBSCRIPTION_TOKENS:
        return conf * 0.35   # heavy penalty → falls below 0.30 threshold → triggers fallback
    return conf


def _apply_category_preferences(df: pd.DataFrame, user_id: str) -> pd.DataFrame:
    """Override transaction categories with the user's confirmed merchant preferences."""
    try:
        from api.db import get_category_preferences
        prefs = get_category_preferences(user_id)
        if not prefs:
            return df
        df = df.copy()
        norm = df["merchant"].str.strip().str.lower()
        df["category"] = [
            "other" if _is_p2p_merchant(m) else prefs.get(m, cat)
            for m, cat in zip(norm, df["category"])
        ]
    except Exception:
        pass
    return df


def _auto_categorize(df: pd.DataFrame, min_confidence: float | None = None) -> pd.DataFrame:
    """
    Reclassify 'other' rows using the description classifier.

    Pipeline: classifier.predict_proba on combined merchant+description text
    → if conf < threshold or the classifier still says 'other'
    → Ollama → local_llm → 'other'.
    The threshold is taken from the classifier's own `confidence_threshold` attribute so
    each backend can self-declare its calibration level.
    Pass min_confidence explicitly to override for testing or one-off calls.
    """
    reg = ModelRegistry
    if reg.description_clf is None:
        return df
    if min_confidence is None:
        min_confidence = getattr(reg.description_clf, "confidence_threshold", _AUTO_CATEGORIZE_MIN_CONF)
    mask = df["category"] == "other"
    if not mask.any():
        return df
    # Never reclassify P2P rails — the actual spend category is unknowable from text.
    if "merchant" in df.columns:
        mask = mask & ~df["merchant"].fillna("").apply(_is_p2p_merchant)
    if not mask.any():
        return df
    # When description is blank (common for CSV uploads), fall back to merchant name.
    has_desc = "description" in df.columns
    has_merch = "merchant" in df.columns
    if not has_desc and not has_merch:
        return df
    raw_descs = df.loc[mask, "description"].fillna("") if has_desc else pd.Series([""] * mask.sum())
    raw_merchs = df.loc[mask, "merchant"].fillna("") if has_merch else pd.Series([""] * mask.sum())
    classifier_texts = [
        combine_transaction_text(merchant=str(m), description=str(d))
        for d, m in zip(raw_descs, raw_merchs)
    ]
    proba_list = reg.description_clf.predict_proba(classifier_texts)
    auto_cats = []
    for desc, p in zip(classifier_texts, proba_list):
        best_cat = max(p, key=p.get)
        should_fallback = (best_cat == "other") or (p[best_cat] < min_confidence)
        if not should_fallback:
            auto_cats.append(best_cat)
        else:
            # Try Ollama when the model is unsure or effectively declines to reclassify.
            ollama_cat, _ = _ollama_classify(desc)
            auto_cats.append(ollama_cat if ollama_cat else "other")
    df = df.copy()
    df.loc[mask, "category"] = auto_cats
    return df


_WEB_CATEGORY_HINTS: dict[str, list[str]] = {
    "dining": ["restaurant", "coffee", "cafe", "food", "eatery", "grill", "bakery", "diner", "bistro"],
    "groceries": ["grocery", "supermarket", "market", "produce", "foods", "food store", "deli"],
    "subscriptions": ["subscription", "monthly plan", "annual plan", "membership", "saas", "netflix", "spotify", "hulu"],
    "transportation": ["fuel", "gas station", "transit", "rideshare", "automotive", "toll", "parking", "airline", "car rental"],
    "utilities": ["utility", "electric", "internet provider", "wireless carrier", "phone bill", "water utility", "energy provider"],
    "entertainment": ["entertainment", "movie theater", "video games", "concert", "tickets", "arcade", "amusement"],
    "shopping": ["retail", "store", "clothing", "apparel", "electronics", "ecommerce", "department store"],
    "healthcare": ["pharmacy", "medical", "clinic", "hospital", "health care", "prescription", "dentist", "optometry"],
}


def _extract_merchant_query(text: str) -> str:
    """Extract a concise merchant-like query phrase from noisy statement text."""
    tokens = re.sub(r"[^A-Za-z0-9 ]", " ", str(text)).upper().split()
    stop = {
        "VISA", "MASTERCARD", "DEBIT", "CREDIT", "CARD", "PUR", "PURCHASE", "POS", "DDA",
        "AP", "AUTH", "PENDING", "POSTED", "NUM", "NO", "REF", "ID", "CHECK", "CHK",
        "PAYMENT", "TRANSFER", "ACH",
    }
    keep = [t for t in tokens if t not in stop and not t.isdigit() and len(t) > 2]
    if not keep:
        return str(text).strip()[:64]
    return " ".join(keep[:6])


def _infer_category_from_search(text: str) -> tuple[str | None, float, list[dict]]:
    """
    Use DuckDuckGo/Brave web search snippets to infer a category when model confidence is low.
    Returns (category, confidence, search_results).
    """
    if not _check_web_search():
        return None, 0.0, []
    try:
        from rules.deal_finder import _search  # reuse already configured search backend + cache
    except Exception:
        return None, 0.0, []

    q = _extract_merchant_query(text)
    if not q:
        return None, 0.0, []

    results = _search(f"{q} merchant business type")
    if not results:
        return None, 0.0, []

    score: dict[str, float] = {c: 0.0 for c in _WEB_CATEGORY_HINTS}
    for r in results:
        content = f"{r.get('title', '')} {r.get('snippet', '')}".lower()
        for cat, hints in _WEB_CATEGORY_HINTS.items():
            hits = sum(1 for h in hints if h in content)
            score[cat] += float(hits)

    best_cat = max(score, key=score.get)
    best = score[best_cat]
    total = sum(score.values())
    if best <= 0.0:
        return None, 0.0, results
    conf = best / total if total > 0 else 0.0
    if conf < 0.34:
        return None, conf, results
    return best_cat, conf, results


def _compute_trend_multiplier(df: pd.DataFrame) -> np.ndarray:
    """
    Per-category spending trend: compare recent half of history vs earlier half.
    Returns multiplier array capped at [0.7, 1.4] to avoid noise overreaction.
    """
    mult = np.ones(len(CATEGORIES), dtype=np.float32)
    span = df["date"].max() - df["date"].min()
    if span.days < 1:
        return mult
    midpoint = df["date"].min() + span / 2
    recent  = df[df["date"] >= midpoint]
    earlier = df[df["date"] <  midpoint]
    if earlier.empty or recent.empty:
        return mult
    r_days = max((recent["date"].max()  - recent["date"].min()).days  + 1, 1)
    e_days = max((earlier["date"].max() - earlier["date"].min()).days + 1, 1)
    for i, cat in enumerate(CATEGORIES):
        r = float(recent[recent["category"]   == cat]["amount"].sum()) / r_days
        e = float(earlier[earlier["category"] == cat]["amount"].sum()) / e_days
        if e > 0.5:
            mult[i] = float(np.clip(r / e, 0.7, 1.4))
    return mult


def _calibrated_forecast(
    fp: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Anchor transformer predictions to the user's real spending rate.

    Point estimate = actual_rate * trend_multiplier (no transformer scale bias).
    Bounds         = transformer's relative uncertainty rescaled to the point estimate.

    Categories with zero real spend fall back to raw transformer output.
    """
    date_span_days = max(1, (df["date"].max() - df["date"].min()).days + 1)
    monthly_scale  = 30.0 / date_span_days
    actual_rate = np.array([
        float(df[df["category"] == cat]["amount"].sum()) * monthly_scale
        for cat in CATEGORIES
    ], dtype=np.float32)

    has_signal = actual_rate > 1.0  # at least $1/month real spend

    if date_span_days >= 14:
        trend_mult = _compute_trend_multiplier(df)
        point = np.where(has_signal, actual_rate * trend_mult, 0.0)
    else:
        # Too little data for reliable trend — use 90% actual rate + 10% transformer
        point = np.where(has_signal, 0.1 * fp + 0.9 * actual_rate, 0.0)

    # Preserve transformer's relative uncertainty width, rescaled to new point
    safe_fp   = np.maximum(fp, 1e-6)
    rel_lower = np.clip((fp - lower) / safe_fp, 0.0, 0.6)
    rel_upper = np.clip((upper - fp) / safe_fp, 0.0, 0.6)
    cal_lower = np.maximum(point * (1.0 - rel_lower), 0.0)
    cal_upper = point * (1.0 + rel_upper)

    return point, cal_lower, cal_upper


def _build_subscription_inventory(df: pd.DataFrame) -> dict:
    """Build additive subscription inventory for model_outputs."""
    try:
        from rules.subscription_analyzer import SubscriptionAnalyzer
        items = SubscriptionAnalyzer().analyze(df)
    except Exception:
        items = []

    # Fallback: surface category-tagged subscription transactions not caught by
    # the recurring detector (requires 3+ months). These show as "unconfirmed".
    if not df.empty and "category" in df.columns:
        detected_merchants = {i["merchant"] for i in items}
        sub_df = df[df["category"].str.lower().str.contains("subscri", na=False)]
        for merchant, mdf in sub_df.groupby("merchant"):
            if _is_p2p_merchant(str(merchant)):
                continue
            if merchant in detected_merchants:
                continue
            monthly_cost = round(float(mdf["amount"].mean()), 2)
            last_date = pd.to_datetime(mdf["date"].max())
            items.append({
                "merchant":            str(merchant),
                "category":            "subscriptions",
                "monthly_cost":        monthly_cost,
                "annual_cost":         round(monthly_cost * 12, 2),
                "tenure_months":       int(len(mdf.groupby(pd.to_datetime(mdf["date"]).dt.to_period("M")))),
                "usage_score":         0.0,
                "is_trap":             False,
                "status":              "unconfirmed",
                "next_expected_charge_date": str((last_date + pd.Timedelta(days=30)).date()),
                "last_charge_date":    str(last_date.date()),
                "charge_count_detected": int(len(mdf)),
                "action_recommendation": "Review — insufficient history to confirm recurrence",
                "negotiation_script":  "",
                "cancel_instructions": "",
            })

    total_monthly = float(sum(float(s.get("monthly_cost", 0.0)) for s in items))
    total_annual = total_monthly * 12.0
    trap_count = int(sum(1 for s in items if s.get("status") == "trap" or s.get("is_trap")))
    return {
        "items": items,
        "total_monthly": round(total_monthly, 2),
        "total_annual": round(total_annual, 2),
        "trap_count": trap_count,
    }


def _run_anomaly_head(x: torch.Tensor, reg: "ModelRegistry") -> dict | None:
    """Run anomaly + contrastive models and return a unified anomaly dict."""
    if not reg.anomaly:
        return None
    res = reg.anomaly.predict(x, threshold=reg.threshold)
    base_score = float(res["anomaly_score"][0])
    contrastive_score = None
    if reg.contrastive_anomaly is not None:
        try:
            ca_res = reg.contrastive_anomaly.predict(x)
            contrastive_score = float(ca_res["anomaly_score"][0])
        except Exception:
            pass
    combined_score = max(base_score, contrastive_score) if contrastive_score is not None else base_score
    return {
        "score":             combined_score,
        "base_recon_score":  base_score,
        "contrastive_score": contrastive_score,
        "is_flag":           combined_score > reg.threshold,
        "class":             int(res["anomaly_class"][0]),
        "_res":              res,  # raw result for anomaly_result dict downstream
    }


def _build_analysis_context(
    df: pd.DataFrame,
    budget: dict[str, float] | None,
    model_outputs: dict,
    run_id: str | None = None,
    partial_details: bool = False,
) -> dict:
    """Build additive run metadata for UI transparency and history restore."""
    effective_budget = budget or DecisionEngine()._infer_budgets(df)
    tx_rows = [
        {
            "date": str(pd.to_datetime(row["date"]).date()),
            "merchant": str(row.get("merchant", "Unknown")),
            "category": str(row.get("category", "other")),
            "amount": round(float(row.get("amount", 0.0)), 2),
        }
        for _, row in df.sort_values("date").iterrows()
    ]
    truncated = len(tx_rows) > ANALYSIS_MAX_TRANSACTIONS
    tx_items = tx_rows[:ANALYSIS_MAX_TRANSACTIONS]

    categories_present = sorted(
        {str(c) for c in df.get("category", pd.Series(dtype=str)).dropna().tolist()}
    )

    thresholds = {
        "forecast_warning_pct": 0.10,
        "min_anomaly_score": 0.50,
        "min_bulk_buy_confidence": 0.60,
        "evidence_window_days": 30,
        "merchant_min_txns": 3,
    }

    model_versions = {
        "forecast": "forecast_transformer.pt",
        "anomaly": "anomaly_transformer.pt",
        "bulk_buy": "bulkbuy_transformer.pt",
        "preprocessor": "preprocessor.pkl",
        "contrastive_anomaly": "contrastive_anomaly.pt" if (SAVE_DIR / "contrastive_anomaly.pt").exists() else None,
    }

    date_min = str(df["date"].min().date()) if not df.empty else None
    date_max = str(df["date"].max().date()) if not df.empty else None

    return {
        "run_id": run_id or str(uuid4()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "effective_budget": effective_budget,
        "thresholds": thresholds,
        "model_versions": model_versions,
        "input_stats": {
            "num_transactions": int(len(df)),
            "date_range": {"start": date_min, "end": date_max},
            "categories_present": categories_present,
        },
        "transactions_used": {
            "items": tx_items,
            "truncated": truncated,
            "total_available": len(tx_rows),
            "max_returned": ANALYSIS_MAX_TRANSACTIONS,
        },
        "partial_details": partial_details,
    }


def _run_intelligence_detectors(
    df: pd.DataFrame,
    model_outputs: dict,
    budget: dict | None,
) -> tuple[list[Suggestion], dict]:
    """
    Run the pure-Python intelligence detectors (no GPU).
    Returns (additional_suggestions, intel_model_outputs_fragment).
    Never raises — detectors are advisory.
    """
    try:
        from models.intelligence.life_event_detector import LifeEventDetector
        from models.intelligence.behavioral_bias_detector import BehavioralBiasDetector
        from models.intelligence.cash_crunch_predictor import CashCrunchPredictor
        from models.intelligence.goal_inferencer import GoalInferencer

        life_events       = LifeEventDetector().detect(df)
        behavioral_biases = BehavioralBiasDetector().detect_all(df)
        forecast_cats = {
            cat: model_outputs.get("forecast", {}).get(cat, {}).get("point", 0.0)
            for cat in CATEGORIES
        } if "forecast" in model_outputs else {}
        cash_crunch   = CashCrunchPredictor().predict(df, forecast_cats)
        inferred_goal = GoalInferencer().infer(df)

        engine = DecisionEngine(monthly_budget_per_category=budget or {})
        intel_suggestions = engine.generate_suggestions(
            user_transactions=df,
            life_events=life_events,
            behavioral_biases=behavioral_biases,
            cash_crunch=cash_crunch,
            inferred_goal=inferred_goal,
        )
        intel_fragment = {
            "life_events":             life_events,
            "behavioral_biases":       behavioral_biases,
            "cash_crunch_danger_dates": cash_crunch.get("danger_dates", []),
            "inferred_goal":           inferred_goal.get("goal_name") if inferred_goal else None,
        }
        return intel_suggestions, intel_fragment
    except Exception:
        return [], {}


def _run_all_models(df: pd.DataFrame, budget: dict | None) -> tuple[dict, list[Suggestion]]:
    reg = ModelRegistry
    if reg.prep is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Train models first (run training scripts).",
        )

    x = _df_to_tensor(df, reg.prep)
    model_outputs: dict = {}
    forecast_pred   = None
    forecast_lower  = None
    forecast_upper  = None
    anomaly_result  = None
    bulkbuy_result  = None

    if reg.forecast:
        mc_out = reg.forecast.mc_predict(x, n_samples=30)
        fp     = mc_out["mean"].cpu().numpy()[0]    # (NUM_CATEGORIES,)
        lower  = mc_out["lower"].cpu().numpy()[0]
        upper  = mc_out["upper"].cpu().numpy()[0]
        std    = mc_out["std"].cpu().numpy()[0]

        fp_blended, lower_blended, upper_blended = _calibrated_forecast(fp, lower, upper, df)

        forecast_pred  = fp_blended
        forecast_lower = lower_blended
        forecast_upper = upper_blended
        model_outputs["forecast"] = {
            cat: {
                "point": round(float(fp_blended[i]),    2),
                "lower": round(float(lower_blended[i]), 2),
                "upper": round(float(upper_blended[i]), 2),
                "std":   round(float(std[i]),            2),
            }
            for i, cat in enumerate(CATEGORIES)
        }

    anomaly_head = _run_anomaly_head(x, reg)
    if anomaly_head is not None:
        res = anomaly_head.pop("_res")
        anomaly_result = {k: v[0] for k, v in res.items()}
        anomaly_result["anomaly_score"] = anomaly_head["score"]
        model_outputs["anomaly"] = anomaly_head

    if reg.bulkbuy:
        res = reg.bulkbuy.predict(x)
        bulkbuy_result = {k: v[0] for k, v in res.items()}
        model_outputs["bulk_buy"] = {
            "probability":  float(res["bulk_prob"][0]),
            "recommend":    bool(res["recommend"][0]),
            "category":     int(res["target_category"][0]),
            "est_savings":  float(res["savings_estimate"][0]),
        }

    model_outputs["subscriptions"] = _build_subscription_inventory(df)

    engine = DecisionEngine(monthly_budget_per_category=budget or {})
    suggestions = engine.generate_suggestions(
        user_transactions=df,
        forecast_pred=forecast_pred,
        forecast_lower=forecast_lower,
        forecast_upper=forecast_upper,
        anomaly_result=anomaly_result,
        bulkbuy_result=bulkbuy_result,
        run_heuristic_rules=False,
    )
    return model_outputs, suggestions


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """Run all three models + intelligence detectors and return unified budget suggestions."""
    from api.db import (
        upsert_transactions,
        get_user_transactions,
        save_suggestions,
        get_user_anomaly_history,
        append_anomaly_score,
        update_transaction_categories,
    )

    df = _transactions_to_df(req.transactions)

    # Persist and merge historical transactions if user_id provided
    if req.user_id:
        try:
            upsert_transactions(req.user_id, [t.model_dump() for t in req.transactions])
            hist = get_user_transactions(req.user_id, limit_days=180)
            if hist:
                hist_df = pd.DataFrame(hist)
                hist_df["date"] = pd.to_datetime(hist_df["date"])
                # Merge: request transactions take precedence
                merged = pd.concat([hist_df, df], ignore_index=True)
                merged["merchant"] = merged["merchant"].str.lower().str.strip()
                merged = merged.sort_values("date").drop_duplicates(
                    subset=["date", "amount", "merchant"], keep="last"
                ).reset_index(drop=True)
                # Add required columns if missing
                for col in ["user_id", "is_anomaly", "is_bulk_buy"]:
                    if col not in merged.columns:
                        merged[col] = 0 if col == "user_id" else False
                df = merged
        except Exception:
            pass   # DB errors are non-fatal

    # Auto-categorize "other" transactions using description classifier
    pre_auto = df[["date", "amount", "merchant", "category"]].copy()
    df = _auto_categorize(df)

    # Apply user's confirmed merchant→category preferences (highest priority).
    if req.user_id:
        df = _apply_category_preferences(df, req.user_id)

    # Persist auto-categorized labels for real user history.
    if req.user_id and not df.empty:
        try:
            changed_mask = pre_auto["category"].astype(str) != df["category"].astype(str)
            if changed_mask.any():
                changed = df.loc[changed_mask, ["date", "amount", "merchant", "category"]].copy()
                changed["date"] = pd.to_datetime(changed["date"]).dt.strftime("%Y-%m-%d")
                updates = changed.to_dict(orient="records")
                update_transaction_categories(req.user_id, updates)
        except Exception:
            pass

    model_outputs, suggestions = _run_all_models(df, req.budget)

    # Intelligence detectors (pure Python/numpy — no GPU needed)
    intel_suggestions, intel_fragment = _run_intelligence_detectors(df, model_outputs, req.budget)
    if intel_suggestions:
        suggestions = suggestions + intel_suggestions
        suggestions.sort(key=lambda s: s.confidence, reverse=True)
    if intel_fragment:
        model_outputs["intelligence"] = intel_fragment

    date_range = (
        f"{df['date'].min().date()} → {df['date'].max().date()}"
        if not df.empty else "N/A"
    )

    # Attach explainability to forecast warnings
    try:
        from api.explainer import SuggestionExplainer
        explainer = SuggestionExplainer()
        x_for_explain = _df_to_tensor(df, ModelRegistry.prep) if ModelRegistry.prep else None
        suggestion_outs: list[SuggestionOut] = []
        for s in suggestions:
            d   = s.to_dict()
            exp = None
            if x_for_explain is not None:
                if s.type == "forecast_warning" and ModelRegistry.forecast is not None:
                    cat_idx = CATEGORIES.index(s.category) if s.category in CATEGORIES else 0
                    fc_model = (ModelRegistry.forecast.members[0]
                                if hasattr(ModelRegistry.forecast, "members")
                                else ModelRegistry.forecast)
                    exp = explainer.explain_forecast(fc_model, x_for_explain, cat_idx)
                elif s.type == "anomaly_alert" and ModelRegistry.anomaly is not None:
                    exp = explainer.explain_anomaly(ModelRegistry.anomaly, x_for_explain)
            suggestion_outs.append(SuggestionOut(**d, explanation=exp))
    except Exception:
        suggestion_outs = [SuggestionOut(**s.to_dict()) for s in suggestions]

    # Persist suggestions to DB
    if req.user_id:
        try:
            save_suggestions(req.user_id, [s.to_dict() for s in suggestions])
            # Persist anomaly score for per-user z-score
            if "anomaly" in model_outputs:
                append_anomaly_score(req.user_id, model_outputs["anomaly"]["score"])
        except Exception:
            pass

    # Peer comparison (requires trained VAE + cohort stats)
    peer_comparison = None
    reg = ModelRegistry
    if reg.user_vae is not None and reg.archetypes is not None and reg.cohort_stats is not None and reg.prep is not None:
        try:
            from models.intelligence.peer_comparator import PeerComparator
            x_vae     = _df_to_tensor(df, reg.prep)
            embedding = reg.user_vae.get_user_embedding(x_vae)   # (1, 64)

            date_range_days = max(1, (df["date"].max() - df["date"].min()).days + 1)
            user_monthly = {
                cat: float(df[df["category"] == cat]["amount"].sum() * 30 / date_range_days)
                for cat in CATEGORIES
            }
            peer_comparison = PeerComparator(reg.archetypes, reg.cohort_stats).compare(
                embedding, user_monthly
            )
            model_outputs["peer_comparison"] = peer_comparison
        except Exception:
            pass

    # Web deal search (DuckDuckGo, cached 24 h in SQLite)
    deals: list | None = None
    try:
        from rules.deal_finder import DealFinder
        deals = DealFinder().find(df, [s.to_dict() for s in suggestions], model_outputs)
    except Exception:
        pass

    analysis_context = _build_analysis_context(
        df=df,
        budget=req.budget,
        model_outputs=model_outputs,
    )

    return AnalyzeResponse(
        suggestions=suggestion_outs,
        num_transactions=len(df),
        date_range=date_range,
        model_outputs=model_outputs,
        analysis_context=analysis_context,
        peer_comparison=peer_comparison,
        deals=deals or None,
    )


@app.get("/users/{user_id}/history", dependencies=[Depends(_verify_api_key)])
def user_history(user_id: str, days: int = Query(default=90, ge=1, le=365)):
    """Return the last `days` days of persisted transactions for a user."""
    from api.db import get_user_transactions
    txns = get_user_transactions(user_id, limit_days=days)
    return {"user_id": user_id, "transactions": txns, "count": len(txns)}


class DealsRequest(BaseModel):
    transactions: list[Transaction] = Field(..., min_length=1, max_length=10000)
    suggestions:  list[dict] = Field(default_factory=list)
    user_id:      str | None = None


@app.post("/deals")
async def find_deals_endpoint(req: DealsRequest):
    """
    Re-run deal search for a set of transactions and suggestions.

    Used by the UI when restoring a history entry that has no cached deals.
    Results are served from the 24-hour SQLite cache when the same queries
    were already run during the original analysis, so this is usually fast.
    """
    from rules.deal_finder import DealFinder

    df = _transactions_to_df(req.transactions)
    deals = await asyncio.to_thread(DealFinder().find, df, req.suggestions, {})
    return {"deals": deals or []}


class CategoryPreferenceRequest(BaseModel):
    merchant:  str
    category:  str

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        if v not in _VALID_CATEGORIES:
            raise ValueError(f"Unknown category: {v!r}")
        return v


@app.post("/users/{user_id}/category-preference", status_code=204, dependencies=[Depends(_verify_api_key)])
def save_user_category_preference(user_id: str, req: CategoryPreferenceRequest):
    """Persist a user's confirmed merchant→category preference for future auto-categorization."""
    from api.db import save_category_preference
    save_category_preference(user_id, req.merchant, req.category)


@app.get("/users/{user_id}/category-preferences", dependencies=[Depends(_verify_api_key)])
def get_user_category_preferences(user_id: str):
    """Return all confirmed merchant→category preferences for a user."""
    from api.db import get_category_preferences
    prefs = get_category_preferences(user_id)
    return {"user_id": user_id, "preferences": prefs, "count": len(prefs)}


class CategorizeRequest(BaseModel):
    descriptions: list[str] = Field(..., min_length=1, max_length=500)
    merchants: list[str] | None = None


_SETFIT_CONF_SCALAR: float = 0.80   # SetFit logistic head is overconfident; scale before comparisons

def _run_classifier_choice(clf, classifier_text: str, source: str) -> dict | None:
    if clf is None or not classifier_text.strip():
        return None
    probs = clf.predict_proba([classifier_text])[0]
    cat = max(probs, key=probs.get)
    raw_conf = float(probs[cat])
    conf = raw_conf * _SETFIT_CONF_SCALAR if source == "setfit" else raw_conf
    return {
        "source": source,
        "category": cat,
        "confidence": round(conf, 4),
        "probabilities": probs,
    }


def _build_heuristic_choice(
    *,
    description: str,
    merchant: str,
    classifier_text: str,
    primary_choice: dict | None,
    secondary_choice: dict | None = None,
) -> tuple[dict, list[dict], dict | None]:
    if _is_p2p_merchant(description) or _is_p2p_merchant(merchant):
        # Hard-block: Apple Cash, Zelle, Cash App — no service context is possible
        _is_hard = _is_p2p_hard_block(description) or _is_p2p_hard_block(merchant)
        # Soft-block passthrough: PayPal/Venmo/etc. — allow when ML found a real service
        _p2p_can_identify = not _is_hard and any(
            c and c.get("category") != "other" and float(c.get("confidence", 0)) > 0.30
            for c in [primary_choice, secondary_choice]
        )
        if not _p2p_can_identify:
            return (
                {
                    "source": "heuristic",
                    "category": "other",
                    "confidence": 1.0,
                    "decision_source": "p2p_blocklist",
                    "probabilities": primary_choice["probabilities"] if primary_choice else {},
                },
                [],
                None,
            )

    # Deterministic keyword rules — highest priority, no ML involved
    kw_cat = _keyword_classify(classifier_text)
    if kw_cat is not None:
        return (
            {
                "source": "heuristic",
                "category": kw_cat,
                "confidence": 1.0,
                "decision_source": "keyword_rule",
                "probabilities": primary_choice["probabilities"] if primary_choice else {},
            },
            [],
            None,
        )

    if primary_choice is None:
        return (
            {
                "source": "heuristic",
                "category": "other",
                "confidence": 0.0,
                "decision_source": "no_model",
                "probabilities": {},
            },
            [],
            None,
        )

    chosen_cat  = primary_choice["category"]
    chosen_conf = _penalize_subscription_fp(
        classifier_text, chosen_cat, float(primary_choice["confidence"])
    )
    decision_source = primary_choice["source"]
    web_evidence: list[dict] = []
    feedback_row: dict | None = None

    # Use secondary (TF-IDF) when it's more confident than primary (SetFit), or primary says "other"
    if secondary_choice is not None:
        sec_cat  = secondary_choice["category"]
        sec_conf = _penalize_subscription_fp(
            classifier_text, sec_cat, float(secondary_choice["confidence"])
        )
        if sec_cat != "other" and (
            chosen_cat == "other"
            or sec_conf > chosen_conf + 0.10
        ):
            chosen_cat  = sec_cat
            chosen_conf = sec_conf
            decision_source = secondary_choice["source"]

    if chosen_conf < 0.30 or chosen_cat == "other":
        web_cat, web_conf, web_results = _infer_category_from_search(classifier_text)
        if web_cat is not None and web_conf >= max(0.35, chosen_conf):
            chosen_cat = web_cat
            chosen_conf = float(web_conf)
            decision_source = "web_fallback"
            web_evidence = web_results[:2]
            feedback_row = {
                "description": classifier_text,
                "category": web_cat,
                "confidence": web_conf,
                "source": "web_fallback",
            }
        else:
            ollama_cat, ollama_conf = _ollama_classify(classifier_text)
            if ollama_cat and ollama_cat != "other":
                chosen_cat = ollama_cat
                chosen_conf = ollama_conf
                decision_source = "ollama"
                feedback_row = {
                    "description": classifier_text,
                    "category": ollama_cat,
                    "confidence": ollama_conf,
                    "source": "ollama",
                }

    # Last resort: never leave as "other" if any ML model has a non-"other" prediction.
    # This ensures all transactions get a category; user can override via AI Compare.
    if chosen_cat == "other":
        candidates = [
            c for c in [primary_choice, secondary_choice]
            if c and c.get("category") != "other"
        ]
        if candidates:
            best = max(
                candidates,
                key=lambda c: _penalize_subscription_fp(
                    classifier_text, c["category"], float(c["confidence"])
                ),
            )
            chosen_cat = best["category"]
            chosen_conf = _penalize_subscription_fp(
                classifier_text, chosen_cat, float(best["confidence"])
            )
            decision_source = best["source"]

    return (
        {
            "source": "heuristic",
            "category": chosen_cat,
            "confidence": round(float(chosen_conf), 4),
            "decision_source": decision_source,
            "probabilities": primary_choice["probabilities"],
        },
        web_evidence,
        feedback_row,
    )


@app.post("/categorize")
def categorize(req: CategorizeRequest):
    """
    Auto-categorize raw transaction description strings.

    Returns the predicted category and per-category probabilities for each description.
    Requires train_description_classifier.py to have been run first.
    """
    reg = ModelRegistry
    if reg.description_clf is None:
        raise HTTPException(
            status_code=503,
            detail="Description classifier not loaded. Run: python training/train_description_classifier.py",
        )

    from api.db import save_auto_category_feedback

    merchants = req.merchants or [""] * len(req.descriptions)
    if len(merchants) != len(req.descriptions):
        raise HTTPException(status_code=422, detail="'merchants' must match descriptions length.")

    classifier_texts = [
        combine_transaction_text(merchant=str(m), description=str(d))
        for d, m in zip(req.descriptions, merchants)
    ]
    results: list[dict] = []
    feedback_rows: list[dict] = []

    for d, m, classifier_text in zip(req.descriptions, merchants, classifier_texts):
        setfit_choice = _run_classifier_choice(reg.setfit_description_clf, classifier_text, "setfit")
        tfidf_choice = _run_classifier_choice(reg.tfidf_description_clf, classifier_text, "tfidf")

        primary_choice = setfit_choice if reg.active_description_backend == "setfit" else tfidf_choice
        if primary_choice is None:
            primary_choice = setfit_choice or tfidf_choice

        secondary_choice = (
            tfidf_choice if reg.active_description_backend == "setfit"
            else setfit_choice
        )

        heuristic_choice, web_evidence, feedback_row = _build_heuristic_choice(
            description=str(d),
            merchant=str(m),
            classifier_text=classifier_text,
            primary_choice=primary_choice,
            secondary_choice=secondary_choice,
        )
        if feedback_row is not None:
            feedback_rows.append(feedback_row)

        ollama_choice = None
        if heuristic_choice["decision_source"] == "ollama":
            ollama_choice = {
                "source": "ollama",
                "category": heuristic_choice["category"],
                "confidence": heuristic_choice["confidence"],
            }
        else:
            ollama_cat, ollama_disp_conf = _ollama_classify(classifier_text)
            if ollama_cat:
                ollama_choice = {
                    "source": "ollama",
                    "category": ollama_cat,
                    "confidence": ollama_disp_conf if ollama_cat != "other" else 0.0,
                }

        classifications = [heuristic_choice]
        if setfit_choice is not None:
            classifications.append(setfit_choice)
        if tfidf_choice is not None:
            classifications.append(tfidf_choice)
        if ollama_choice is not None:
            classifications.append(ollama_choice)

        results.append(
            {
                "description": d,
                "merchant": m,
                "predicted_category": heuristic_choice["category"],
                "confidence": heuristic_choice["confidence"],
                "probabilities": heuristic_choice["probabilities"],
                "source": "heuristic",
                "decision_source": heuristic_choice["decision_source"],
                "active_backend": reg.active_description_backend,
                "web_evidence": web_evidence,
                "classifications": classifications,
            }
        )

    if feedback_rows:
        try:
            save_auto_category_feedback(feedback_rows)
        except Exception:
            pass

    return {
        "results": results
    }


@app.get("/users/{user_id}/suggestions", dependencies=[Depends(_verify_api_key)])
def user_suggestions(user_id: str, limit: int = 30):
    """Return the last `limit` persisted suggestions for a user."""
    from api.db import get_user_suggestions
    sugs = get_user_suggestions(user_id, limit=limit)
    return {"user_id": user_id, "suggestions": sugs, "count": len(sugs)}


# ── Custom categories ─────────────────────────────────────────────────────────

class UserCategoryRequest(BaseModel):
    name:        str
    description: str  = ""
    color:       str  = "#4a6580"
    icon:        str  = "📦"


class CategoryExampleRequest(BaseModel):
    category:    str
    merchant:    str
    description: str = ""


@app.get("/users/{user_id}/categories", dependencies=[Depends(_verify_api_key)])
def list_user_categories(user_id: str):
    """Return all custom categories defined by this user."""
    from api.db import get_user_categories
    cats = get_user_categories(user_id)
    return {"user_id": user_id, "categories": cats}


@app.post("/users/{user_id}/categories")
def create_user_category(user_id: str, req: UserCategoryRequest):
    """Create or update a custom spending category for a user."""
    from api.db import upsert_user_category
    upsert_user_category(
        user_id,
        name=req.name,
        description=req.description,
        color=req.color,
        icon=req.icon,
    )
    return {"status": "ok", "category": req.name.strip().lower()}


@app.delete("/users/{user_id}/categories/{category_name}")
def delete_user_category_endpoint(user_id: str, category_name: str):
    """Delete a custom category and its labeled examples."""
    from api.db import delete_user_category
    delete_user_category(user_id, category_name)
    return {"status": "deleted", "category": category_name}


@app.post("/users/{user_id}/categories/examples")
def add_example(user_id: str, req: CategoryExampleRequest):
    """
    Tag a merchant/description as belonging to a category.
    Accepted for both built-in and custom categories.
    This data is used when retraining the description classifier (--from-db).
    """
    from api.db import add_category_example
    add_category_example(
        user_id,
        category=req.category,
        merchant=req.merchant,
        description=req.description,
    )
    # If Ollama is available, also try to predict now so the user sees instant feedback
    predicted, _ = _ollama_classify(
        f"{req.merchant} {req.description}".strip(),
        categories=[req.category],
    )
    return {
        "status": "saved",
        "category": req.category,
        "merchant": req.merchant,
        "ollama_confirmed": predicted == req.category if predicted else None,
    }


@app.get("/users/{user_id}/categories/examples")
def list_examples(user_id: str, category: str | None = None):
    """Return labeled examples for this user."""
    from api.db import get_category_examples
    return {"examples": get_category_examples(user_id, category)}


# ── Ollama integration ────────────────────────────────────────────────────────

_OLLAMA_AVAILABLE: bool | None = None  # None = unchecked
_OLLAMA_LAST_CHECK: float = 0.0
_OLLAMA_RECHECK_INTERVAL: float = 300.0  # re-probe after 5 min if previously down
_WEB_SEARCH_AVAILABLE: bool | None = None
_WEB_SEARCH_LAST_CHECK: float = 0.0
_WEB_SEARCH_RECHECK_INTERVAL: float = 300.0  # same TTL as Ollama


def _check_web_search() -> bool:
    """Probe outbound HTTP; re-probe every 5 min if previously unavailable."""
    global _WEB_SEARCH_AVAILABLE, _WEB_SEARCH_LAST_CHECK
    now = time.monotonic()
    if _WEB_SEARCH_AVAILABLE is True:
        return True
    if _WEB_SEARCH_AVAILABLE is False and (now - _WEB_SEARCH_LAST_CHECK) < _WEB_SEARCH_RECHECK_INTERVAL:
        return False
    try:
        import urllib.request as _ur
        _ur.urlopen("https://duckduckgo.com", timeout=2)
        _WEB_SEARCH_AVAILABLE = True
    except Exception:
        _WEB_SEARCH_AVAILABLE = False
        print("[API] Outbound web search unavailable — will retry in 5 min")
    _WEB_SEARCH_LAST_CHECK = now
    return _WEB_SEARCH_AVAILABLE


def _check_ollama(host: str = "http://localhost:11434") -> bool:
    """Probe Ollama; re-probe every 5 min if previously unavailable."""
    global _OLLAMA_AVAILABLE, _OLLAMA_LAST_CHECK
    now = time.monotonic()
    if _OLLAMA_AVAILABLE is True:
        return True
    if _OLLAMA_AVAILABLE is False and (now - _OLLAMA_LAST_CHECK) < _OLLAMA_RECHECK_INTERVAL:
        return False
    try:
        import urllib.request as _ur
        with _ur.urlopen(f"{host}/api/tags", timeout=1) as r:
            _OLLAMA_AVAILABLE = r.status == 200
    except Exception:
        _OLLAMA_AVAILABLE = False
    _OLLAMA_LAST_CHECK = now
    if not _OLLAMA_AVAILABLE:
        print("[API] Ollama not available — will retry in 5 min")
    return _OLLAMA_AVAILABLE


def _ollama_classify(
    text: str,
    categories: list[str] | None = None,
    model: str = "llama3",
    host: str = "http://localhost:11434",
) -> tuple[str | None, float]:
    """
    Zero-shot categorization.  Tries Ollama first; falls back to local llama-cpp-python.
    Returns (category, confidence) where confidence is LLM-reported (0–1) or 0.65 default.
    Returns (None, 0.0) on failure / unavailability.
    """
    if not text or not text.strip():
        return None, 0.0
    all_cats = list(categories or []) + CATEGORIES
    seen: set[str] = set()
    cat_list = [c for c in all_cats if not (c in seen or seen.add(c))]  # type: ignore[arg-type]

    # ── Ollama (if running) ───────────────────────────────────────────────────
    if _check_ollama(host):
        cats_str = ", ".join(cat_list)
        prompt = (
            f"Classify this bank transaction into exactly one category.\n"
            f"Categories: {cats_str}\n"
            f"Reply with ONLY: <category> <0-100>\n"
            f"Transaction: {text.strip()}\n"
            f"Response:"
        )
        try:
            import urllib.request, json as _json, math as _math
            payload = _json.dumps({
                "model": model,
                "prompt": prompt,
                "stream": False,
                "logprobs": True,
                "options": {"temperature": 0, "num_predict": 6},
            }).encode()
            req = urllib.request.Request(
                f"{host}/api/generate", data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = _json.loads(resp.read())
            raw = str(data.get("response", "")).strip().lower()
            words = [w.strip(".,!?;:") for w in raw.split()]
            found_cat: str | None = None
            found_conf: float = 0.65
            for i, word in enumerate(words):
                for c in cat_list:
                    if c.lower() == word:
                        found_cat = c
                        if i + 1 < len(words):
                            try:
                                found_conf = min(int(words[i + 1]) / 100.0, 1.0)
                            except (ValueError, TypeError):
                                pass
                        break
                if found_cat:
                    break
            if found_cat:
                # best-effort: override self-report with logprob-derived confidence
                logprobs = data.get("logprobs")
                if logprobs and isinstance(logprobs, list) and len(logprobs) > 0:
                    try:
                        lp_values = [
                            lp if isinstance(lp, (int, float)) else list(lp.values())[0]
                            for lp in logprobs[:len(found_cat.split())]
                            if lp is not None
                        ]
                        if lp_values:
                            found_conf = min(_math.exp(sum(lp_values) / len(lp_values)), 1.0)
                    except Exception:
                        pass
                return found_cat, found_conf
        except Exception:
            pass

    # ── Local llama-cpp-python fallback ──────────────────────────────────────
    try:
        from models.intelligence import local_llm  # noqa: PLC0415
        cat = local_llm.classify(text, cat_list)
        return (cat, 0.65) if cat else (None, 0.0)
    except Exception:
        return None, 0.0


def _ollama_generate(
    prompt: str,
    model: str = "llama3",
    host: str = "http://localhost:11434",
    timeout: int = 30,
) -> str | None:
    """
    Free-form text generation.  Tries Ollama first; falls back to local llama-cpp-python.
    Returns generated text or None on failure / unavailability.
    """
    if not prompt or not prompt.strip():
        return None

    # ── Ollama (if running) ───────────────────────────────────────────────────
    if _check_ollama(host):
        try:
            import urllib.request, json as _json
            payload = _json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
            req = urllib.request.Request(
                f"{host}/api/generate", data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = _json.loads(resp.read())
            result = str(data.get("response", "")).strip()
            if result:
                return result
        except Exception:
            pass

    # ── Local llama-cpp-python fallback ──────────────────────────────────────
    try:
        from models.intelligence import local_llm  # noqa: PLC0415
        return local_llm.generate(prompt, max_tokens=200, temperature=0.6)
    except Exception:
        return None


class NarrativeRequest(BaseModel):
    suggestions:      list[dict]
    model_outputs:    dict
    date_range:       str = ""
    num_transactions: int = 0
    peer_comparison:  dict | None = None


@app.post("/narrative")
def narrative(req: NarrativeRequest):
    """
    Generate a plain-English financial health summary.
    Tries Ollama first, then falls back to local llama-cpp-python.
    Returns {"narrative": str} or 503 if no LLM backend is available.
    """

    top_sugs = req.suggestions[:6]
    sug_lines = "\n".join(
        f"- [{s.get('type','?')}] {s.get('category','?')}: {s.get('message','')}"
        for s in top_sugs
    ) or "No specific issues detected."

    forecast_lines = ""
    if "forecast" in req.model_outputs:
        fc = req.model_outputs["forecast"]
        top_cats = sorted(
            fc.items(),
            key=lambda kv: (kv[1].get("point", 0) if isinstance(kv[1], dict) else kv[1]),
            reverse=True,
        )[:5]
        forecast_lines = "\n".join(
            f"- {cat}: ${(v['point'] if isinstance(v, dict) else v):.0f}/mo predicted"
            for cat, v in top_cats
        )

    archetype_line = ""
    if req.peer_comparison:
        arch = req.peer_comparison.get("archetype", "")
        msg  = req.peer_comparison.get("message", "")
        if arch:
            archetype_line = f"Financial archetype: {arch}. {msg}"

    sub_line = ""
    subs = req.model_outputs.get("subscriptions", {})
    if subs.get("total_monthly", 0) > 0:
        sub_line = f"Recurring subscriptions: ${subs['total_monthly']:.2f}/month ({subs.get('trap_count', 0)} potential traps)."

    prompt = (
        f"You are a concise personal finance advisor. Write a 3-4 sentence plain-English "
        f"financial health summary based on the data below. Be specific with dollar amounts. "
        f"End with exactly one actionable recommendation. No bullet points, no markdown, no headers.\n\n"
        f"Period: {req.date_range} ({req.num_transactions} transactions)\n"
        f"{archetype_line}\n"
        f"{sub_line}\n\n"
        f"Model findings:\n{sug_lines}\n\n"
        f"Predicted next-month spending:\n{forecast_lines}"
    )

    text = _ollama_generate(prompt)
    if text is None:
        raise HTTPException(503, "No LLM backend available — install llama-cpp-python or start Ollama.")
    return {"narrative": text}


@app.get("/users")
def list_all_users():
    """Return all registered users with transaction counts and date ranges."""
    from api.db import list_users
    return {"users": list_users()}


@app.post("/users/{user_id}", status_code=201)
def create_user_endpoint(user_id: str):
    """Explicitly register a user_id (also happens implicitly on first analyze)."""
    from api.db import upsert_user
    user_id = user_id.strip()
    if not user_id:
        raise HTTPException(400, "user_id must not be empty.")
    upsert_user(user_id)
    return {"user_id": user_id, "status": "created"}


@app.get("/users/{user_id}/vae-embedding")
def user_vae_embedding(user_id: str, days: int = Query(default=60, ge=7, le=365)):
    """
    Encode a user's stored transaction history into their 64-dim latent embedding.

    Useful for inspecting the VAE latent space without re-uploading transactions.
    Returns embedding, financial archetype, and nearest-cohort category spend.
    """
    from api.db import get_user_transactions
    reg = ModelRegistry
    if reg.user_vae is None or reg.prep is None:
        raise HTTPException(503, "User VAE not loaded — run train_user_vae.py first.")

    rows = get_user_transactions(user_id, limit_days=days)
    if not rows:
        raise HTTPException(404, f"No transactions found for user '{user_id}'.")

    txns = [Transaction(**r) for r in rows]
    df   = _transactions_to_df(txns)
    x    = _df_to_tensor(df, reg.prep)

    embedding = reg.user_vae.get_user_embedding(x)   # (1, 64)
    emb_list  = embedding[0].cpu().tolist()

    archetype = "Unknown"
    if reg.archetypes is not None:
        import torch.nn.functional as F
        centers = reg.archetypes["centers"].to(embedding.device)
        names   = reg.archetypes["names"]
        emb_n   = F.normalize(embedding, dim=-1)
        cen_n   = F.normalize(centers,   dim=-1)
        sims    = (emb_n @ cen_n.T).squeeze(0)
        archetype = names[int(sims.argmax())]

    cohort_stats: dict = {}
    if reg.cohort_embs is not None:
        import torch.nn.functional as F
        indices, _ = reg.user_vae.nearest_cohort(
            embedding, reg.cohort_embs.to(embedding.device), top_k=5
        )
        near_embs = reg.cohort_embs[indices[0]].to(embedding.device)
        with torch.no_grad():
            near_recon = reg.user_vae.decode(near_embs)
        avg_monthly = near_recon.mean(dim=1).mean(dim=0).cpu().tolist()
        cohort_stats = {
            cat: round(float(avg_monthly[i]) * 30, 2)
            for i, cat in enumerate(CATEGORIES)
        }

    # Auto-save a monthly snapshot so drift can be tracked over time
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    try:
        from api.db import save_vae_snapshot
        save_vae_snapshot(user_id, month, emb_list, archetype)
    except Exception:
        pass

    return {
        "user_id":               user_id,
        "transaction_count":     len(rows),
        "embedding":             emb_list,
        "embedding_dim":         len(emb_list),
        "financial_archetype":   archetype,
        "nearest_cohort_stats":  cohort_stats,
    }


@app.get("/users/{user_id}/vae-drift")
def user_vae_drift(user_id: str):
    """
    Return the user's monthly VAE embedding snapshot history with drift metrics.

    Drift is cosine distance between consecutive monthly embeddings, expressed
    as a 0-100 percentage.  Each month also reports drift vs. 3 and 6 months
    prior so you can distinguish short-term spikes from long-term shifts.
    """
    from api.db import get_vae_snapshots
    import math

    snapshots = get_vae_snapshots(user_id)
    if not snapshots:
        return {"user_id": user_id, "snapshots": [], "has_drift_data": False}

    def _cosine_dist(a: list[float], b: list[float]) -> float:
        dot  = sum(x * y for x, y in zip(a, b))
        na   = math.sqrt(sum(x * x for x in a))
        nb   = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return 1.0 - dot / (na * nb)

    def _drift_pct(a, b) -> float:
        return round(_cosine_dist(a, b) * 100, 1)

    result = []
    for i, snap in enumerate(snapshots):
        entry: dict = {
            "month":     snap["snapshot_month"],
            "archetype": snap["archetype"],
        }
        if i == 0:
            entry["drift_from_prev"]   = None
            entry["drift_from_3mo"]    = None
            entry["drift_from_6mo"]    = None
        else:
            entry["drift_from_prev"] = _drift_pct(
                snapshots[i - 1]["embedding"], snap["embedding"]
            )
            entry["drift_from_3mo"] = _drift_pct(
                snapshots[max(0, i - 3)]["embedding"], snap["embedding"]
            ) if i >= 3 else None
            entry["drift_from_6mo"] = _drift_pct(
                snapshots[max(0, i - 6)]["embedding"], snap["embedding"]
            ) if i >= 6 else None
        result.append(entry)

    # Overall drift: first snapshot → latest
    total_drift = _drift_pct(snapshots[0]["embedding"], snapshots[-1]["embedding"]) \
        if len(snapshots) > 1 else 0.0

    return {
        "user_id":      user_id,
        "snapshots":    result,
        "total_drift":  total_drift,
        "has_drift_data": len(snapshots) > 1,
    }


@app.get("/users/{user_id}/vae-surprise")
def user_vae_surprise(user_id: str, days: int = Query(default=30, ge=7, le=90)):
    """
    Holistic anomaly: how surprised is the VAE by the user's recent spending?

    Reconstructs the user's last `days` of transactions and returns the MSE
    between the real category-spend distribution and the VAE's reconstruction.
    High surprise = this period looks unlike anything in the model's training data.
    Also returns per-category reconstruction error so you can see *where* the
    surprise is coming from.
    """
    from api.db import get_user_transactions
    reg = ModelRegistry
    if reg.user_vae is None or reg.prep is None:
        raise HTTPException(503, "User VAE not loaded — run train_user_vae.py first.")

    rows = get_user_transactions(user_id, limit_days=days)
    if not rows:
        raise HTTPException(404, f"No transactions found for user '{user_id}'.")

    txns = [Transaction(**r) for r in rows]
    df   = _transactions_to_df(txns)
    x    = _df_to_tensor(df, reg.prep)

    reg.user_vae.eval()
    with torch.no_grad():
        out = reg.user_vae.forward(x)      # includes recon (1, seq_len, num_cats)

    recon = out["recon"][0]                # (seq_len, num_cats)

    # Aggregate real spend per category from the raw transactions
    cat_spend = {cat: 0.0 for cat in CATEGORIES}
    for t in txns:
        if t.category in cat_spend:
            cat_spend[t.category] += t.amount

    total = sum(cat_spend.values()) or 1.0
    real_dist  = torch.tensor([cat_spend[c] / total for c in CATEGORIES])
    recon_dist = recon.mean(dim=0).cpu()   # average reconstructed daily spend
    recon_dist = recon_dist / (recon_dist.sum() + 1e-8)

    mse_per_cat = ((real_dist - recon_dist) ** 2).tolist()
    overall_mse = float(sum(mse_per_cat))

    # Surprise score 0-100: scale so MSE=0.05 → 50 (calibrated heuristic)
    surprise_score = min(100.0, round(overall_mse * 1000, 1))

    real_list  = real_dist.tolist()
    recon_list = recon_dist.tolist()

    return {
        "user_id":         user_id,
        "days_analyzed":   days,
        "surprise_score":  surprise_score,
        "surprise_label":  "High" if surprise_score > 60 else "Moderate" if surprise_score > 25 else "Low",
        "per_category_error": {
            cat: round(mse_per_cat[i], 4)
            for i, cat in enumerate(CATEGORIES)
        },
        "per_category_actual": {
            cat: round(real_list[i], 4)
            for i, cat in enumerate(CATEGORIES)
        },
        "per_category_expected": {
            cat: round(recon_list[i], 4)
            for i, cat in enumerate(CATEGORIES)
        },
        "interpretation": (
            "Your spending pattern this period looks very unusual compared to your typical habits."
            if surprise_score > 60 else
            "Some categories are spending differently than usual."
            if surprise_score > 25 else
            "Your spending pattern is consistent with your typical habits."
        ),
    }


@app.delete("/users/{user_id}", dependencies=[Depends(_verify_api_key)])
def delete_user_data(user_id: str):
    """Delete all data for a user (GDPR compliance)."""
    from api.db import delete_user
    delete_user(user_id)
    return {"user_id": user_id, "status": "deleted"}


@app.delete("/users/{user_id}/transactions")
def clear_user_transactions_endpoint(user_id: str):
    """Delete all stored transactions for a user without removing the user record."""
    from api.db import clear_user_transactions
    deleted = clear_user_transactions(user_id)
    return {"user_id": user_id, "deleted": deleted, "status": "cleared"}


@app.post("/forecast")
def forecast_only(req: AnalyzeRequest):
    """Spending forecast for next 30 days per category, with 80% prediction intervals."""
    df  = _transactions_to_df(req.transactions)
    reg = ModelRegistry
    if reg.forecast is None or reg.prep is None:
        raise HTTPException(503, "Forecast model not loaded.")
    x = _df_to_tensor(df, reg.prep)
    mc_out = reg.forecast.mc_predict(x, n_samples=30)
    fp    = mc_out["mean"].cpu().numpy()[0]
    lower = mc_out["lower"].cpu().numpy()[0]
    upper = mc_out["upper"].cpu().numpy()[0]
    std   = mc_out["std"].cpu().numpy()[0]
    fp, lower, upper = _calibrated_forecast(fp, lower, upper, df)
    from data.preprocessor import CATEGORIES
    return {
        "forecast": {
            cat: {
                "point": round(float(fp[i]),    2),
                "lower": round(float(lower[i]), 2),
                "upper": round(float(upper[i]), 2),
                "std":   round(float(std[i]),   2),
            }
            for i, cat in enumerate(CATEGORIES)
        }
    }


@app.post("/anomaly")
def anomaly_only(req: AnalyzeRequest):
    """Anomaly detection on recent transactions."""
    df  = _transactions_to_df(req.transactions)
    reg = ModelRegistry
    if reg.anomaly is None or reg.prep is None:
        raise HTTPException(503, "Anomaly model not loaded.")
    x   = _df_to_tensor(df, reg.prep)
    res = reg.anomaly.predict(x, threshold=reg.threshold)
    return {
        "anomaly_score": float(res["anomaly_score"][0]),
        "is_anomaly":    bool(res["is_anomaly"][0]),
        "anomaly_class": int(res["anomaly_class"][0]),
        "class_probs":   res["class_probs"][0].tolist(),
    }


@app.post("/recommend")
def recommend_only(req: AnalyzeRequest):
    """Bulk-buy recommendation from transaction history."""
    df  = _transactions_to_df(req.transactions)
    reg = ModelRegistry
    if reg.bulkbuy is None or reg.prep is None:
        raise HTTPException(503, "Bulk-buy model not loaded.")
    x   = _df_to_tensor(df, reg.prep)
    res = reg.bulkbuy.predict(x)
    from data.preprocessor import CATEGORIES
    cat_idx = int(res["target_category"][0])
    return {
        "recommend":        bool(res["recommend"][0]),
        "bulk_probability": float(res["bulk_prob"][0]),
        "target_category":  CATEGORIES[cat_idx] if 0 <= cat_idx < len(CATEGORIES) else "other",
        "savings_estimate": float(res["savings_estimate"][0]),
        "category_probs":   {
            cat: float(res["category_probs"][0][i])
            for i, cat in enumerate(CATEGORIES)
        },
    }


@app.post("/analyze/stream", include_in_schema=True)
async def analyze_stream(req: AnalyzeRequest):
    """
    SSE endpoint: streams model-by-model progress events then the final result.

    Each event is a JSON object on a `data:` line.
    Status values: preprocessing → forecast → anomaly → bulkbuy → suggestions → complete | error
    """
    from api.db import (
        upsert_transactions,
        get_user_transactions,
        update_transaction_categories,
    )

    df = _transactions_to_df(req.transactions)

    # Persist and merge historical transactions — same logic as sync /analyze
    if req.user_id:
        try:
            upsert_transactions(req.user_id, [t.model_dump() for t in req.transactions])
            hist = get_user_transactions(req.user_id, limit_days=180)
            if hist:
                hist_df = pd.DataFrame(hist)
                hist_df["date"] = pd.to_datetime(hist_df["date"])
                merged = pd.concat([hist_df, df], ignore_index=True)
                merged["merchant"] = merged["merchant"].str.lower().str.strip()
                merged = merged.sort_values("date").drop_duplicates(
                    subset=["date", "amount", "merchant"], keep="last"
                ).reset_index(drop=True)
                for col in ["user_id", "is_anomaly", "is_bulk_buy"]:
                    if col not in merged.columns:
                        merged[col] = 0 if col == "user_id" else False
                df = merged
        except Exception:
            pass

    pre_auto = df[["date", "amount", "merchant", "category"]].copy()
    df = _auto_categorize(df)

    if req.user_id:
        df = _apply_category_preferences(df, req.user_id)

    # Persist auto-categorized label changes
    if req.user_id and not df.empty:
        try:
            changed_mask = pre_auto["category"].astype(str) != df["category"].astype(str)
            if changed_mask.any():
                changed = df.loc[changed_mask, ["date", "amount", "merchant", "category"]].copy()
                changed["date"] = pd.to_datetime(changed["date"]).dt.strftime("%Y-%m-%d")
                update_transaction_categories(req.user_id, changed.to_dict(orient="records"))
        except Exception:
            pass

    async def _generate() -> AsyncGenerator[str, None]:
        def _evt(payload: dict) -> str:
            return f"data: {json.dumps(payload)}\n\n"

        try:
            yield _evt({"status": "preprocessing", "message": "Preprocessing transactions…"})
            await asyncio.sleep(0)  # flush to client before blocking CPU work

            reg = ModelRegistry
            if reg.prep is None:
                yield _evt({"status": "error", "message": "Models not loaded — run the training scripts first."})
                return

            x = _df_to_tensor(df, reg.prep)
            model_outputs: dict = {}
            forecast_pred   = None
            forecast_lower  = None
            forecast_upper  = None
            anomaly_result  = None
            bulkbuy_result  = None

            if reg.forecast:
                yield _evt({"status": "forecast", "message": "Running spending forecast…"})
                await asyncio.sleep(0)
                mc_out = reg.forecast.mc_predict(x, n_samples=30)
                fp     = mc_out["mean"].cpu().numpy()[0]
                lower  = mc_out["lower"].cpu().numpy()[0]
                upper  = mc_out["upper"].cpu().numpy()[0]
                std    = mc_out["std"].cpu().numpy()[0]
                fp_cal, lower_cal, upper_cal = _calibrated_forecast(fp, lower, upper, df)
                forecast_pred  = fp_cal
                forecast_lower = lower_cal
                forecast_upper = upper_cal
                model_outputs["forecast"] = {
                    cat: {
                        "point": round(float(fp_cal[i]),    2),
                        "lower": round(float(lower_cal[i]), 2),
                        "upper": round(float(upper_cal[i]), 2),
                        "std":   round(float(std[i]),       2),
                    }
                    for i, cat in enumerate(CATEGORIES)
                }

            anomaly_head = _run_anomaly_head(x, reg)
            if anomaly_head is not None:
                yield _evt({"status": "anomaly", "message": "Checking for anomalies…"})
                await asyncio.sleep(0)
                res = anomaly_head.pop("_res")
                anomaly_result = {k: v[0] for k, v in res.items()}
                anomaly_result["anomaly_score"] = anomaly_head["score"]
                model_outputs["anomaly"] = anomaly_head

            if reg.bulkbuy:
                yield _evt({"status": "bulkbuy", "message": "Analyzing bulk-buy opportunities…"})
                await asyncio.sleep(0)
                res = reg.bulkbuy.predict(x)
                bulkbuy_result = {k: v[0] for k, v in res.items()}
                model_outputs["bulk_buy"] = {
                    "probability": float(res["bulk_prob"][0]),
                    "recommend":   bool(res["recommend"][0]),
                    "category":    int(res["target_category"][0]),
                    "est_savings": float(res["savings_estimate"][0]),
                }

            model_outputs["subscriptions"] = _build_subscription_inventory(df)

            yield _evt({"status": "suggestions", "message": "Generating personalized suggestions…"})
            await asyncio.sleep(0)
            engine = DecisionEngine(monthly_budget_per_category=req.budget or {})
            suggestions = engine.generate_suggestions(
                user_transactions=df,
                forecast_pred=forecast_pred,
                forecast_lower=forecast_lower,
                forecast_upper=forecast_upper,
                anomaly_result=anomaly_result,
                bulkbuy_result=bulkbuy_result,
                run_heuristic_rules=False,
            )

            # Intelligence detectors — mirror /analyze behaviour
            intel_suggestions, intel_fragment = _run_intelligence_detectors(
                df, model_outputs, req.budget
            )
            if intel_suggestions:
                suggestions = suggestions + intel_suggestions
                suggestions.sort(key=lambda s: s.confidence, reverse=True)
            if intel_fragment:
                model_outputs["intelligence"] = intel_fragment

            date_range = (
                f"{df['date'].min().date()} → {df['date'].max().date()}"
                if not df.empty else "N/A"
            )

            # Peer comparison
            stream_peer: dict | None = None
            if reg.user_vae is not None and reg.archetypes is not None and reg.cohort_stats is not None and reg.prep is not None:
                try:
                    from models.intelligence.peer_comparator import PeerComparator
                    x_vae     = _df_to_tensor(df, reg.prep)
                    embedding = reg.user_vae.get_user_embedding(x_vae)
                    date_range_days = max(1, (df["date"].max() - df["date"].min()).days + 1)
                    user_monthly = {
                        cat: float(df[df["category"] == cat]["amount"].sum() * 30 / date_range_days)
                        for cat in CATEGORIES
                    }
                    stream_peer = PeerComparator(reg.archetypes, reg.cohort_stats).compare(
                        embedding, user_monthly
                    )
                    model_outputs["peer_comparison"] = stream_peer
                except Exception:
                    pass

            analysis_context = _build_analysis_context(
                df=df,
                budget=req.budget,
                model_outputs=model_outputs,
            )

            # Persist suggestions + anomaly score (non-blocking)
            if req.user_id:
                try:
                    from api.db import save_suggestions, append_anomaly_score
                    _sug_rows = [s.to_dict() for s in suggestions]
                    await asyncio.to_thread(save_suggestions, req.user_id, _sug_rows)
                    if "anomaly" in model_outputs:
                        await asyncio.to_thread(
                            append_anomaly_score,
                            req.user_id,
                            model_outputs["anomaly"]["score"],
                        )
                except Exception:
                    pass

            # Emit complete immediately — page renders now, deals load separately
            yield _evt({
                "status": "complete",
                "data": {
                    "suggestions":      [s.to_dict() for s in suggestions],
                    "num_transactions": len(df),
                    "date_range":       date_range,
                    "model_outputs":    model_outputs,
                    "analysis_context": analysis_context,
                    "peer_comparison":  stream_peer,
                    "deals":            None,
                },
            })
            await asyncio.sleep(0)

            # Search for deals after the page is already visible
            try:
                from rules.deal_finder import DealFinder
                _sug_dicts = [s.to_dict() for s in suggestions]
                stream_deals = await asyncio.to_thread(
                    DealFinder().find, df, _sug_dicts, model_outputs
                ) or None
                if stream_deals:
                    yield _evt({"status": "deals_ready", "data": {"deals": stream_deals}})
            except Exception:
                pass
        except Exception as exc:
            yield _evt({"status": "error", "message": str(exc)})

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/embed")
def embed(req: AnalyzeRequest):
    """
    Encode a user's transactions into a 64-dim financial embedding.

    Returns the user's latent embedding, the nearest cohort spending statistics,
    and their assigned financial archetype label from k-means clustering.
    """
    reg = ModelRegistry
    if reg.user_vae is None or reg.prep is None:
        raise HTTPException(503, "User VAE not loaded — run train_user_vae.py first.")

    df = _transactions_to_df(req.transactions)
    x  = _df_to_tensor(df, reg.prep)

    embedding = reg.user_vae.get_user_embedding(x)   # (1, 64)
    emb_list  = embedding[0].cpu().tolist()

    # Financial archetype
    archetype = "Unknown"
    if reg.archetypes is not None:
        centers = reg.archetypes["centers"].to(embedding.device)
        names   = reg.archetypes["names"]
        import torch.nn.functional as F
        emb_n  = F.normalize(embedding, dim=-1)
        cen_n  = F.normalize(centers,   dim=-1)
        sims   = (emb_n @ cen_n.T).squeeze(0)    # (k,)
        archetype = names[int(sims.argmax())]

    # Nearest cohort stats
    cohort_stats: dict = {}
    if reg.cohort_embs is not None:
        indices, sims = reg.user_vae.nearest_cohort(
            embedding, reg.cohort_embs.to(embedding.device), top_k=5
        )
        # Nearest 5 users — approximate their spend by decoding their embeddings
        near_embs = reg.cohort_embs[indices[0]].to(embedding.device)   # (5, 64)
        with torch.no_grad():
            near_recon = reg.user_vae.decode(near_embs)   # (5, seq_len, num_categories)
        avg_monthly = near_recon.mean(dim=1).mean(dim=0).cpu().tolist()   # (num_categories,)
        cohort_stats = {
            cat: round(float(avg_monthly[i]) * 30, 2)   # approximate monthly
            for i, cat in enumerate(CATEGORIES)
        }

    return {
        "embedding":             emb_list,
        "financial_archetype":   archetype,
        "nearest_cohort_stats":  cohort_stats,
    }


@app.get("/status", response_class=HTMLResponse, include_in_schema=False)
def status_page():
    """Human-readable status page for self-hosters."""
    health = {
        "forecast": ModelRegistry.forecast is not None,
        "anomaly":  ModelRegistry.anomaly  is not None,
        "bulkbuy":  ModelRegistry.bulkbuy  is not None,
    }
    rows = "".join(
        f"<tr><td style='padding:6px 16px;border-bottom:1px solid #e2e8f0'>{k}</td>"
        f"<td style='padding:6px 16px;border-bottom:1px solid #e2e8f0'>"
        f"<span style='color:{'#16a34a' if v else '#dc2626'};font-weight:600'>{'Loaded' if v else 'Not loaded'}</span>"
        f"</td></tr>"
        for k, v in health.items()
    )
    return HTMLResponse(f"""<!DOCTYPE html><html><head><title>BudgetML Status</title>
<style>body{{font-family:system-ui,sans-serif;max-width:540px;margin:60px auto;padding:0 20px;color:#1e293b}}
h1{{font-size:1.4rem;font-weight:700;margin-bottom:4px}}
table{{width:100%;border-collapse:collapse;border:1px solid #e2e8f0;border-radius:10px;overflow:hidden}}
th{{background:#f8fafc;padding:8px 16px;text-align:left;font-size:.8rem;text-transform:uppercase;letter-spacing:.05em;color:#64748b}}</style>
</head><body>
<h1>BudgetML — System Status</h1>
<p style='color:#64748b;margin-bottom:20px'>Device: <strong>{DEVICE}</strong></p>
<table><thead><tr><th>Model</th><th>Status</th></tr></thead><tbody>{rows}</tbody></table>
<p style='margin-top:24px;font-size:.85rem;color:#94a3b8'>
  <a href='/docs' style='color:#6366f1'>API Docs (Swagger)</a> &nbsp;·&nbsp;
  <a href='/redoc' style='color:#6366f1'>ReDoc</a> &nbsp;·&nbsp;
  <a href='/health' style='color:#6366f1'>Health JSON</a>
</p></body></html>""")
