"""
rules/decision_engine.py — Post-model rule engine that converts model outputs
into human-readable, actionable budget suggestions.

Pipeline
--------
  Model outputs (forecast, anomaly, bulk-buy)
      ↓
  DecisionEngine.generate_suggestions()
      ↓
  List[Suggestion] — each has: type, message, confidence, category, amount_impact

Each model "classifies a target" and the decision engine applies rule-based
logic to produce a concrete deal or action for the user.
"""

from __future__ import annotations

import heapq
import re
from dataclasses import dataclass, field, asdict
import pandas as pd
import numpy as np

from rules.time_value_calculator import TimeValueCalculator

_tvc = TimeValueCalculator()


CATEGORIES = [
    "dining", "groceries", "subscriptions", "transportation",
    "utilities", "entertainment", "shopping", "healthcare", "other",
]

# Suggested bulk retailers per category
BULK_RETAILERS: dict[str, list[str]] = {
    "groceries":    ["Costco", "Sam's Club", "Walmart Grocery (bulk)", "Aldi"],
    "subscriptions": ["Amazon Prime Annual Plan", "YouTube Premium Family"],
    "shopping":     ["Costco", "Amazon Subscribe & Save"],
    "healthcare":   ["Costco Pharmacy", "GoodRx", "Amazon Pharmacy"],
    "utilities":    ["Groupon", "BillShark (negotiation service)"],
    "dining":       ["Costco food court", "restaurant.com gift cards"],
    "transportation": ["GasBuddy", "Costco Gas", "AAA membership"],
    "entertainment": ["Costco tickets", "AMC A-List", "bundle streaming plans"],
}

ANOMALY_CLASS_NAMES = {
    0: "normal",
    1: "anomaly",
}

# Actionable links per suggestion type / category (shown in the UI)
SUGGESTION_LINKS: dict[str, list[dict]] = {
    "forecast_warning": [
        {"label": "YNAB budgeting", "url": "https://www.youneedabudget.com/"},
        {"label": "Mint tracker",   "url": "https://mint.intuit.com/"},
    ],
    "bulk_buy_opportunity": [
        {"label": "Costco membership", "url": "https://www.costco.com/join-costco.html"},
        {"label": "Sam's Club",        "url": "https://www.samsclub.com/join"},
        {"label": "Amazon Subscribe & Save", "url": "https://www.amazon.com/subscribe-save/"},
    ],
    "subscription_trap": [
        {"label": "Cancel subscriptions (DoNotPay)", "url": "https://donotpay.com/learn/cancel-subscriptions/"},
        {"label": "Rocket Money tracker",            "url": "https://www.rocketmoney.com/"},
    ],
    "price_intelligence": [
        {"label": "GasBuddy (gas prices)",  "url": "https://www.gasbuddy.com/"},
        {"label": "GoodRx (rx prices)",     "url": "https://www.goodrx.com/"},
        {"label": "Google Shopping",        "url": "https://shopping.google.com/"},
    ],
    "behavioral_bias": [
        {"label": "r/personalfinance",  "url": "https://www.reddit.com/r/personalfinance/"},
        {"label": "NerdWallet tips",    "url": "https://www.nerdwallet.com/article/finance/stop-impulse-buying"},
    ],
    "cash_crunch_warning": [
        {"label": "Emergency fund guide", "url": "https://www.nerdwallet.com/article/banking/emergency-fund-why-it-matters"},
        {"label": "Chime SpotMe",         "url": "https://www.chime.com/spotme/"},
    ],
    "goal_progress": [
        {"label": "High-yield savings (NerdWallet)", "url": "https://www.nerdwallet.com/best/banking/high-yield-online-savings-accounts"},
        {"label": "I Bonds (TreasuryDirect)",        "url": "https://www.treasurydirect.gov/savings-bonds/i-bonds/"},
    ],
    "price_increase": [
        {"label": "Cancel/manage subscriptions (DoNotPay)", "url": "https://donotpay.com/learn/cancel-subscriptions/"},
        {"label": "Rocket Money (negotiate bills)",          "url": "https://www.rocketmoney.com/"},
    ],
}

# Direct cancel / manage-plan URLs for common subscription merchants.
# Key: lowercase merchant name fragment; value: direct account/cancel URL.
MERCHANT_CANCEL_URLS: dict[str, dict] = {
    "netflix":     {"label": "Manage Netflix plan",      "url": "https://www.netflix.com/account"},
    "spotify":     {"label": "Manage Spotify plan",      "url": "https://www.spotify.com/account/subscription/"},
    "hulu":        {"label": "Manage Hulu plan",         "url": "https://secure.hulu.com/account"},
    "disney":      {"label": "Manage Disney+ plan",      "url": "https://www.disneyplus.com/en-gb/identity/login"},
    "apple":       {"label": "Manage Apple subscriptions","url": "https://support.apple.com/en-us/118428"},
    "amazon":      {"label": "Manage Prime membership",  "url": "https://www.amazon.com/mc"},
    "youtube":     {"label": "Manage YouTube Premium",   "url": "https://www.youtube.com/paid_memberships"},
    "max":         {"label": "Manage Max plan",          "url": "https://www.max.com/settings/subscription"},
    "peacock":     {"label": "Manage Peacock plan",      "url": "https://www.peacocktv.com/account"},
    "paramount":   {"label": "Manage Paramount+ plan",   "url": "https://www.paramountplus.com/account/"},
    "xfinity":     {"label": "Negotiate Xfinity bill",   "url": "https://www.xfinity.com/support/articles/lower-your-bill"},
    "comcast":     {"label": "Negotiate Comcast bill",   "url": "https://www.xfinity.com/support/articles/lower-your-bill"},
    "verizon":     {"label": "Review Verizon plan",      "url": "https://www.verizon.com/solutions-and-services/change-plan/"},
    "att":         {"label": "Review AT&T plan",         "url": "https://www.att.com/buy/broadband/plans.html"},
    "t-mobile":    {"label": "Review T-Mobile plan",     "url": "https://www.t-mobile.com/plans"},
    "gym":         {"label": "Cancel gym (DoNotPay)",    "url": "https://donotpay.com/learn/how-to-cancel-gym-membership/"},
    "planet":      {"label": "Manage Planet Fitness",    "url": "https://www.planetfitness.com/gym-memberships"},
    "audible":     {"label": "Manage Audible",           "url": "https://www.audible.com/account/manage-membership"},
    "adobe":       {"label": "Manage Adobe plan",        "url": "https://account.adobe.com/plans"},
    "microsoft":   {"label": "Manage Microsoft 365",     "url": "https://account.microsoft.com/services/"},
    "dropbox":     {"label": "Manage Dropbox plan",      "url": "https://www.dropbox.com/account/plan"},
    "duolingo":    {"label": "Manage Duolingo Plus",     "url": "https://www.duolingo.com/settings/subscription"},
    "nytimes":     {"label": "Manage NYT subscription",  "url": "https://myaccount.nytimes.com/seg/subscription"},
    "new york":    {"label": "Manage NYT subscription",  "url": "https://myaccount.nytimes.com/seg/subscription"},
    "wsj":         {"label": "Manage WSJ subscription",  "url": "https://store.wsj.com/shop/wsjcom/myaccount"},
    "linkedin":    {"label": "Manage LinkedIn Premium",  "url": "https://www.linkedin.com/premium/products/"},
    "chatgpt":     {"label": "Manage ChatGPT Plus",      "url": "https://chat.openai.com/#settings/subscription"},
    "openai":      {"label": "Manage OpenAI plan",       "url": "https://platform.openai.com/account/billing"},
}


def _merchant_cancel_link(merchant: str) -> dict | None:
    """Return a {label, url} dict for the most specific matching cancel URL, or None."""
    m = merchant.lower()
    for fragment, link in MERCHANT_CANCEL_URLS.items():
        if fragment in m:
            return link
    return None


_BANK_NOISE_RE = re.compile(
    r"\b(?:VISA|MASTERCARD|DEBIT|CREDIT|DDA|PUR|POS|ACH|AUTH|TRNSFR|XFER|"
    r"PENDING|POSTED|RECURRING|PAYMENT|TRANSFER|PURCHASE|WITHDRAWAL|CHK|CHECK|"
    r"INTL\s+T|TXN\s+FEE|FOREIGN\s+FEE|SERVICE\s+FEE|NSF|OVERDRAFT)\b",
    re.IGNORECASE,
)

_MERCHANT_MIN_TXNS: int = 3   # ignore merchants with fewer transactions
_MERCHANT_MAX_NAME_LEN: int = 50


def _is_noisy_merchant(name: str) -> bool:
    """Return True if the merchant name looks like raw bank statement noise."""
    if not name or name in ("<UNK>", "Unknown", ""):
        return True
    if len(name) > _MERCHANT_MAX_NAME_LEN:
        return True
    digit_ratio = sum(c.isdigit() for c in name) / max(len(name), 1)
    if digit_ratio > 0.30:
        return True
    return bool(_BANK_NOISE_RE.search(name))


def rank_anomalous_transactions(
    user_transactions: pd.DataFrame,
    lookback_days: int = 30,
    top_n: int = 5,
) -> list[dict]:
    """
    Rank likely anomalous transactions in the recent window using a composite score.

    Composite score components:
    - category amount z-score
    - merchant novelty
    - amount-to-category-average ratio
    - recency boost
    """
    if user_transactions.empty:
        return []

    df = user_transactions.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "amount", "category", "merchant"])
    if df.empty:
        return []

    max_date = df["date"].max()
    cutoff = max_date - pd.Timedelta(days=max(1, lookback_days))
    recent = df[df["date"] >= cutoff].copy()
    if recent.empty:
        return []

    cat_stats = (
        df.groupby("category")["amount"]
        .agg(cat_mean="mean", cat_std="std", cat_count="count")
        .fillna(0.0)
    )
    merch_counts = df.groupby("merchant")["amount"].count().to_dict()

    candidates: list[dict] = []
    for _, row in recent.iterrows():
        cat = str(row["category"])
        merchant = str(row["merchant"])
        amount = float(row["amount"])
        date_val = pd.to_datetime(row["date"])

        cstats = cat_stats.loc[cat] if cat in cat_stats.index else None
        cat_mean = float(cstats["cat_mean"]) if cstats is not None else 0.0
        cat_std = float(cstats["cat_std"]) if cstats is not None else 0.0
        cat_count = int(cstats["cat_count"]) if cstats is not None else 0

        z_score = 0.0
        if cat_count >= 3 and cat_std > 1e-6:
            z_score = max(0.0, (amount - cat_mean) / cat_std)
        z_norm = min(1.0, z_score / 4.0)

        merchant_count = int(merch_counts.get(merchant, 0))
        novelty = 1.0 if merchant_count <= 1 else min(1.0, 1.0 / np.sqrt(merchant_count))

        # Use ratio only when the category average is meaningful (≥ $1).
        # Below that floor the ratio inflates artificially and adds no signal.
        ratio = (amount / cat_mean) if cat_mean >= 1.0 else 1.0
        ratio_norm = min(1.0, max(0.0, (ratio - 1.0) / 4.0))

        days_ago = max(0.0, (max_date - date_val).days)
        recency = max(0.0, 1.0 - (days_ago / max(float(lookback_days), 1.0)))

        composite = (
            0.40 * z_norm +
            0.25 * novelty +
            0.25 * ratio_norm +
            0.10 * recency
        )

        reasons: list[str] = []
        if merchant_count <= 1:
            reasons.append("new_merchant")
        if z_score >= 2.0:
            reasons.append("high_amount_zscore")
        if ratio >= 1.8:
            reasons.append(f"{ratio:.1f}x_category_avg")
        if recency >= 0.8:
            reasons.append("recent_outlier")
        if not reasons:
            reasons.append("account_level_pattern")

        candidates.append({
            "date": str(date_val.date()),
            "merchant": merchant,
            "amount": round(amount, 2),
            "category": cat,
            "score": round(float(composite), 4),
            "confidence": round(float(min(1.0, composite)), 4),
            "reasons": reasons,
            "evidence": {
                "amount_zscore": round(float(z_score), 3),
                "amount_vs_category_avg": round(float(ratio), 3),
                "merchant_visit_count": merchant_count,
                "days_ago": int(days_ago),
                "category_avg_amount": round(float(cat_mean), 2),
            },
        })

    return heapq.nlargest(max(1, top_n), candidates, key=lambda c: c["score"])


# ─────────────────────────────────────────────────────────────────────────────
# Suggestion dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Suggestion:
    """
    A single actionable budget suggestion surfaced to the user.

    Fields
    ------
    type          : str   "forecast_warning" | "anomaly_alert" | "bulk_buy" | "budget_tip"
    category      : str   spending category this suggestion targets
    message       : str   human-readable suggestion text
    confidence    : float model confidence / anomaly score (0–1)
    amount_impact : float estimated $/month impact (positive = savings)
    details       : dict  extra context (merchant, date, model name, etc.)
    """
    type:          str
    category:      str
    message:       str
    confidence:    float
    amount_impact: float
    details:       dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Decision Engine
# ─────────────────────────────────────────────────────────────────────────────

class DecisionEngine:
    """
    Converts raw model outputs into user-facing suggestions via decision rules.

    Usage
    -----
    engine = DecisionEngine(monthly_budget_per_category={...})
    suggestions = engine.generate_suggestions(
        forecast_pred=...,
        anomaly_result=...,
        bulkbuy_result=...,
        user_transactions=df,
    )
    """

    def __init__(
        self,
        monthly_budget_per_category: dict[str, float] | None = None,
        forecast_warning_pct: float = 0.10,   # warn if predicted > budget * (1 + this)
        min_bulk_buy_confidence: float = 0.45,
        min_anomaly_score: float = 0.50,
    ):
        """
        Parameters
        ----------
        monthly_budget_per_category : dict[category → $ budget]
            If None, budgets are inferred from user's historical median.
        forecast_warning_pct : float
            Fraction above budget to trigger a forecast warning (default 10%).
        min_bulk_buy_confidence : float
            Minimum bulk-buy model probability to generate a recommendation.
        min_anomaly_score : float
            Minimum anomaly score (0–1) to surface an alert.
        """
        self.budgets             = monthly_budget_per_category or {}
        self.forecast_warning_pct = forecast_warning_pct
        self.min_bulk_buy_conf   = min_bulk_buy_confidence
        self.min_anomaly_score   = min_anomaly_score

    # ── Budget inference ──────────────────────────────────────────────────────

    def _infer_budgets(self, df: pd.DataFrame) -> dict[str, float]:
        """Compute per-category monthly median spend as a budget estimate."""
        df = df.copy()
        df["year_month"] = df["date"].dt.to_period("M")
        monthly = (
            df.groupby(["year_month", "category"])["amount"]
            .sum()
            .unstack(fill_value=0.0)
        )
        medians = monthly.median(axis=0)
        return {cat: float(medians.get(cat, 0.0)) for cat in CATEGORIES}

    # ── Rule: Forecast warnings ────────────────────────────────────────────────

    def _forecast_rules(
        self,
        forecast_pred:  np.ndarray,                 # (NUM_CATEGORIES,) point estimate
        user_transactions: pd.DataFrame,
        forecast_lower: np.ndarray | None = None,   # (NUM_CATEGORIES,) 10th percentile
        forecast_upper: np.ndarray | None = None,   # (NUM_CATEGORIES,) 90th percentile
    ) -> list[Suggestion]:
        suggestions: list[Suggestion] = []

        budgets = self.budgets if self.budgets else self._infer_budgets(user_transactions)

        # Per-category actual spend + top merchant for data-driven tips
        recent_days = max(1, (user_transactions["date"].max() - user_transactions["date"].min()).days + 1)
        monthly_rate = 30.0 / recent_days
        actual_monthly: dict[str, float] = {}
        top_merchants:  dict[str, tuple[str, int, float]] = {}  # cat → (merchant, count, spend)
        for cat_g, gdf in user_transactions.groupby("category"):
            actual_monthly[str(cat_g)] = float(gdf["amount"].sum()) * monthly_rate
            mc = gdf.groupby("merchant")["amount"].agg(["sum", "count"])
            if not mc.empty:
                tm = mc["sum"].idxmax()
                top_merchants[str(cat_g)] = (
                    str(tm),
                    int(mc.loc[tm, "count"]),
                    float(mc.loc[tm, "sum"]),
                )

        calibration_caveat = ""

        for i, cat in enumerate(CATEGORIES):
            pred_spend = float(forecast_pred[i])
            budget     = budgets.get(cat, 0.0)
            if budget <= 0:
                continue

            overspend_pct = (pred_spend - budget) / budget

            if overspend_pct > self.forecast_warning_pct:
                overspend_amt = pred_spend - budget
                confidence    = min(1.0, overspend_pct)

                # Data-driven tip with actual merchant context
                tm_info = top_merchants.get(cat)
                tip = _category_saving_tip(
                    cat,
                    overspend_amt,
                    top_merchant=tm_info[0] if tm_info else None,
                    txn_count=tm_info[1] if tm_info else None,
                    actual_spend=tm_info[2] if tm_info else None,
                )

                # Build interval clause when bounds are available
                interval_clause = ""
                lower_over_budget = False
                if forecast_lower is not None and forecast_upper is not None:
                    lo = float(forecast_lower[i])
                    hi = float(forecast_upper[i])
                    interval_clause = f" (80% CI: ${lo:.0f}–${hi:.0f})"
                    # Most alarming: even the optimistic lower bound exceeds budget
                    if lo > budget:
                        lower_over_budget = True

                if lower_over_budget:
                    lo = float(forecast_lower[i])
                    lo_over = lo - budget
                    message = (
                        f"Your {cat} spending is forecast to reach ${pred_spend:.0f} "
                        f"next month{interval_clause} — ${overspend_amt:.0f} over your "
                        f"${budget:.0f} budget even in the optimistic case "
                        f"(lower bound ${lo:.0f} is ${lo_over:.0f} over budget). {tip}"
                    )
                else:
                    message = (
                        f"Your {cat} spending is forecast to reach ${pred_spend:.0f} "
                        f"next month{interval_clause} — ${overspend_amt:.0f} over your "
                        f"${budget:.0f} budget. {tip}"
                    )

                opp = _tvc.opportunity_cost_dict(overspend_amt, cat)
                # Opportunity cost is for the overspend only, not total category spend
                opp_msg = _tvc.format_opportunity_cost(
                    overspend_amt, cat,
                    label=f"the ${overspend_amt:.0f} monthly overspend",
                )
                suggestions.append(Suggestion(
                    type="forecast_warning",
                    category=cat,
                    message=message + f" {opp_msg}.{calibration_caveat}",
                    confidence=float(confidence),
                    amount_impact=-overspend_amt,
                    details={
                        "predicted_spend":    pred_spend,
                        "budget":             budget,
                        "overspend_pct":      round(overspend_pct * 100, 1),
                        "lower_bound":        float(forecast_lower[i]) if forecast_lower is not None else None,
                        "upper_bound":        float(forecast_upper[i]) if forecast_upper is not None else None,
                        "lower_over_budget":  lower_over_budget,
                        "links":              SUGGESTION_LINKS.get("forecast_warning", []),
                        **opp,
                    },
                ))

        return suggestions

    # ── Rule: Anomaly alerts ──────────────────────────────────────────────────

    def _anomaly_rules(
        self,
        anomaly_score:  float,
        anomaly_class:  int,                 # 0=normal, 1=anomaly
        anomaly_probs:  np.ndarray,          # (NUM_CLASSES,) class probabilities
        user_transactions: pd.DataFrame,
    ) -> list[Suggestion]:
        suggestions: list[Suggestion] = []

        # Normalise score to 0–1 if needed
        score_01 = min(1.0, float(anomaly_score))

        if score_01 < self.min_anomaly_score:
            return suggestions

        class_name = ANOMALY_CLASS_NAMES.get(anomaly_class, "unknown")

        candidates = rank_anomalous_transactions(user_transactions, lookback_days=30, top_n=5)
        primary = candidates[0] if candidates else None
        candidate_conf = float(primary.get("confidence", 0.0)) if primary else 0.0

        merchant = str(primary.get("merchant", "Unknown")) if primary else "Unknown"
        cat = str(primary.get("category", "other")) if primary else "other"
        amount = float(primary.get("amount", 0.0)) if primary else 0.0
        date_str = str(primary.get("date", "")) if primary else ""
        evidence = primary.get("evidence", {}) if primary else {}

        avg_cat_amount = float(evidence.get("category_avg_amount", 0.0))
        ratio = float(evidence.get("amount_vs_category_avg", 1.0))
        merchant_count = int(evidence.get("merchant_visit_count", 0))
        is_new_merchant = merchant_count <= 1

        if anomaly_class == 1 and primary is not None and candidate_conf >= 0.45:
            peer_ctx = (
                f" This is {ratio:.1f}× your average {cat} transaction (avg ${avg_cat_amount:.0f})."
                if avg_cat_amount > 0 else ""
            )
            merch_ctx = (
                " First time seeing this merchant."
                if is_new_merchant
                else f" You've transacted at {merchant} {merchant_count} time(s) before."
            )
            message = (
                f"Unusual transaction: ${amount:.0f} at {merchant} on {date_str}."
                f"{peer_ctx}{merch_ctx} "
                "Verify this was intentional or flag as a billing error."
            )
        elif anomaly_class == 1:
            message = (
                f"Anomaly detected in your recent transactions (score: {score_01:.2f}). "
                "This appears to be an account-level pattern anomaly rather than one clearly "
                "attributable transaction."
            )
        else:
            message = (
                f"Anomaly detected in your recent transactions (score: {score_01:.2f}). "
                "Please review your account activity."
            )

        suggestions.append(Suggestion(
            type="anomaly_alert",
            category=cat,
            message=message,
            confidence=score_01,
            amount_impact=0.0,
            details={
                "anomaly_class":            class_name,
                "anomaly_score":            score_01,
                "class_probs":              anomaly_probs.tolist(),
                "top_merchant":             merchant,
                "top_amount":               amount,
                "top_date":                 date_str,
                "avg_category_amount":      round(avg_cat_amount, 2),
                "amount_vs_average_ratio":  round(ratio, 2),
                "is_new_merchant":          is_new_merchant,
                "merchant_visit_count":     merchant_count,
                "anomaly_candidates":       candidates,
                "primary_anomaly":          primary,
                "evidence_window_days":     30,
                "anomaly_scope":            "transaction" if primary is not None and candidate_conf >= 0.45 else "account-level pattern anomaly",
            },
        ))
        return suggestions

    # ── Rule: Heuristic anomaly fallback (no trained model required) ──────────

    def _heuristic_anomaly_rules(
        self, user_transactions: pd.DataFrame
    ) -> list[Suggestion]:
        """
        Flag obviously suspicious transactions using rank_anomalous_transactions
        when the ML anomaly model has not produced a result.

        Fires only when composite score >= 0.65 (requires at least two strong
        signals: new merchant, large amount ratio, or high z-score).
        """
        _HEURISTIC_ANOMALY_THRESHOLD = 0.65
        suggestions: list[Suggestion] = []
        if user_transactions.empty:
            return suggestions

        candidates = rank_anomalous_transactions(user_transactions, lookback_days=60, top_n=3)
        for c in candidates:
            if c["confidence"] < _HEURISTIC_ANOMALY_THRESHOLD:
                break  # nlargest — once below threshold, rest are too
            evidence  = c.get("evidence", {})
            merchant  = c["merchant"]
            amount    = c["amount"]
            date_str  = c["date"]
            cat       = c["category"]
            ratio     = float(evidence.get("amount_vs_category_avg", 1.0))
            avg_amt   = float(evidence.get("category_avg_amount", 0.0))
            mc        = int(evidence.get("merchant_visit_count", 0))

            peer_ctx = (
                f" This is {ratio:.1f}× your average {cat} transaction (avg ${avg_amt:.0f})."
                if avg_amt > 0 else ""
            )
            merch_ctx = (
                " First time seeing this merchant."
                if mc <= 1
                else f" You've transacted at {merchant} {mc} time(s) before."
            )
            suggestions.append(Suggestion(
                type="anomaly_alert",
                category=cat,
                message=(
                    f"Unusual transaction: ${amount:.0f} at {merchant} on {date_str}."
                    f"{peer_ctx}{merch_ctx} "
                    "Verify this was intentional or flag as a billing error."
                ),
                confidence=c["confidence"],
                amount_impact=0.0,
                details={
                    "anomaly_class":           "heuristic",
                    "anomaly_score":           c["confidence"],
                    "top_merchant":            merchant,
                    "top_amount":              amount,
                    "top_date":                date_str,
                    "avg_category_amount":     avg_amt,
                    "amount_vs_average_ratio": ratio,
                    "is_new_merchant":         mc <= 1,
                    "merchant_visit_count":    mc,
                    "reasons":                 c.get("reasons", []),
                    "anomaly_scope":           "transaction",
                },
            ))
        return suggestions

    # ── Rule: Bulk-buy recommendations ────────────────────────────────────────

    def _bulk_buy_rules(
        self,
        bulk_prob:        float,
        target_category:  int,              # category index
        category_probs:   np.ndarray,       # (NUM_CATEGORIES,)
        savings_estimate: float,
        user_transactions: pd.DataFrame,
    ) -> list[Suggestion]:
        suggestions: list[Suggestion] = []

        if float(bulk_prob) < self.min_bulk_buy_conf:
            return suggestions

        cat_idx = int(target_category)
        cat = CATEGORIES[cat_idx] if 0 <= cat_idx < len(CATEGORIES) else "other"

        # Find the most frequently purchased merchant in this category
        cat_txns = user_transactions[user_transactions["category"] == cat]
        if cat_txns.empty:
            return suggestions

        merchant_counts = cat_txns["merchant"].value_counts()
        if merchant_counts.empty:
            return suggestions
        top_merchant    = merchant_counts.index[0]
        purchase_freq   = float(merchant_counts.iloc[0])
        avg_spend       = float(cat_txns["amount"].mean())

        # Pick bulk retailer suggestions for this category
        retailers = BULK_RETAILERS.get(cat, ["Costco", "Sam's Club"])
        retailer_str = " or ".join(retailers[:2])

        est_savings = max(float(savings_estimate), avg_spend * 0.15)  # at least 15% estimate

        message = (
            f"You purchase {cat} at {top_merchant} about {purchase_freq:.0f} times over the last "
            f"60 days (avg ${avg_spend:.0f}/transaction). Switching to bulk buying at "
            f"{retailer_str} could save you ~${est_savings:.0f}/month."
        )

        opp = _tvc.opportunity_cost_dict(est_savings, cat, label=f"${est_savings:.0f}/month in bulk savings")
        suggestions.append(Suggestion(
            type="bulk_buy_opportunity",
            category=cat,
            message=message + f" {opp['opportunity_cost_message']}.",
            confidence=float(bulk_prob),
            amount_impact=est_savings,
            details={
                "top_merchant":        top_merchant,
                "purchase_frequency":  purchase_freq,
                "avg_transaction":     avg_spend,
                "suggested_retailers": retailers[:2],
                "category_probs":      category_probs.tolist(),
                "links":               SUGGESTION_LINKS.get("bulk_buy_opportunity", []),
                **opp,
            },
        ))
        return suggestions

    # ── Rule: Heuristic bulk-buy (frequency-based, no ML required) ────────────

    def _heuristic_bulk_buy_rules(
        self,
        user_transactions: pd.DataFrame,
        existing_bulk_cats: set[str],
    ) -> list[Suggestion]:
        """
        Fire bulk-buy suggestions purely from transaction frequency.
        Catches cases where the ML model is underconfident or untrained.
        """
        suggestions: list[Suggestion] = []
        if user_transactions.empty:
            return suggestions

        bulk_eligible = {"groceries", "dining", "shopping", "transportation", "entertainment"}
        dates = pd.to_datetime(user_transactions["date"])
        date_range_days = max(1, (dates.max() - dates.min()).days + 1)

        agg = (
            user_transactions.groupby("category")["amount"]
            .agg(["count", "sum", "mean"])
            .reset_index()
        )

        # ≥2 purchases AND avg ≥ $12 AND not already covered by ML tip
        eligible = agg[
            agg["category"].isin(bulk_eligible)
            & ~agg["category"].isin(existing_bulk_cats)
            & (agg["count"] >= 2)
            & (agg["mean"] >= 12.0)
        ].sort_values("sum", ascending=False)

        for _, row in eligible.head(2).iterrows():
            cat       = str(row["category"])
            count     = int(row["count"])
            avg       = float(row["mean"])
            monthly   = float(row["sum"]) * 30 / date_range_days
            est_sav   = round(monthly * 0.15, 2)

            cat_txns    = user_transactions[user_transactions["category"] == cat]
            top_merchant = (
                cat_txns["merchant"].value_counts().index[0]
                if not cat_txns.empty else "your usual store"
            )
            retailers   = BULK_RETAILERS.get(cat, ["Costco", "Sam's Club"])
            retailer_str = " or ".join(retailers[:2])
            opp = _tvc.opportunity_cost_dict(est_sav, cat, label=f"${est_sav:.0f}/month in bulk savings")

            suggestions.append(Suggestion(
                type="bulk_buy_opportunity",
                category=cat,
                message=(
                    f"You made {count} {cat} purchases this period "
                    f"(avg ${avg:.0f} each, ~${monthly:.0f}/month at {top_merchant}). "
                    f"Buying in bulk at {retailer_str} could save ~${est_sav:.0f}/month. "
                    f"{opp['opportunity_cost_message']}."
                ),
                confidence=0.55,
                amount_impact=est_sav,
                details={
                    "top_merchant":        top_merchant,
                    "purchase_frequency":  count,
                    "avg_transaction":     avg,
                    "suggested_retailers": retailers[:2],
                    "links":               SUGGESTION_LINKS.get("bulk_buy_opportunity", []),
                    **opp,
                },
            ))
        return suggestions

    # ── Rule: Subscription analysis ───────────────────────────────────────────

    def _subscription_rules(self, user_transactions: pd.DataFrame) -> list[Suggestion]:
        """Detect subscription traps and add cancellation/negotiation suggestions."""
        from rules.subscription_analyzer import SubscriptionAnalyzer
        suggestions: list[Suggestion] = []
        try:
            subs = SubscriptionAnalyzer().analyze(user_transactions)
        except Exception:
            return suggestions

        if subs:
            total_monthly = float(sum(float(s.get("monthly_cost", 0.0)) for s in subs))
            total_annual = total_monthly * 12.0
            trap_count = int(sum(1 for s in subs if s.get("status") == "trap" or s.get("is_trap")))
            suggestions.append(Suggestion(
                type="subscription_summary",
                category="subscriptions",
                message=(
                    f"We found {len(subs)} active subscriptions totaling ${total_monthly:.0f}/mo "
                    f"(${total_annual:.0f}/yr)."
                ),
                confidence=0.8,
                amount_impact=total_monthly,
                details={
                    "subscription_count": len(subs),
                    "total_monthly": round(total_monthly, 2),
                    "total_annual": round(total_annual, 2),
                    "trap_count": trap_count,
                    "items": subs,
                    "cta_tab": "subscriptions",
                },
            ))

        for sub in subs:
            if not sub.get("is_trap"):
                continue
            monthly = sub["monthly_cost"]
            opp = _tvc.opportunity_cost_dict(monthly, "subscriptions", label=f"${monthly:.0f}/month by canceling {sub['merchant']}")
            suggestions.append(Suggestion(
                type="subscription_trap",
                category="subscriptions",
                message=(
                    f"{sub['merchant']} costs ${monthly:.0f}/month. "
                    f"Take a moment to review how often you actually use it — "
                    f"you've been subscribed for {sub['tenure_months']} months. "
                    f"Negotiation tip: {sub['negotiation_script'][:120]}... "
                    f"{opp['opportunity_cost_message']}."
                ),
                confidence=min(1.0, 1.0 - (sub["usage_score"] or 0.0)),
                amount_impact=monthly,
                details={
                    **sub, **opp,
                    "links": SUGGESTION_LINKS.get("subscription_trap", []),
                },
            ))
        return suggestions

    # ── Rule: Subscription price-increase detector ────────────────────────────

    def _subscription_price_increase_rules(
        self, user_transactions: pd.DataFrame
    ) -> list[Suggestion]:
        """
        Detect when a recurring subscription silently raised its price.

        Compares average charge per subscription merchant in the most recent
        30-day window vs the prior 30-day window (days 31-60 before max date).
        Fires only when increase ≥ 5% AND ≥ $1.00 — avoids noise from
        timing differences in billing cycles.
        """
        suggestions: list[Suggestion] = []
        if user_transactions.empty:
            return suggestions

        df = user_transactions.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "amount", "merchant", "category"])

        subs_df = df[df["category"] == "subscriptions"]
        if len(subs_df) < 4:
            return suggestions

        max_date = subs_df["date"].max()
        cur_start  = max_date - pd.Timedelta(days=30)
        prev_start = max_date - pd.Timedelta(days=60)

        cur_window  = subs_df[subs_df["date"] >= cur_start]
        prev_window = subs_df[(subs_df["date"] > prev_start) & (subs_df["date"] <= cur_start)]

        if cur_window.empty or prev_window.empty:
            return suggestions

        cur_avg  = cur_window.groupby("merchant")["amount"].mean()
        prev_avg = prev_window.groupby("merchant")["amount"].mean()

        for merchant in cur_avg.index:
            if merchant not in prev_avg.index:
                continue
            cur   = float(cur_avg[merchant])
            prev  = float(prev_avg[merchant])
            if prev < 1.0:
                continue
            increase_pct = (cur - prev) / prev
            increase_amt = cur - prev
            if increase_pct < 0.05 or increase_amt < 1.0:
                continue

            cancel_link = _merchant_cancel_link(merchant)
            links = (
                [cancel_link] + SUGGESTION_LINKS.get("price_increase", [])
                if cancel_link
                else SUGGESTION_LINKS.get("price_increase", [])
            )

            opp = _tvc.opportunity_cost_dict(
                increase_amt, "subscriptions",
                label=f"${increase_amt:.2f}/month increase at {merchant}",
            )
            suggestions.append(Suggestion(
                type="subscription_trap",
                category="subscriptions",
                message=(
                    f"{merchant} quietly raised its price from ${prev:.2f} to ${cur:.2f}/month "
                    f"(+{increase_pct * 100:.0f}%). That's an extra ${increase_amt * 12:.0f}/year. "
                    f"Call or chat to request a loyalty rate — providers typically offer 20–40% off "
                    f"to customers who ask before canceling. {opp['opportunity_cost_message']}."
                ),
                confidence=min(1.0, 0.75 + increase_pct),
                amount_impact=increase_amt,
                details={
                    "merchant":          merchant,
                    "prev_monthly":      round(prev, 2),
                    "cur_monthly":       round(cur, 2),
                    "increase_pct":      round(increase_pct * 100, 1),
                    "increase_monthly":  round(increase_amt, 2),
                    "increase_annual":   round(increase_amt * 12, 2),
                    "links":             links,
                    **opp,
                },
            ))

        return suggestions

    # ── Rule: Price intelligence ──────────────────────────────────────────────

    def _price_intelligence_rules(
        self, user_transactions: pd.DataFrame
    ) -> list[Suggestion]:
        """
        Detect when the user consistently pays above-market prices at a merchant
        by comparing their per-merchant average to their own cross-merchant
        average for the same category. No external data required.

        Guards:
        - Skips merchants with < _MERCHANT_MIN_TXNS transactions (too noisy).
        - Skips merchant names that look like raw bank statement noise.
        - Skips comparisons where the cheapest merchant is free/near-zero (fees).
        """
        suggestions: list[Suggestion] = []
        if user_transactions.empty:
            return suggestions

        df = user_transactions.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Only compare merchants in categories where they sell the same commodity.
        # Dining/shopping merchants differ by product, not price — comparisons mislead.
        _FUNGIBLE_CATEGORIES = {"transportation", "groceries"}

        for cat, cat_df in df.groupby("category"):
            if cat not in _FUNGIBLE_CATEGORIES:
                continue
            if len(cat_df) < 10:
                continue

            # Count transactions per merchant; drop noisy or sparse merchants
            merch_counts = cat_df.groupby("merchant")["amount"].count()
            clean_merchs = [
                m for m, cnt in merch_counts.items()
                if cnt >= _MERCHANT_MIN_TXNS and not _is_noisy_merchant(str(m))
            ]
            if len(clean_merchs) < 2:
                continue

            cat_clean = cat_df[cat_df["merchant"].isin(clean_merchs)]
            merch_avg  = cat_clean.groupby("merchant")["amount"].mean()

            overall_cat_avg = float(merch_avg.mean())
            if overall_cat_avg <= 0:
                continue

            cheapest_merch = merch_avg.idxmin()
            cheapest_avg   = float(merch_avg.min())
            # Skip if cheapest is essentially a fee/zero-dollar entry
            if cheapest_avg < 2.0:
                continue

            links = SUGGESTION_LINKS.get("price_intelligence", [])

            for merchant, avg_amt in merch_avg.items():
                if merchant == cheapest_merch:
                    continue
                ratio = avg_amt / max(cheapest_avg, 1.0)
                if ratio >= 1.5 and avg_amt - cheapest_avg >= 5.0:
                    date_range_months = max(1, (cat_clean["date"].max() - cat_clean["date"].min()).days / 30)
                    monthly_visits = max(1, int(len(cat_clean[cat_clean["merchant"] == merchant]) / date_range_months))
                    monthly_extra  = (avg_amt - cheapest_avg) * monthly_visits
                    opp = _tvc.opportunity_cost_dict(monthly_extra, str(cat), label=f"${monthly_extra:.0f}/month by switching to {cheapest_merch}")
                    suggestions.append(Suggestion(
                        type="price_intelligence",
                        category=str(cat),
                        message=(
                            f"You pay ${avg_amt:.0f} avg at {merchant} for {cat}, "
                            f"but ${cheapest_avg:.0f} avg at {cheapest_merch} — "
                            f"a {ratio:.1f}× difference. Switching could save "
                            f"~${monthly_extra:.0f}/month. {opp['opportunity_cost_message']}."
                        ),
                        confidence=min(1.0, (ratio - 1.5) / 1.0 + 0.4),
                        amount_impact=monthly_extra,
                        details={
                            "expensive_merchant": merchant,
                            "cheap_merchant":     cheapest_merch,
                            "expensive_avg":      round(avg_amt, 2),
                            "cheap_avg":          round(cheapest_avg, 2),
                            "ratio":              round(ratio, 2),
                            "est_monthly_savings": round(monthly_extra, 2),
                            "links":              links,
                            **opp,
                        },
                    ))

        return suggestions

    # ── Rule: Intelligence models ─────────────────────────────────────────────

    def _life_event_rules(self, life_events: list[dict]) -> list[Suggestion]:
        suggestions: list[Suggestion] = []
        for evt in life_events:
            suggestions.append(Suggestion(
                type="life_event_detected",
                category="other",
                message=(
                    f"Life event detected: {evt['event'].replace('_', ' ').title()} "
                    f"around {evt['date']} (confidence {evt['confidence']:.0%}). "
                    "Your budget targets may need updating to reflect this change."
                ),
                confidence=float(evt["confidence"]),
                amount_impact=0.0,
                details=evt,
            ))
        return suggestions

    def _behavioral_bias_rules(self, biases: list[dict]) -> list[Suggestion]:
        suggestions: list[Suggestion] = []
        for bias in biases:
            suggestions.append(Suggestion(
                type="behavioral_bias",
                category="other",
                message=f"{bias['bias_name']}: {bias['evidence']}",
                confidence=float(bias["confidence"]),
                amount_impact=float(bias.get("impact_estimate", 0.0)),
                details={**bias, "links": SUGGESTION_LINKS.get("behavioral_bias", [])},
            ))
        return suggestions

    def _cash_crunch_rules(self, cash_crunch: dict) -> list[Suggestion]:
        suggestions: list[Suggestion] = []
        danger = cash_crunch.get("danger_dates", [])
        if not danger:
            return suggestions
        suggestions.append(Suggestion(
            type="cash_crunch_warning",
            category="other",
            message=cash_crunch.get("recommendation", "Cash flow risk detected."),
            confidence=0.85,
            amount_impact=0.0,
            details={
                "danger_dates":     danger,
                "largest_upcoming": cash_crunch.get("largest_upcoming"),
                "links":            SUGGESTION_LINKS.get("cash_crunch_warning", []),
            },
        ))
        return suggestions

    def _goal_rules(self, inferred_goal: dict | None) -> list[Suggestion]:
        if not inferred_goal:
            return []
        return [Suggestion(
            type="goal_progress",
            category="other",
            message=inferred_goal.get("message", ""),
            confidence=float(inferred_goal.get("confidence", 0.5)),
            amount_impact=float(inferred_goal.get("monthly_spend_drop", 0.0)),
            details={**inferred_goal, "links": SUGGESTION_LINKS.get("goal_progress", [])},
        )]

    # ── Main entry point ──────────────────────────────────────────────────────

    def generate_suggestions(
        self,
        user_transactions: pd.DataFrame,
        forecast_pred:    np.ndarray | None = None,
        forecast_lower:   np.ndarray | None = None,
        forecast_upper:   np.ndarray | None = None,
        anomaly_result:   dict | None       = None,
        bulkbuy_result:   dict | None       = None,
        life_events:      list | None       = None,
        behavioral_biases: list | None      = None,
        cash_crunch:      dict | None       = None,
        inferred_goal:    dict | None       = None,
        run_heuristic_rules: bool           = True,
    ) -> list[Suggestion]:
        """
        Generate all suggestions from model outputs.

        Parameters
        ----------
        user_transactions  : pd.DataFrame — full transaction history
        forecast_pred      : (NUM_CATEGORIES,) point-estimate monthly spend
        forecast_lower     : (NUM_CATEGORIES,) 10th-percentile CI bound
        forecast_upper     : (NUM_CATEGORIES,) 90th-percentile CI bound
        anomaly_result     : dict from AnomalyDetectionTransformer.predict()
        bulkbuy_result     : dict from BulkBuyRecommendationTransformer.predict()
        life_events        : list of dicts from LifeEventDetector.detect()
        behavioral_biases  : list of dicts from BehavioralBiasDetector.detect_all()
        cash_crunch        : dict from CashCrunchPredictor.predict()
        inferred_goal      : dict from GoalInferencer.infer() or None

        Returns
        -------
        list[Suggestion] sorted by confidence descending
        """
        suggestions: list[Suggestion] = []

        if forecast_pred is not None:
            suggestions.extend(
                self._forecast_rules(
                    forecast_pred, user_transactions,
                    forecast_lower=forecast_lower,
                    forecast_upper=forecast_upper,
                )
            )

        if anomaly_result is not None:
            import torch
            score = anomaly_result["anomaly_score"]
            cls   = anomaly_result["anomaly_class"]
            probs = anomaly_result["class_probs"]

            # Convert tensors to numpy/python scalars
            if hasattr(score, "item"):
                score = score.item()
            if hasattr(cls, "item"):
                cls = int(cls.item())
            if hasattr(probs, "numpy"):
                probs = probs.numpy()

            suggestions.extend(
                self._anomaly_rules(
                    anomaly_score=float(score),
                    anomaly_class=int(cls),
                    anomaly_probs=np.array(probs),
                    user_transactions=user_transactions,
                )
            )

        if bulkbuy_result is not None:
            import torch
            prob  = bulkbuy_result["bulk_prob"]
            cat   = bulkbuy_result["target_category"]
            cprob = bulkbuy_result["category_probs"]
            sav   = bulkbuy_result["savings_estimate"]

            if hasattr(prob,  "item"): prob  = prob.item()
            if hasattr(cat,   "item"): cat   = int(cat.item())
            if hasattr(cprob, "numpy"): cprob = cprob.numpy()
            if hasattr(sav,   "item"): sav   = sav.item()

            suggestions.extend(
                self._bulk_buy_rules(
                    bulk_prob=float(prob),
                    target_category=int(cat),
                    category_probs=np.array(cprob),
                    savings_estimate=float(sav),
                    user_transactions=user_transactions,
                )
            )

        if life_events:
            suggestions.extend(self._life_event_rules(life_events))

        if behavioral_biases:
            suggestions.extend(self._behavioral_bias_rules(behavioral_biases))

        if cash_crunch:
            suggestions.extend(self._cash_crunch_rules(cash_crunch))

        if inferred_goal:
            suggestions.extend(self._goal_rules(inferred_goal))

        if run_heuristic_rules:
            # Anomaly fallback — only when the ML model hasn't already flagged one
            if not any(s.type == "anomaly_alert" for s in suggestions):
                suggestions.extend(self._heuristic_anomaly_rules(user_transactions))

            # Heuristic bulk-buy fallback — fires even without a trained model
            existing_bulk_cats = {s.category for s in suggestions if s.type == "bulk_buy_opportunity"}
            suggestions.extend(self._heuristic_bulk_buy_rules(user_transactions, existing_bulk_cats))

            try:
                suggestions.extend(self._subscription_rules(user_transactions))
            except Exception:
                pass

            try:
                suggestions.extend(self._subscription_price_increase_rules(user_transactions))
            except Exception:
                pass

            try:
                suggestions.extend(self._price_intelligence_rules(user_transactions))
            except Exception:
                pass

        # Sort by confidence (highest first)
        suggestions.sort(key=lambda s: s.confidence, reverse=True)
        return suggestions


# ── Category-specific saving tips ────────────────────────────────────────────

def _category_saving_tip(
    category:     str,
    overspend:    float,
    top_merchant: str | None = None,
    txn_count:    int | None = None,
    actual_spend: float | None = None,
) -> str:
    """
    Return a context-aware saving tip. If top_merchant is provided the tip
    names the actual merchant rather than giving a generic suggestion.
    """
    merch_clause = f" at {top_merchant}" if top_merchant else ""
    count_clause = f" ({txn_count} visits)" if txn_count else ""
    actual_clause = f" (you spent ${actual_spend:.0f} there{count_clause})" if actual_spend else ""

    data_tips: dict[str, str] = {
        "dining": (
            f"Your biggest dining spend is{merch_clause}{actual_clause}. "
            "Cooking at home 2 extra nights per week typically cuts dining spend by 20–30%."
        ),
        "groceries": (
            f"You're spending most{merch_clause}{actual_clause}. "
            "Store brands and weekly sales on staples can cut grocery bills 15–25%."
        ),
        "subscriptions": (
            "Check each subscription: if you haven't used it in 30 days, cancel it — "
            "auto-renewals are the #1 source of subscription creep."
        ),
        "transportation": (
            f"Most transportation spend is{merch_clause}{actual_clause}. "
            "Combining errands into fewer trips and using GasBuddy for fuel can help."
        ),
        "utilities": (
            "Call your internet/phone provider and ask for a loyalty discount — "
            "mentioning a competitor rate often saves $15–30/month on the spot."
        ),
        "entertainment": (
            f"Top entertainment spend{merch_clause}{actual_clause}. "
            "Free alternatives: library cards, YouTube, local parks — all $0."
        ),
        "shopping": (
            f"Most shopping spend{merch_clause}{actual_clause}. "
            f"Try a 48-hour rule: add to cart, wait 2 days — ~30% of impulse buys get abandoned."
        ),
        "healthcare": (
            "Compare prescription prices at GoodRx.com — savings of 50–80% vs retail "
            "are common. Ask your doctor about therapeutically equivalent generics."
        ),
        "other": (
            f"Review recent transactions{merch_clause} — look for recurring charges "
            "you may have forgotten about."
        ),
    }
    return data_tips.get(
        category,
        f"Review your {category} spending{merch_clause} for charges you can reduce or cut.",
    )
