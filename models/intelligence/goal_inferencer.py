"""
models/intelligence/goal_inferencer.py — Infer user financial goals from spending patterns.

Detects emerging savings behavior from spending trends using exponential smoothing
on weekly savings rates and pattern matching on category shifts.

No training needed — purely signal-based inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


GOAL_SIGNATURES = {
    "EmergencyFund": {
        "description": "You appear to be building an emergency fund",
        "signals":     ["savings_rate_rising", "no_large_discretionary"],
        "target_multiplier": 3.0,   # target = 3 months of spend
    },
    "VacationSaving": {
        "description": "You appear to be saving for a vacation",
        "signals":     ["savings_rate_rising", "travel_category_appearing"],
        "target_multiplier": 1.5,
    },
    "DebtPayoff": {
        "description": "You appear to be aggressively paying off debt",
        "signals":     ["transfer_spiking", "discretionary_falling"],
        "target_multiplier": None,
    },
    "LargePurchase": {
        "description": "You appear to be saving toward a large purchase",
        "signals":     ["irregular_large_savings", "shopping_falling"],
        "target_multiplier": None,
    },
    "RetirementFocus": {
        "description": "You appear to be prioritizing retirement savings",
        "signals":     ["investment_merchants_present", "entertainment_falling"],
        "target_multiplier": None,
    },
}

_INVESTMENT_KEYWORDS = {"fidelity", "vanguard", "schwab", "robinhood", "etrade", "401k", "ira"}
_TRAVEL_KEYWORDS    = {"airline", "flight", "hotel", "airbnb", "expedia", "kayak", "travel"}


class GoalInferencer:
    """
    Infers user financial goals from spending pattern shifts.
    Uses exponential smoothing on weekly savings rates.
    """

    def infer(self, user_transactions: pd.DataFrame) -> dict | None:
        """
        Returns None if no clear goal detected. Otherwise:
          goal_name          : str
          confidence         : float
          monthly_spend_drop : float  (observable spend reduction vs baseline, $/month)
          estimated_target   : float | None  (rule-of-thumb benchmark, NOT actual saved amount)
          message            : str
        """
        if user_transactions.empty:
            return None

        df = user_transactions.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        # Compute weekly totals
        df["week"] = df["date"].dt.to_period("W")
        weekly_spend = df.groupby("week")["amount"].sum()

        if len(weekly_spend) < 6:
            return None

        # Exponential smoothing on weekly spend to detect savings trend
        alpha   = 0.3
        smooth  = _ema(weekly_spend.values, alpha)
        trend   = smooth[-1] - smooth[0]   # negative = spending falling (saving more)

        signals = self._detect_signals(df)

        # Score each goal
        best_goal: str | None = None
        best_score = 0.0

        for goal_name, spec in GOAL_SIGNATURES.items():
            required = spec["signals"]
            matched  = sum(1 for s in required if s in signals)
            score    = matched / len(required)
            if score > best_score and score >= 0.5:
                best_score = score
                best_goal  = goal_name

        if best_goal is None:
            return None

        # Observable fact: how much less per week are you spending recently vs baseline?
        # We cannot know how much was "saved" — that requires savings account data.
        # We only know spending went down.
        baseline_weekly = float(weekly_spend.values[:4].mean())
        recent_weekly   = float(weekly_spend.values[-4:].mean()) if len(weekly_spend) >= 4 else baseline_weekly
        weekly_spend_drop = max(0.0, baseline_weekly - recent_weekly)
        monthly_spend_drop = weekly_spend_drop * 4.33

        # Suppress if spending hasn't meaningfully changed — no signal to report.
        if monthly_spend_drop < 25.0:
            return None

        # Reference targets (rule-of-thumb benchmarks, not actual saved amounts)
        spec = GOAL_SIGNATURES[best_goal]
        monthly_spend   = baseline_weekly * 4.33
        target_mult     = spec.get("target_multiplier")
        estimated_target = monthly_spend * target_mult if target_mult else None

        confidence = min(1.0, best_score * 0.7 + min(1.0, monthly_spend_drop / 500) * 0.3)
        desc       = spec["description"]
        message    = (
            f"{desc} — your spending has dropped by ~${monthly_spend_drop:.0f}/month recently."
        )
        if estimated_target is not None:
            message += (
                f" A common benchmark for this goal is ~${estimated_target:.0f} "
                f"(based on your spending level)."
            )

        return {
            "goal_name":          best_goal,
            "confidence":         round(confidence, 3),
            "monthly_spend_drop": round(monthly_spend_drop, 2),
            "estimated_target":   round(estimated_target, 2) if estimated_target else None,
            "message":            message,
        }

    def _detect_signals(self, df: pd.DataFrame) -> set[str]:
        """Detect which goal signals are present in the transaction history."""
        signals: set[str] = set()
        df["week"]  = df["date"].dt.to_period("W")
        df["month"] = df["date"].dt.to_period("M")

        # Weekly spend trend (recent 4 weeks vs prior 4 weeks)
        weekly_spend = df.groupby("week")["amount"].sum().values
        if len(weekly_spend) >= 8:
            recent_4  = weekly_spend[-4:].mean()
            prior_4   = weekly_spend[-8:-4].mean()
            if recent_4 < prior_4 * 0.85:
                signals.add("savings_rate_rising")

        # No large discretionary: dining + entertainment + shopping declining
        discr = df[df["category"].isin(["dining", "entertainment", "shopping"])]
        if not discr.empty:
            monthly_discr = discr.groupby("month")["amount"].sum().values
            if len(monthly_discr) >= 3 and monthly_discr[-1] < monthly_discr[0] * 0.8:
                signals.add("no_large_discretionary")
                signals.add("discretionary_falling")

        # Transfer/payment spiking
        transfer_df = df[(df["category"].isin(["other"])) & (df["amount"] > 200)]
        if len(transfer_df) >= 3:
            monthly_t = transfer_df.groupby("month")["amount"].sum().values
            if len(monthly_t) >= 2 and monthly_t[-1] > monthly_t[0] * 1.5:
                signals.add("transfer_spiking")

        # Travel merchants appearing recently
        recent_merchants = set(
            df[df["date"] >= df["date"].max() - pd.Timedelta(days=60)]["merchant"]
            .fillna("").astype(str).str.lower()
        )
        if any(kw in m for m in recent_merchants for kw in _TRAVEL_KEYWORDS):
            signals.add("travel_category_appearing")

        # Investment merchants appearing
        all_merchants = set(df["merchant"].fillna("").astype(str).str.lower())
        if any(kw in m for m in all_merchants for kw in _INVESTMENT_KEYWORDS):
            signals.add("investment_merchants_present")

        # Irregular large savings pattern
        large = df[df["amount"] > df["amount"].quantile(0.90)]
        if len(large) >= 3:
            signals.add("irregular_large_savings")

        # Shopping falling
        shop = df[df["category"] == "shopping"]
        if not shop.empty:
            monthly_shop = shop.groupby("month")["amount"].sum().values
            if len(monthly_shop) >= 3 and monthly_shop[-1] < monthly_shop[0] * 0.8:
                signals.add("shopping_falling")

        # Entertainment falling
        ent = df[df["category"] == "entertainment"]
        if not ent.empty:
            monthly_ent = ent.groupby("month")["amount"].sum().values
            if len(monthly_ent) >= 3 and monthly_ent[-1] < monthly_ent[0] * 0.8:
                signals.add("entertainment_falling")

        return signals


def _ema(values: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average."""
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out
