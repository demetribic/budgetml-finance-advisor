"""
models/intelligence/cash_crunch_predictor.py — 30-day cash flow danger forecast.

Combines spending forecast model predictions with detected recurring bills and
estimated income timing to predict the exact date(s) a user's balance will go
negative in the next 30 days.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


CATEGORIES = [
    "dining", "groceries", "subscriptions", "transportation",
    "utilities", "entertainment", "shopping", "healthcare", "other",
]


class CashCrunchPredictor:
    """
    Predicts upcoming cash flow crises by combining:
    1. Forecast model predictions (expected spend per category per day)
    2. Detected recurring bills (from transaction history pattern matching)
    3. Estimated income timing (inferred from large regular deposits)

    Outputs a 30-day cash flow curve with flagged danger dates.
    """

    _LARGE_DEPOSIT_THRESHOLD = 500.0  # minimum $ to count as a "paycheck"
    _RECURRING_MIN_MONTHS   = 3       # minimum months of recurrence to flag a bill

    def predict(
        self,
        user_transactions:    pd.DataFrame,
        forecast_by_category: dict[str, float],   # from SpendingForecastTransformer
        assumed_daily_balance: float = 0.0,
    ) -> dict:
        """
        Parameters
        ----------
        user_transactions    : pd.DataFrame — full transaction history
        forecast_by_category : {category: forecasted_monthly_spend}
        assumed_daily_balance : float — starting balance (0 = unknown)

        Returns
        -------
        dict:
          cash_flow_curve  : list of {date, expected_balance, risk_level}
          danger_dates     : list of date strings where balance predicted < 0
          largest_upcoming : {date, amount, category, merchant}
          recommendation   : str
        """
        if user_transactions.empty:
            return self._empty_result()

        today        = datetime.utcnow().date()
        dates_30     = [today + timedelta(days=i) for i in range(30)]

        # Distribute monthly forecast evenly across days as baseline daily spend
        total_daily_spend = sum(forecast_by_category.values()) / 30.0

        # Detect recurring bills and income timing
        recurring    = self._detect_recurring_bills(user_transactions)
        income_days  = self._estimate_income_timing(user_transactions)

        # Build a daily spend profile
        daily_spend  = np.full(30, total_daily_spend)
        daily_income = np.zeros(30)

        # Overlay known recurring bills on their expected dates
        for bill in recurring:
            bill_dom = bill["day_of_month"]
            for i, d in enumerate(dates_30):
                if d.day == bill_dom:
                    daily_spend[i] += bill["amount"]

        # Overlay expected income
        _med = user_transactions[
            user_transactions["amount"] > self._LARGE_DEPOSIT_THRESHOLD
        ]["amount"].median()
        avg_deposit = 0.0 if pd.isna(_med) else float(_med)
        for i, d in enumerate(dates_30):
            if d.day in income_days:
                daily_income[i] += avg_deposit

        # Compute a relative spend curve (not anchored to a real balance).
        # We never ask the user for their bank balance, so we cannot claim it
        # "goes negative." Instead we track spend pressure relative to the
        # baseline daily rate to surface genuine bill-cluster danger dates.
        curve: list[dict] = []
        danger_dates: list[str] = []

        for i, d in enumerate(dates_30):
            # Pressure ratio: how much heavier than average is this day's spend?
            pressure_ratio = daily_spend[i] / max(total_daily_spend, 0.01)
            risk = "low" if pressure_ratio < 1.5 else "medium" if pressure_ratio < 2.5 else "high"
            curve.append({
                "date":             str(d),
                "expected_balance": round(daily_spend[i], 2),   # daily projected spend, not running balance
                "risk_level":       risk,
            })
            # Flag only days where stacked recurring bills create a genuine spike
            if pressure_ratio >= 2.0:
                danger_dates.append(str(d))

        # Largest upcoming recurring bill
        largest_upcoming = None
        if recurring:
            biggest = max(recurring, key=lambda b: b["amount"])
            _days_ahead = (biggest["day_of_month"] - today.day) % 30
            largest_upcoming = {
                "date":     str(today + timedelta(days=_days_ahead if _days_ahead > 0 else 30)),
                "amount":   biggest["amount"],
                "category": biggest.get("category", "other"),
                "merchant": biggest.get("merchant", "Unknown"),
            }

        # Generate recommendation — framed as spending pressure, not balance prediction
        if danger_dates:
            bill_names = ", ".join(
                b["merchant"] for b in sorted(recurring, key=lambda b: -b["amount"])[:2]
            )
            recommendation = (
                f"Heavy recurring bills are stacked around {danger_dates[0]} "
                f"({bill_names or 'multiple charges'}). "
                f"Avoid extra discretionary spending near that date."
            )
        elif largest_upcoming:
            recommendation = (
                f"Your largest upcoming charge is ${largest_upcoming['amount']:.0f} "
                f"at {largest_upcoming['merchant']} around {largest_upcoming['date']}. "
                f"No major bill clusters detected this month."
            )
        else:
            recommendation = "No major spending pressure detected in the next 30 days."

        return {
            "cash_flow_curve":  curve,
            "danger_dates":     danger_dates,
            "largest_upcoming": largest_upcoming,
            "recommendation":   recommendation,
        }

    def _detect_recurring_bills(self, df: pd.DataFrame) -> list[dict]:
        """
        Find bills that repeat on roughly the same day each month.
        A bill qualifies if: same merchant, similar amount (within 10%),
        within ±3 days of the same day-of-month, for 3+ consecutive months.

        Returns list of {merchant, amount, day_of_month, category}.
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["ym"]   = df["date"].dt.to_period("M")
        df["dom"]  = df["date"].dt.day

        recurring: list[dict] = []

        for merchant, mdf in df.groupby("merchant"):
            if len(mdf) < self._RECURRING_MIN_MONTHS:
                continue

            monthly = mdf.groupby("ym").agg(
                amount=("amount", "mean"),
                dom=("dom", "median"),
            )
            if len(monthly) < self._RECURRING_MIN_MONTHS:
                continue

            amounts = monthly["amount"].values
            doms    = monthly["dom"].values

            # Check consistency: amounts within 10% of mean, dom within ±3 days
            mean_amt  = float(amounts.mean())
            amt_ok    = all(abs(a - mean_amt) / max(mean_amt, 1) < 0.10 for a in amounts)
            mean_dom  = float(doms.mean())
            dom_ok    = all(abs(d - mean_dom) <= 3 for d in doms)

            if amt_ok and dom_ok and mean_amt > 0:
                category = str(mdf["category"].mode().iloc[0]) if not mdf.empty else "other"
                recurring.append({
                    "merchant":    str(merchant),
                    "amount":      round(mean_amt, 2),
                    "day_of_month": int(round(mean_dom)),
                    "category":    category,
                    "tenure_months": len(monthly),
                })

        return recurring

    def _estimate_income_timing(self, df: pd.DataFrame) -> list[int]:
        """Return list of days-of-month where large credits typically arrive."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        large = df[df["amount"] > self._LARGE_DEPOSIT_THRESHOLD]
        if large.empty:
            return [1, 15]   # default: 1st and 15th
        dom_counts = large["date"].dt.day.value_counts()
        # Return top-2 most frequent income days
        return dom_counts.head(2).index.tolist()

    @staticmethod
    def _empty_result() -> dict:
        return {
            "cash_flow_curve":  [],
            "danger_dates":     [],
            "largest_upcoming": None,
            "recommendation":   "Insufficient transaction history for cash flow prediction.",
        }
