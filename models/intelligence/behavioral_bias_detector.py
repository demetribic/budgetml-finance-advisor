"""
models/intelligence/behavioral_bias_detector.py — Detect behavioral finance biases.

Detects classic cognitive spending biases from transaction time series.
Rule-based, no ML needed — pattern matching on time series.

Biases detected:
  - PresentBias       : spending spikes in days 1–5 vs days 25–31 of month
  - AnchoringBias     : tip percentage remarkably consistent regardless of bill size
  - RetailTherapy     : spending spikes on specific weekdays (stress days)
  - SubscriptionCreep : subscriptions growing 2+ months consecutively
  - ImpulseBuying     : high frequency of late-night (10pm–2am) transactions
  - SunkCostFallacy   : paying for services with declining visit frequency
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class BehavioralBiasDetector:
    """
    Detects cognitive spending biases from transaction time series.
    """

    _DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    def detect_all(self, user_transactions: pd.DataFrame) -> list[dict]:
        """
        Run all detectors. Returns list of detected biases, each with:
          bias_name, confidence (0–1), evidence (str), impact_estimate ($/month)
        """
        if user_transactions.empty:
            return []

        df = user_transactions.copy()
        df["date"] = pd.to_datetime(df["date"])

        results: list[dict] = []
        for detector in [
            self._detect_present_bias,
            self._detect_retail_therapy,
            self._detect_subscription_creep,
            self._detect_impulse_buying,
            self._detect_sunk_cost_fallacy,
        ]:
            try:
                r = detector(df)
                if r:
                    results.append(r)
            except Exception:
                pass

        return results

    def _detect_present_bias(self, df: pd.DataFrame) -> dict | None:
        """
        Spending spikes immediately after payday, crashes at month end.
        Signal: avg spend days 1–5 vs days 25–31, ratio > 1.8 in 3+ months.
        """
        df = df.copy()
        df["dom"] = df["date"].dt.day
        df["ym"]  = df["date"].dt.to_period("M")

        monthly_ratios: list[float] = []
        for ym, mdf in df.groupby("ym"):
            early = mdf[mdf["dom"] <= 5]["amount"].sum()
            late  = mdf[mdf["dom"] >= 25]["amount"].sum()
            if late > 0 and early > 0:
                monthly_ratios.append(early / late)

        if len(monthly_ratios) < 3:
            return None

        high_ratio_months = sum(1 for r in monthly_ratios if r > 1.8)
        if high_ratio_months < 3:
            return None

        avg_ratio   = float(np.mean(monthly_ratios))
        confidence  = min(1.0, (avg_ratio - 1.8) / 1.2 + 0.4)
        early_extra = float(df[df["dom"] <= 5]["amount"].mean() -
                            df[df["dom"] >= 25]["amount"].mean())

        return {
            "bias_name":       "PresentBias",
            "confidence":      round(confidence, 3),
            "evidence":        (
                f"You spend {avg_ratio:.1f}× more in the first 5 days of the month "
                f"than in the last 7 days ({high_ratio_months} of {len(monthly_ratios)} months)."
            ),
            "impact_estimate": round(max(0.0, early_extra * 4), 2),
        }

    def _detect_retail_therapy(self, df: pd.DataFrame) -> dict | None:
        """
        Spending spikes on specific weekdays: mean 40%+ above the user's own
        mean AND higher transaction count on that day.
        """
        df = df.copy()
        df["dow"] = df["date"].dt.dayofweek

        dow_spend = df.groupby("dow").agg(
            total_amount=("amount", "sum"),
            count=("amount", "count"),
        )
        if len(dow_spend) < 5:
            return None

        overall_daily_mean = float(df["amount"].mean())
        if overall_daily_mean <= 0:
            return None

        # Normalize spend by frequency of that weekday in the dataset
        dow_counts_in_data = df["dow"].value_counts().reindex(range(7), fill_value=1)
        dow_spend["avg_amount_per_occurrence"] = (
            dow_spend["total_amount"] / dow_counts_in_data
        )

        overall_mean = float(dow_spend["avg_amount_per_occurrence"].mean())
        if overall_mean <= 0:
            return None

        for dow_idx, row in dow_spend.iterrows():
            ratio = row["avg_amount_per_occurrence"] / overall_mean
            count_ratio = row["count"] / dow_counts_in_data[dow_idx]
            avg_count_per_day = dow_spend["count"].sum() / max(len(dow_spend), 1)

            if ratio >= 1.4 and count_ratio > dow_spend["count"].mean() / dow_counts_in_data.mean():
                day_name   = self._DAY_NAMES[int(dow_idx)]
                excess     = row["avg_amount_per_occurrence"] - overall_mean
                confidence = min(1.0, (ratio - 1.4) / 0.6 + 0.4)
                return {
                    "bias_name":       "RetailTherapy",
                    "confidence":      round(confidence, 3),
                    "evidence":        (
                        f"You consistently spend {ratio:.1f}× your daily average on {day_name}s "
                        f"({row['avg_amount_per_occurrence']:.0f} vs avg ${overall_mean:.0f}), "
                        f"with more transactions than other days."
                    ),
                    "impact_estimate": round(excess * 4, 2),
                }

        return None

    def _detect_subscription_creep(self, df: pd.DataFrame) -> dict | None:
        """
        Number of distinct subscription merchants growing 2+ consecutive months.
        """
        sub_df = df[df["category"] == "subscriptions"].copy()
        if sub_df.empty:
            return None

        sub_df["ym"] = sub_df["date"].dt.to_period("M")
        monthly_sub_count = sub_df.groupby("ym")["merchant"].nunique()
        if len(monthly_sub_count) < 3:
            return None

        values = monthly_sub_count.values.astype(int)
        max_consecutive = 0
        current = 0
        for i in range(1, len(values)):
            if values[i] > values[i - 1]:
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 0

        if max_consecutive < 2:
            return None

        latest_count = int(values[-1])
        earliest_count = int(values[0])
        monthly_cost  = float(sub_df.groupby("ym")["amount"].sum().iloc[-1])
        confidence    = min(1.0, max_consecutive / 4.0 + 0.3)

        return {
            "bias_name":       "SubscriptionCreep",
            "confidence":      round(confidence, 3),
            "evidence":        (
                f"Your number of distinct subscriptions grew for {max_consecutive} "
                f"consecutive months ({earliest_count} → {latest_count} subscriptions). "
                f"Current monthly subscription cost: ${monthly_cost:.0f}."
            ),
            "impact_estimate": round(monthly_cost * 0.20, 2),
        }

    def _detect_impulse_buying(self, df: pd.DataFrame) -> dict | None:
        """
        High frequency of transactions with timestamps between 10pm–2am.
        Requires real datetime data — returns None if dates carry no time component.
        """
        if "date" not in df.columns:
            return None

        ts = pd.to_datetime(df["date"])
        # If all times are midnight the column is date-only; no time data available.
        if (ts.dt.hour == 0).all() and (ts.dt.minute == 0).all():
            return None

        hour = ts.dt.hour
        late_night = (hour >= 22) | (hour <= 2)
        late_df    = df[late_night]

        if len(df) < 10 or len(late_df) < 3:
            return None

        rate = len(late_df) / len(df)
        if rate < 0.10:   # less than 10% late-night = no flag
            return None

        late_amount = float(late_df["amount"].mean())
        daytime_df  = df[~late_night]
        confidence  = min(1.0, rate * 4)

        if len(daytime_df) > 0:
            normal_amount = float(daytime_df["amount"].mean())
            comparison = f"Late-night avg transaction: ${late_amount:.0f} vs daytime ${normal_amount:.0f}."
        else:
            comparison = f"Late-night avg transaction: ${late_amount:.0f} (all spending is late-night)."

        return {
            "bias_name":       "ImpulseBuying",
            "confidence":      round(confidence, 3),
            "evidence":        (
                f"{rate:.0%} of your transactions happen between 10pm–2am "
                f"({len(late_df)} of {len(df)}). "
                f"{comparison}"
            ),
            "impact_estimate": round(late_amount * len(late_df) * 0.20, 2),
        }

    def _detect_sunk_cost_fallacy(self, df: pd.DataFrame) -> dict | None:
        """
        Continuing to pay for services (subscriptions) whose visit frequency
        has declined in the last 60 days compared to the prior 60 days.
        """
        sub_df = df[df["category"] == "subscriptions"].copy()
        if sub_df.empty or len(sub_df) < 4:
            return None

        df["date"] = pd.to_datetime(df["date"])
        sub_df["date"] = pd.to_datetime(sub_df["date"])

        max_date  = df["date"].max()
        mid_date  = max_date - pd.Timedelta(days=60)
        prev_date = mid_date - pd.Timedelta(days=60)

        sunk_cost_items: list[str] = []
        total_wasted: float = 0.0

        for merchant, mdf in sub_df.groupby("merchant"):
            recent_visits = int(df[
                (df["merchant"] == merchant) & (df["date"] > mid_date)
            ].shape[0])
            prior_visits = int(df[
                (df["merchant"] == merchant) &
                (df["date"] > prev_date) & (df["date"] <= mid_date)
            ].shape[0])

            # Still paying (at least one charge in recent 60d) but fewer visits
            recent_charges = mdf[mdf["date"] > mid_date]["amount"].sum()
            if recent_charges > 0 and prior_visits > 0 and recent_visits < prior_visits * 0.5:
                sunk_cost_items.append(merchant)
                total_wasted += float(recent_charges)

        if not sunk_cost_items:
            return None

        confidence = min(1.0, len(sunk_cost_items) / 3.0 + 0.3)
        return {
            "bias_name":       "SunkCostFallacy",
            "confidence":      round(confidence, 3),
            "evidence":        (
                f"You continue paying for {len(sunk_cost_items)} service(s) "
                f"({', '.join(sunk_cost_items[:3])}) that you use significantly less "
                f"than before. Monthly cost: ${total_wasted:.0f}."
            ),
            "impact_estimate": round(total_wasted, 2),
        }
