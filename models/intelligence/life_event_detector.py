"""
models/intelligence/life_event_detector.py — Detect life events from spending shifts.

Life events (moved cities, got a pet, new relationship, job change, new child)
produce measurable shifts in spending category distributions. Detected via
Jensen-Shannon divergence between consecutive 30-day spending windows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


CATEGORIES = [
    "dining", "groceries", "subscriptions", "transportation",
    "utilities", "entertainment", "shopping", "healthcare", "other",
]

# Signature patterns: (category, expected direction, min_ratio_change)
# direction: +1 = increase expected, -1 = decrease expected
LIFE_EVENT_SIGNATURES: dict[str, list[tuple[str, int, float]]] = {
    "moved":        [("utilities", +1, 0.30), ("transportation", -1, 0.20), ("shopping", +1, 0.15)],
    "new_pet":      [("healthcare", +1, 0.20), ("shopping", +1, 0.25)],
    "relationship": [("dining", +1, 0.30), ("entertainment", +1, 0.25)],
    "job_change":   [("transportation", +1, 0.40), ("dining", +1, 0.30)],
    "new_child":    [("healthcare", +1, 0.50), ("shopping", +1, 0.40), ("entertainment", -1, 0.30)],
}

# JS divergence threshold to trigger event detection
_JS_THRESHOLD = 0.05


class LifeEventDetector:
    """
    Detects spending regime changes that indicate life events.

    Method: Jensen-Shannon divergence between consecutive 30-day
    category spend distributions. Large divergence = event likely.

    No training needed — threshold is fixed.
    """

    def detect(
        self,
        user_transactions: pd.DataFrame,
        window_days: int = 30,
    ) -> list[dict]:
        """
        Slide two consecutive 30-day windows over the user's history.
        Return detected life events with timestamp and confidence.

        Parameters
        ----------
        user_transactions : pd.DataFrame with columns: date, amount, category
        window_days       : int   window size in days

        Returns
        -------
        list of {event, date, confidence, js_divergence}
        """
        if user_transactions.empty:
            return []

        df = user_transactions.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.sort_values("date")

        min_date = df["date"].min()
        max_date = df["date"].max()
        total_days = (max_date - min_date).days

        if total_days < window_days * 2:
            return []

        events: list[dict] = []
        dates_to_check = pd.date_range(
            min_date + pd.Timedelta(days=window_days),
            max_date - pd.Timedelta(days=window_days),
            freq=f"{window_days}D",
        )

        for pivot in dates_to_check:
            before_start = pivot - pd.Timedelta(days=window_days)
            before_end   = pivot
            after_start  = pivot
            after_end    = pivot + pd.Timedelta(days=window_days)

            before_df = df[(df["date"] >= before_start) & (df["date"] < before_end)]
            after_df  = df[(df["date"] >= after_start)  & (df["date"] < after_end)]

            if before_df.empty or after_df.empty:
                continue

            p = self._category_distribution(before_df)
            q = self._category_distribution(after_df)
            js = self._js_divergence(p, q)

            if js < _JS_THRESHOLD:
                continue

            # Check which specific life event signature best fits
            for event_name, signatures in LIFE_EVENT_SIGNATURES.items():
                matched = self._match_signature(before_df, after_df, signatures)
                if matched:
                    confidence = min(1.0, js / 0.2)
                    events.append({
                        "event":        event_name,
                        "date":         str(pivot.date()),
                        "confidence":   round(confidence, 3),
                        "js_divergence": round(js, 4),
                    })

        # Deduplicate: keep highest confidence per event type
        seen: dict[str, dict] = {}
        for e in events:
            k = e["event"]
            if k not in seen or e["confidence"] > seen[k]["confidence"]:
                seen[k] = e
        return sorted(seen.values(), key=lambda e: e["confidence"], reverse=True)

    def _category_distribution(self, df: pd.DataFrame) -> np.ndarray:
        """Return normalized category spend vector (sums to 1)."""
        spend = df.groupby("category")["amount"].sum()
        vec   = np.array([float(spend.get(c, 0.0)) for c in CATEGORIES])
        total = vec.sum()
        if total <= 0:
            return np.ones(len(CATEGORIES)) / len(CATEGORIES)
        return vec / total

    def _js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Jensen-Shannon divergence between two distributions (0–1 range)."""
        eps = 1e-10
        m   = 0.5 * (p + q + eps)
        kl_pm = np.sum(p * np.log((p + eps) / m))
        kl_qm = np.sum(q * np.log((q + eps) / m))
        return float(0.5 * kl_pm + 0.5 * kl_qm)

    def _match_signature(
        self,
        before_df: pd.DataFrame,
        after_df:  pd.DataFrame,
        signatures: list[tuple[str, int, float]],
    ) -> bool:
        """
        Check if at least half of the signature conditions are met.
        Each condition: (category, direction, min_ratio_change).
        """
        if before_df.empty or after_df.empty:
            return False

        before_spend = before_df.groupby("category")["amount"].sum()
        after_spend  = after_df.groupby("category")["amount"].sum()

        matched = 0
        for cat, direction, min_change in signatures:
            b = float(before_spend.get(cat, 0.0))
            a = float(after_spend.get(cat, 0.0))
            if b <= 0:
                continue
            ratio_change = (a - b) / b
            if direction == +1 and ratio_change >= min_change:
                matched += 1
            elif direction == -1 and ratio_change <= -min_change:
                matched += 1

        return matched >= max(1, len(signatures) // 2)
