"""
rules/subscription_analyzer.py — Subscription detection and trap analysis.

Detects recurring charges and evaluates whether they're worth keeping based on
usage frequency. Generates personalized negotiation scripts for low-usage subs.
"""

from __future__ import annotations

import pandas as pd


class SubscriptionAnalyzer:
    """
    Identifies subscriptions from transaction history and analyzes their value.

    Detection: recurring charges within ±3 days of the same day-of-month,
    same merchant, similar amount (within 10%), for 3+ consecutive months.

    For each detected subscription:
    - Compute usage score: visit frequency to merchant in past 60 days
    - Flag as "trap" if usage_score < LOW_USAGE_THRESHOLD
    - Generate a personalized cancellation/negotiation script
    """

    LOW_USAGE_THRESHOLD = 0.5   # less than 1 non-subscription visit per 2 months

    NEGOTIATION_SCRIPTS: dict[str, str] = {
        "gym":       ("Hi, I've been a member for {tenure_months} months but I'm considering "
                      "canceling due to my busy schedule. Do you have any retention offers "
                      "or a pause option?"),
        "streaming": ("I'd like to review my subscription. I've been a customer for "
                      "{tenure_months} months. Is there a loyalty discount or annual plan "
                      "available that would save me money?"),
        "default":   ("I've been a customer for {tenure_months} months and I'm reviewing "
                      "all my subscriptions. Can you offer a better rate or I'll need to "
                      "cancel?"),
    }

    _GYM_KEYWORDS      = {"gym", "fitness", "planet", "24 hour", "anytime", "equinox", "ymca"}
    _STREAMING_KEYWORDS = {"netflix", "spotify", "hulu", "disney", "apple tv", "youtube", "amazon prime"}
    _MIN_MONTHS        = 2

    # Peer-to-peer and payment-pass-through services — recurring charges here
    # are user-initiated transfers, not subscriptions, so skip them entirely.
    _P2P_BLOCKLIST = frozenset({
        "apple cash", "venmo", "zelle", "cash app", "cashapp",
        "paypal", "google pay", "samsung pay", "square cash",
        "chime", "current", "wise", "revolut",
    })

    def analyze(self, user_transactions: pd.DataFrame) -> list[dict]:
        """
                Returns list of detected subscriptions, each with:
                    merchant, category, monthly_cost, tenure_months, usage_score,
                    is_trap (bool), status, next_expected_charge_date, last_charge_date,
                    charge_count_detected, negotiation_script, annual_cost,
                    cancel_instructions, action_recommendation
        """
        if user_transactions.empty:
            return []

        df = user_transactions.copy()
        df["date"] = pd.to_datetime(df["date"])

        recurring = self._detect_recurring(df)
        results: list[dict] = []

        for sub in recurring:
            merchant  = sub["merchant"]
            tenure    = sub["tenure_months"]
            monthly   = sub["monthly_cost"]
            dom       = sub["day_of_month"]
            charge_count = int(sub.get("charge_count_detected", tenure))
            raw_date = sub.get("last_charge_date")
            if not raw_date:
                continue
            last_charge_date = pd.to_datetime(raw_date, errors="coerce")
            if pd.isna(last_charge_date):
                continue
            next_charge_date = (last_charge_date + pd.Timedelta(days=30)).date()

            # For digital/streaming services, bank data only ever shows the
            # monthly charge — there are no "I used this today" transactions.
            # usage_score would always be 0, making every digital sub look like
            # a trap. Instead, classify these by cost only.
            is_digital = (
                any(kw in merchant.lower() for kw in self._STREAMING_KEYWORDS)
                or sub["category"] == "subscriptions"
            )

            if is_digital:
                usage_score = None
                # Flag as watch only if expensive (>$25/mo); otherwise healthy
                if monthly > 25:
                    is_trap = False
                    status = "watch"
                    action_recommendation = "Check if you're getting value for the price"
                else:
                    is_trap = False
                    status = "healthy"
                    action_recommendation = "Keep active"
            else:
                # Physical services (gym, etc.) — visits can appear in bank data
                cutoff = df["date"].max() - pd.Timedelta(days=60)
                all_visits   = df[(df["merchant"] == merchant) & (df["date"] >= cutoff)]
                day_diff = (all_visits["date"].dt.day - dom).abs()
                charge_visits = all_visits[day_diff.clip(upper=30).apply(lambda d: min(d, 30 - d)) <= 3]
                non_charge    = len(all_visits) - len(charge_visits)
                usage_score   = non_charge / 2.0

                is_trap = usage_score < self.LOW_USAGE_THRESHOLD
                if is_trap:
                    status = "trap"
                    action_recommendation = "You're paying but not using it — consider canceling"
                elif usage_score < 1.0:
                    status = "watch"
                    action_recommendation = "Usage is light — keep an eye on it"
                else:
                    status = "healthy"
                    action_recommendation = "Keep active"

            script = self._generate_script(merchant, tenure)
            cancel_instructions = (
                f"Log in to {merchant}'s website and navigate to Account → Subscription → Cancel. "
                f"Alternatively, call their customer service and reference your {tenure}-month tenure."
            )

            results.append({
                "merchant":            merchant,
                "category":            sub["category"],
                "monthly_cost":        round(monthly, 2),
                "annual_cost":         round(monthly * 12, 2),
                "tenure_months":       tenure,
                "usage_score":         round(usage_score, 2),
                "is_trap":             is_trap,
                "status":              status,
                "next_expected_charge_date": str(next_charge_date),
                "last_charge_date":    str(last_charge_date.date()),
                "charge_count_detected": charge_count,
                "action_recommendation": action_recommendation,
                "negotiation_script":  script,
                "cancel_instructions": cancel_instructions,
            })

        return results

    def _generate_script(self, merchant: str, tenure_months: int) -> str:
        """Select and fill the appropriate negotiation script."""
        m = merchant.lower()
        if any(kw in m for kw in self._GYM_KEYWORDS):
            template = self.NEGOTIATION_SCRIPTS["gym"]
        elif any(kw in m for kw in self._STREAMING_KEYWORDS):
            template = self.NEGOTIATION_SCRIPTS["streaming"]
        else:
            template = self.NEGOTIATION_SCRIPTS["default"]
        return template.format(tenure_months=tenure_months)

    def _detect_recurring(self, df: pd.DataFrame) -> list[dict]:
        """
        Find transactions that recur monthly at a similar amount and day-of-month.
        """
        df = df.copy()
        df["ym"]  = df["date"].dt.to_period("M")
        df["dom"] = df["date"].dt.day

        recurring: list[dict] = []

        for merchant, mdf in df.groupby("merchant"):
            if any(kw in str(merchant).lower() for kw in self._P2P_BLOCKLIST):
                continue

            monthly = mdf.groupby("ym").agg(
                amount=("amount", "mean"),
                dom=("dom", "median"),
            )
            if len(monthly) < self._MIN_MONTHS:
                continue

            amounts = monthly["amount"].values
            doms    = monthly["dom"].values
            mean_amt = float(amounts.mean())
            mean_dom = float(doms.mean())

            amt_ok = all(abs(a - mean_amt) / max(mean_amt, 1e-6) < 0.10 for a in amounts)
            dom_ok = all(min(abs(d - mean_dom), 30 - abs(d - mean_dom)) <= 3 for d in doms)

            if not (amt_ok and dom_ok and mean_amt > 0):
                continue

            _mode = mdf["category"].mode()
            category = str(_mode.iloc[0]) if not _mode.empty else "subscriptions"
            recurring.append({
                "merchant":     str(merchant),
                "monthly_cost": mean_amt,
                "day_of_month": int(round(mean_dom)),
                "tenure_months": len(monthly),
                "charge_count_detected": int(len(mdf)),
                "last_charge_date": mdf["date"].max(),
                "category":     category,
            })

        return recurring
