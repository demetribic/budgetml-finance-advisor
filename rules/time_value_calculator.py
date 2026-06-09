"""
rules/time_value_calculator.py — Opportunity cost calculator using time value of money.

Converts monthly spending amounts into 10/20/30-year opportunity costs assuming
compound growth at the S&P 500 historical average (7% annually).

Used to frame suggestions in terms of long-term wealth impact.
"""

from __future__ import annotations

import math


class TimeValueCalculator:
    """
    Converts monthly spending amounts into future opportunity costs.
    """

    DEFAULT_ANNUAL_RATE: float = 0.07   # S&P 500 historical average

    def monthly_to_future_value(
        self,
        monthly_amount: float,
        years:          int   = 20,
        annual_rate:    float = DEFAULT_ANNUAL_RATE,
    ) -> float:
        """
        Future value of investing `monthly_amount` every month for `years` years.

        Uses the future value of an annuity formula:
            FV = PMT × ((1 + r)^n - 1) / r
        where r = monthly rate and n = total months.

        Parameters
        ----------
        monthly_amount : float   monthly amount (dollars)
        years          : int     investment horizon (default 20)
        annual_rate    : float   annual growth rate (default 0.07)

        Returns
        -------
        float : future value in dollars
        """
        if monthly_amount <= 0 or years <= 0:
            return 0.0
        r = annual_rate / 12.0    # monthly rate
        n = years * 12            # total months
        if r == 0:
            return monthly_amount * n
        return monthly_amount * ((1 + r) ** n - 1) / r

    def format_opportunity_cost(
        self,
        monthly_amount: float,
        category:       str,
        years:          int = 20,
        label:          str | None = None,
    ) -> str:
        """
        Return a human-readable opportunity cost string.

        Parameters
        ----------
        label : override the "monthly_amount on category" phrase, e.g. "this overspend"
        """
        fv = self.monthly_to_future_value(monthly_amount, years=years)
        phrase = label or f"${monthly_amount:.0f}/month in {category} overspend"
        return (
            f"Investing {phrase} instead → ${fv:,.0f} in {years} yrs at "
            f"{self.DEFAULT_ANNUAL_RATE:.0%} growth"
        )

    def opportunity_cost_dict(
        self,
        monthly_amount: float,
        category:       str,
        label:          str | None = None,
    ) -> dict:
        """
        Return a dict with opportunity costs at 10, 20, and 30 years.
        Suitable for adding to Suggestion.details.
        """
        return {
            "monthly_amount":         round(monthly_amount, 2),
            "opportunity_cost_10yr":  round(self.monthly_to_future_value(monthly_amount, years=10), 2),
            "opportunity_cost_20yr":  round(self.monthly_to_future_value(monthly_amount, years=20), 2),
            "opportunity_cost_30yr":  round(self.monthly_to_future_value(monthly_amount, years=30), 2),
            "opportunity_cost_message": self.format_opportunity_cost(monthly_amount, category, label=label),
        }
