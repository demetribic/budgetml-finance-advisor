"""
api/counterfactual.py — Counterfactual Spending Simulator

"What if you had cooked at home 3x/week last year?" — replay a user's actual
transaction history with a hypothetical substitution and show the compounded outcome.

POST /simulate → SimulateResponse
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

CATEGORIES = [
    "dining", "groceries", "subscriptions", "transportation",
    "utilities", "entertainment", "shopping", "healthcare", "other",
]


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class Transaction(BaseModel):
    date: str
    amount: float = Field(default=0.0, ge=0)
    category: str = "other"
    merchant: str = "Unknown"
    description: str = ""


class SimulationScenario(BaseModel):
    type: str  # reduce_category | eliminate_merchant | home_cooking | reduce_frequency | subscription_trim
    category: Optional[str] = None
    merchant: Optional[str] = None
    reduction_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    frequency_reduction: Optional[int] = None  # fewer transactions per month
    annual_return_rate: float = Field(default=0.07, ge=0.0, le=0.30)


class SimulateRequest(BaseModel):
    transactions: list[Transaction] = Field(..., min_length=1)
    scenario: SimulationScenario
    projection_years: int = Field(default=10, ge=1, le=40)


class MonthlyImpact(BaseModel):
    month: str
    actual_spend: float
    simulated_spend: float
    monthly_savings: float
    cumulative_savings: float


class SimulateResponse(BaseModel):
    scenario_description: str
    total_savings_over_period: float
    avg_monthly_savings: float
    compound_value_at_end: float
    monthly_timeline: list[MonthlyImpact]
    insight_message: str
    breakeven_months: Optional[int]


# ── Router ────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="", tags=["simulation"])


# ── Scenario application ──────────────────────────────────────────────────────

def _apply_reduce_category(df: pd.DataFrame, category: str, reduction_pct: float) -> pd.DataFrame:
    """Reduce all spending in a category by reduction_pct (0.0–1.0)."""
    sim = df.copy()
    mask = sim["category"] == category
    sim.loc[mask, "amount"] = sim.loc[mask, "amount"] * (1.0 - reduction_pct)
    return sim


def _apply_eliminate_merchant(
    df: pd.DataFrame,
    merchant: str,
    category: Optional[str] = None,
) -> pd.DataFrame:
    """Remove all transactions at a specific merchant (case-insensitive contains)."""
    sim = df.copy()
    merch_mask = sim["merchant"].str.lower().str.contains(merchant.lower(), na=False, regex=False)
    if category:
        merch_mask = merch_mask & (sim["category"] == category)
    sim = sim[~merch_mask]
    return sim


def _apply_home_cooking(df: pd.DataFrame, reduction_pct: float = 0.75) -> pd.DataFrame:
    """
    Substitute dining out with home cooking.
    Keeps 25% of dining cost (you'd spend ~25% on groceries instead).
    Optionally grows grocery spend proportionally.
    """
    sim = df.copy()
    dining_mask = sim["category"] == "dining"
    saved = sim.loc[dining_mask, "amount"] * reduction_pct
    sim.loc[dining_mask, "amount"] = sim.loc[dining_mask, "amount"] * (1.0 - reduction_pct)
    # Add 25% of savings back as grocery spend (realistic home cooking cost)
    grocery_additions = saved * 0.25
    grocery_rows = sim.loc[dining_mask].copy()
    grocery_rows["category"] = "groceries"
    grocery_rows["merchant"] = "Home Cooking (groceries)"
    grocery_rows["amount"] = grocery_additions.values
    grocery_rows = grocery_rows[grocery_rows["amount"] > 0]
    sim = pd.concat([sim, grocery_rows], ignore_index=True)
    return sim


def _apply_reduce_frequency(
    df: pd.DataFrame,
    category: str,
    frequency_reduction: int,
) -> pd.DataFrame:
    """
    Remove `frequency_reduction` transactions per month from a category,
    specifically the smallest-amount ones (luxury/impulse purchases).
    """
    sim = df.copy()
    sim["_month"] = pd.to_datetime(sim["date"]).dt.to_period("M")
    cat_mask = sim["category"] == category
    cat_df = sim[cat_mask].copy()

    rows_to_remove = []
    for month, group in cat_df.groupby("_month"):
        if len(group) <= frequency_reduction:
            continue
        # Remove the cheapest transactions (most likely discretionary)
        cheapest = group.nsmallest(frequency_reduction, "amount")
        rows_to_remove.extend(cheapest.index.tolist())

    sim = sim.drop(index=rows_to_remove)
    sim = sim.drop(columns=["_month"])
    return sim


def _apply_subscription_trim(
    df: pd.DataFrame,
    merchant: Optional[str] = None,
    threshold_amount: float = 15.0,
) -> pd.DataFrame:
    """
    Remove unused/low-value subscriptions.
    If merchant specified: remove that specific subscription.
    Otherwise: remove subscriptions under threshold_amount.
    """
    sim = df.copy()
    sub_mask = sim["category"] == "subscriptions"

    if merchant:
        merch_mask = sim["merchant"].str.lower().str.contains(merchant.lower(), na=False, regex=False)
        sim = sim[~(sub_mask & merch_mask)]
    else:
        sim = sim[~(sub_mask & (sim["amount"] <= threshold_amount))]
    return sim


def _apply_scenario(df: pd.DataFrame, scenario: SimulationScenario) -> pd.DataFrame:
    stype = scenario.type
    if stype == "reduce_category":
        if not scenario.category:
            raise HTTPException(400, "reduce_category requires 'category'")
        reduction = scenario.reduction_pct if scenario.reduction_pct is not None else 0.30
        return _apply_reduce_category(df, scenario.category, reduction)

    elif stype == "eliminate_merchant":
        if not scenario.merchant:
            raise HTTPException(400, "eliminate_merchant requires 'merchant'")
        return _apply_eliminate_merchant(df, scenario.merchant, scenario.category)

    elif stype == "home_cooking":
        reduction = scenario.reduction_pct if scenario.reduction_pct is not None else 0.75
        return _apply_home_cooking(df, reduction)

    elif stype == "reduce_frequency":
        if not scenario.category:
            raise HTTPException(400, "reduce_frequency requires 'category'")
        freq = scenario.frequency_reduction if scenario.frequency_reduction is not None else 2
        return _apply_reduce_frequency(df, scenario.category, freq)

    elif stype == "subscription_trim":
        return _apply_subscription_trim(df, scenario.merchant)

    else:
        raise HTTPException(400, f"Unknown scenario type: {stype!r}")


# ── Monthly timeline computation ──────────────────────────────────────────────

def _compute_monthly_timeline(
    original: pd.DataFrame,
    simulated: pd.DataFrame,
) -> list[MonthlyImpact]:
    orig = original.copy()
    sim = simulated.copy()
    orig["_month"] = pd.to_datetime(orig["date"]).dt.to_period("M")
    sim["_month"] = pd.to_datetime(sim["date"]).dt.to_period("M")

    orig_monthly = orig.groupby("_month")["amount"].sum()
    sim_monthly = sim.groupby("_month")["amount"].sum()

    all_months = sorted(set(orig_monthly.index) | set(sim_monthly.index))

    timeline = []
    cumulative = 0.0
    for m in all_months:
        actual = float(orig_monthly.get(m, 0.0))
        simulated_spend = float(sim_monthly.get(m, 0.0))
        savings = round(actual - simulated_spend, 2)
        cumulative = round(cumulative + savings, 2)
        timeline.append(MonthlyImpact(
            month=str(m),
            actual_spend=round(actual, 2),
            simulated_spend=round(simulated_spend, 2),
            monthly_savings=savings,
            cumulative_savings=cumulative,
        ))
    return timeline


# ── Compound value formula ────────────────────────────────────────────────────

def _future_value_annuity(monthly_payment: float, annual_rate: float, years: int) -> float:
    """FV of monthly annuity invested at annual_rate for `years` years."""
    if annual_rate <= 0 or monthly_payment <= 0:
        return monthly_payment * 12 * years
    r = annual_rate / 12.0
    n = years * 12
    return monthly_payment * ((1 + r) ** n - 1) / r


# ── Scenario description generator ───────────────────────────────────────────

def _describe_scenario(scenario: SimulationScenario, avg_monthly_savings: float) -> str:
    stype = scenario.type
    if stype == "reduce_category":
        pct = int((scenario.reduction_pct or 0.3) * 100)
        return f"Cut {scenario.category} spending by {pct}%"
    elif stype == "eliminate_merchant":
        return f"Eliminate {scenario.merchant or 'merchant'} transactions"
    elif stype == "home_cooking":
        pct = int((scenario.reduction_pct or 0.75) * 100)
        return f"Cook at home instead of dining out ({pct}% of dining cost)"
    elif stype == "reduce_frequency":
        freq = scenario.frequency_reduction or 2
        return f"Make {freq} fewer {scenario.category} purchases per month"
    elif stype == "subscription_trim":
        if scenario.merchant:
            return f"Cancel {scenario.merchant} subscription"
        return "Remove low-value subscriptions (under $15/month)"
    return "Custom spending scenario"


def _build_insight_message(
    scenario: SimulationScenario,
    avg_monthly_savings: float,
    compound_20yr: float,
    total_period_savings: float,
    projection_years: int,
) -> str:
    desc = _describe_scenario(scenario, avg_monthly_savings)
    if avg_monthly_savings <= 0:
        return f"{desc} — no savings detected in your transaction history for this scenario."

    parts = [f"{desc} saves you ~${avg_monthly_savings:.0f}/month"]
    if total_period_savings > 0:
        parts.append(f"${total_period_savings:,.0f} over the analyzed period")
    if compound_20yr > 0:
        parts.append(
            f"If invested at 7% annually, that monthly saving grows to "
            f"${compound_20yr:,.0f} in 20 years"
        )
    return ". ".join(parts) + "."


# ── Main endpoint ─────────────────────────────────────────────────────────────

@router.post("/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest) -> SimulateResponse:
    """
    Replay transaction history with a counterfactual scenario applied.
    Returns monthly savings timeline + compound growth projection.
    """
    # Build original DataFrame
    records = [
        {
            "date": t.date,
            "amount": t.amount,
            "category": t.category,
            "merchant": t.merchant,
            "description": t.description,
        }
        for t in req.transactions
    ]
    orig_df = pd.DataFrame(records)
    orig_df["date"] = pd.to_datetime(orig_df["date"])
    orig_df = orig_df.sort_values("date").reset_index(drop=True)

    # Apply scenario
    sim_df = _apply_scenario(orig_df.copy(), req.scenario)

    # Build monthly timeline
    timeline = _compute_monthly_timeline(orig_df, sim_df)

    if not timeline:
        raise HTTPException(422, "No monthly data could be computed from the provided transactions.")

    # Aggregate stats
    savings_list = [m.monthly_savings for m in timeline]
    total_savings = round(sum(savings_list), 2)
    avg_monthly = round(total_savings / len(savings_list), 2) if savings_list else 0.0

    # Compound projection
    rate = req.scenario.annual_return_rate
    years = req.projection_years
    compound_end = round(_future_value_annuity(max(avg_monthly, 0), rate, years), 2)
    compound_20yr = round(_future_value_annuity(max(avg_monthly, 0), rate, 20), 2)

    # Breakeven months (when cumulative savings hits $1,000)
    breakeven: Optional[int] = None
    running = 0.0
    for i, m in enumerate(timeline):
        running += m.monthly_savings
        if running >= 1000.0:
            breakeven = i + 1
            break

    scenario_desc = _describe_scenario(req.scenario, avg_monthly)
    insight = _build_insight_message(
        req.scenario, avg_monthly, compound_20yr,
        total_savings, years,
    )

    return SimulateResponse(
        scenario_description=scenario_desc,
        total_savings_over_period=total_savings,
        avg_monthly_savings=avg_monthly,
        compound_value_at_end=compound_end,
        monthly_timeline=timeline,
        insight_message=insight,
        breakeven_months=breakeven,
    )
