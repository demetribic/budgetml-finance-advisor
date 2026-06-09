"""
synthetic.py — Generate realistic synthetic transaction histories as a fallback dataset.

Produces a pandas DataFrame with columns:
    user_id, date, amount, category, merchant, description, is_anomaly, is_bulk_buy

Includes:
  - Weekly/monthly seasonality patterns
  - Built-in spending anomalies (spikes, unusual merchants)
  - Recurring purchases suitable for bulk-buy recommendations
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from data.preprocessor import CATEGORIES  # single canonical source

CATEGORY_SPEND_RANGES: dict[str, tuple[float, float]] = {
    "dining":          (15.0,  85.0),
    "groceries":       (20.0, 200.0),
    "subscriptions":   ( 8.0,  20.0),
    "transportation":  ( 5.0,  60.0),
    "utilities":       (40.0, 180.0),
    "entertainment":   (10.0,  75.0),
    "shopping":        (15.0, 250.0),
    "healthcare":      (10.0, 300.0),
    # "other" is a catch-all label; synthetic data does not generate it directly.
}

CATEGORY_FREQUENCY: dict[str, float] = {
    "dining":          12.0,
    "groceries":        8.0,
    "subscriptions":    3.0,
    "transportation":  15.0,
    "utilities":        2.0,
    "entertainment":    4.0,
    "shopping":         5.0,
    "healthcare":       1.5,
    # "other" is not generated synthetically.
}

MERCHANTS: dict[str, list[str]] = {
    "dining":          ["Chipotle", "Panera Bread", "McDonald's", "Starbucks",
                        "Local Diner", "Olive Garden", "Sushi Palace", "Pizza Hut"],
    "groceries":       ["Whole Foods", "Walmart Grocery", "Kroger", "Trader Joe's",
                        "Aldi", "Costco", "Target Grocery", "Safeway"],
    "subscriptions":   ["Netflix", "Spotify", "Hulu", "Adobe CC",
                        "Amazon Prime", "YouTube Premium", "Gym Membership", "iCloud"],
    "transportation":  ["Shell Gas", "BP Gas", "Uber", "Lyft",
                        "Metro Transit", "Parking Lot A", "ExxonMobil", "Enterprise"],
    "utilities":       ["City Water Dept", "Electric Company", "Gas Company",
                        "Internet Provider", "Waste Management"],
    "entertainment":   ["AMC Theaters", "Steam", "Ticketmaster", "Barnes & Noble",
                        "Bowling Alley", "Mini Golf", "Escape Room"],
    "shopping":        ["Amazon", "Target", "Macy's", "Best Buy",
                        "Home Depot", "Nordstrom", "TJ Maxx", "Wayfair"],
    "healthcare":      ["CVS Pharmacy", "Walgreens", "Doctor's Office", "Urgent Care",
                        "Dental Office", "Vision Center", "Lab Corp"],
}

BULK_BUY_MERCHANTS: dict[str, list[str]] = {
    "groceries":    ["Costco", "Walmart Grocery", "Aldi"],
    "subscriptions": ["Amazon Prime"],
    "shopping":     ["Amazon", "Costco"],
}

UNUSUAL_MERCHANTS: list[str] = [
    "Casino Royale", "Luxury Watch Shop", "Airline Ticket", "Hotel Hilton",
    "Foreign Exchange", "Jewelry Store", "Antique Dealer",
]


def _dow_multiplier(dow: int) -> float:
    return [0.85, 0.80, 0.90, 0.95, 1.10, 1.30, 1.20][dow]


def _month_seasonality(month: int) -> float:
    return {1: 0.80, 2: 0.78, 3: 0.90, 4: 0.95, 5: 1.00,
            6: 1.05, 7: 1.05, 8: 1.00, 9: 0.95,
            10: 1.00, 11: 1.15, 12: 1.35}.get(month, 1.0)


def _generate_user_transactions(
    user_id: int,
    start: datetime,
    end: datetime,
    rng: np.random.Generator,
    anomaly_prob: float = 0.03,
    bulk_buy_prob: float = 0.40,
) -> list[dict]:
    records: list[dict] = []
    user_scale = rng.lognormal(mean=0.0, sigma=0.35)

    bulk_cats = {
        cat for cat in BULK_BUY_MERCHANTS
        if rng.random() < bulk_buy_prob
    }

    for category in CATEGORIES:
        if category not in CATEGORY_SPEND_RANGES:
            # "other" is a catch-all; not generated synthetically.
            continue
        lo, hi = CATEGORY_SPEND_RANGES[category]
        daily_prob = CATEGORY_FREQUENCY[category] / 30.0
        is_sub = (category == "subscriptions")
        pool = MERCHANTS[category]
        fav = str(rng.choice(pool))

        cur = start
        billing_day = int(rng.integers(1, 28))

        while cur <= end:
            m_mult = _month_seasonality(cur.month)
            d_mult = _dow_multiplier(cur.weekday())

            if is_sub:
                if cur.day == billing_day:
                    amount = float(rng.uniform(lo, hi) * user_scale)
                    merch = str(rng.choice(pool))
                    records.append({
                        "user_id": user_id, "date": cur.date(),
                        "amount": round(amount, 2), "category": category,
                        "merchant": merch,
                        "description": f"{merch} monthly subscription",
                        "is_anomaly": False, "is_bulk_buy": False,
                    })
                cur += timedelta(days=1)
                continue

            if rng.random() < daily_prob * m_mult * d_mult:
                base = rng.uniform(lo, hi) * user_scale * m_mult
                amount = float(np.clip(
                    base + rng.normal(0, base * 0.15), lo * 0.5, hi * 3.0
                ))
                merch = fav if rng.random() < 0.35 else str(rng.choice(pool))
                is_bulk = False

                if category in bulk_cats and category in BULK_BUY_MERCHANTS:
                    if rng.random() < 0.55:
                        merch = str(rng.choice(BULK_BUY_MERCHANTS[category]))
                        amount = round(amount * rng.uniform(1.5, 3.5), 2)
                        is_bulk = True

                is_anomaly = False
                if rng.random() < anomaly_prob:
                    if rng.random() < 0.5:
                        amount *= rng.uniform(4.0, 10.0)
                    else:
                        merch = str(rng.choice(UNUSUAL_MERCHANTS))
                        amount = float(rng.uniform(200.0, 2000.0))
                    is_anomaly = True

                records.append({
                    "user_id": user_id, "date": cur.date(),
                    "amount": round(float(amount), 2), "category": category,
                    "merchant": merch,
                    "description": f"{merch} - {category}",
                    "is_anomaly": is_anomaly, "is_bulk_buy": is_bulk,
                })

            cur += timedelta(days=1)

    return records


def generate_synthetic_data(
    num_users: int = 200,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic transaction dataset.

    Returns
    -------
    pd.DataFrame with columns:
        user_id, date, amount, category, merchant, description,
        is_anomaly, is_bulk_buy
    """
    rng = np.random.default_rng(seed)
    ed = datetime.fromisoformat(end_date) if end_date else datetime.today()
    sd = datetime.fromisoformat(start_date) if start_date else ed - timedelta(days=548)

    all_records: list[dict] = []
    for uid in range(num_users):
        all_records.extend(_generate_user_transactions(uid, sd, ed, rng))

    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["user_id", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    df = generate_synthetic_data(num_users=50)
    print(f"Generated {len(df):,} transactions for {df['user_id'].nunique()} users")
    print(f"Anomaly rate:  {df['is_anomaly'].mean():.2%}")
    print(f"Bulk-buy rate: {df['is_bulk_buy'].mean():.2%}")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(df.head(5).to_string())
