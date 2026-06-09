"""
scripts/seed_test_user.py — Seed a realistic test user for User VAE analysis.

Creates user "vae_test_user" with 6 months of transactions that exercise all
spending categories with realistic amounts, weekly patterns, and merchant variety.

Usage:
    python scripts/seed_test_user.py
    python scripts/seed_test_user.py --user-id my_custom_id --months 3
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import date, timedelta
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.db import init_db, upsert_transactions, get_user_transactions

# ── Spending patterns per category ────────────────────────────────────────────
# Each entry: (merchant, (amount_low, amount_high), weekly_frequency)
PATTERNS: dict[str, list[tuple[str, tuple[float, float], float]]] = {
    "groceries": [
        ("Whole Foods",   (55, 130), 1.0),
        ("Trader Joe's",  (40, 90),  0.7),
        ("Costco",        (80, 200), 0.3),
        ("ShopRite",      (35, 75),  0.5),
    ],
    "dining": [
        ("Chipotle",      (10, 18),  2.0),
        ("Starbucks",     (5, 12),   3.5),
        ("McDonald's",    (7, 14),   1.0),
        ("Local Sushi",   (35, 65),  0.4),
        ("Panera Bread",  (12, 22),  1.2),
        ("Pizza Palace",  (18, 40),  0.5),
    ],
    "transportation": [
        ("Shell",         (40, 70),  0.8),
        ("NJ Transit",    (6, 12),   2.5),
        ("Uber",          (12, 35),  1.5),
        ("EZPass",        (5, 25),   0.6),
        ("Parking Meter", (3, 10),   1.0),
    ],
    "utilities": [
        ("PSE&G",         (80, 160), 0.25),   # monthly
        ("Verizon",       (65, 95),  0.25),
        ("Comcast",       (75, 110), 0.25),
        ("Water Dept",    (30, 55),  0.12),   # bi-monthly
    ],
    "entertainment": [
        ("Netflix",       (15, 23),  0.25),
        ("Spotify",       (10, 16),  0.25),
        ("AMC Theaters",  (14, 28),  0.4),
        ("Steam",         (10, 60),  0.3),
        ("Bowling Alley", (20, 45),  0.3),
    ],
    "subscriptions": [
        ("Amazon Prime",  (14, 14),  0.25),
        ("New York Times",(17, 17),  0.25),
        ("Gym Membership",(40, 55),  0.25),
        ("iCloud",        (3, 10),   0.25),
    ],
    "shopping": [
        ("Amazon",        (15, 120), 1.5),
        ("Target",        (25, 95),  0.8),
        ("Best Buy",      (30, 280), 0.2),
        ("IKEA",          (40, 350), 0.1),
        ("H&M",           (20, 80),  0.3),
    ],
    "health": [
        ("CVS Pharmacy",  (12, 55),  0.6),
        ("Walgreens",     (8, 45),   0.4),
        ("Urgent Care",   (25, 150), 0.08),
        ("Gym Co-Pay",    (20, 50),  0.15),
    ],
    "other": [
        ("ATM Withdrawal",(40, 200), 0.4),
        ("Hardware Store",(15, 80),  0.3),
        ("Post Office",   (5, 30),   0.2),
    ],
}


def _generate_transactions(
    start: date,
    end:   date,
    rng:   random.Random,
) -> list[dict]:
    txns: list[dict] = []
    current = start

    while current <= end:
        dow = current.weekday()  # 0=Mon, 6=Sun
        # Weekend multiplier: people shop/dine more on weekends
        weekend = 1.4 if dow >= 5 else 1.0

        for category, merchants in PATTERNS.items():
            for merchant, (lo, hi), weekly_freq in merchants:
                # Daily probability = weekly_freq / 7, scaled by weekend
                prob = (weekly_freq / 7) * weekend
                if rng.random() < prob:
                    amount = round(rng.uniform(lo, hi), 2)
                    # Occasional 20% spike (unexpected large purchase)
                    if rng.random() < 0.04:
                        amount = round(amount * rng.uniform(1.5, 2.8), 2)
                    txns.append({
                        "date":        current.isoformat(),
                        "amount":      amount,
                        "category":    category,
                        "merchant":    merchant,
                        "description": f"{merchant} purchase",
                    })

        current += timedelta(days=1)

    return txns


def seed(user_id: str, months: int, seed_val: int) -> None:
    init_db()
    rng   = random.Random(seed_val)
    end   = date.today()
    start = date(end.year, end.month, 1) - timedelta(days=30 * (months - 1))

    print(f"Generating {months}-month history for user '{user_id}' ({start} → {end})…")
    txns = _generate_transactions(start, end, rng)
    upsert_transactions(user_id, txns)

    stored = get_user_transactions(user_id, limit_days=months * 31)
    print(f"Done. Inserted {len(txns)} raw rows → {len(stored)} unique in DB.")
    print(f"\nTo fetch the VAE embedding (once model is trained):")
    print(f"  curl http://localhost:8000/users/{user_id}/vae-embedding")


def main():
    parser = argparse.ArgumentParser(description="Seed a test user for VAE analysis.")
    parser.add_argument("--user-id", default="vae_test_user",
                        help="User ID to create (default: vae_test_user)")
    parser.add_argument("--months", type=int, default=6,
                        help="Months of history to generate (default: 6)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()
    seed(args.user_id, args.months, args.seed)


if __name__ == "__main__":
    main()
