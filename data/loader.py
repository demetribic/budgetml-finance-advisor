"""
loader.py — Dataset loader for the budgeting ML system.

Priority order (highest quality first):
  1. PersonaLedger (Capital One, HuggingFace) — 117M rows, synthetic but realistic
  2. MoneyVis (thevisgroup/MoneyVis on GitHub)  — 6,500 real anonymized UK bank transactions
  3. Synthetic fallback via synthetic.py

All loaders normalize output to a common schema:
    user_id     : int
    date        : datetime64[ns]
    amount      : float64   (positive = spend, negative = credit)
    category    : str       (one of CATEGORIES)
    merchant    : str
    description : str
    is_anomaly  : bool
    is_bulk_buy : bool
"""

from __future__ import annotations

import io
import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Canonical categories ──────────────────────────────────────────────────────

CATEGORIES = [
    "dining",
    "groceries",
    "subscriptions",
    "transportation",
    "utilities",
    "entertainment",
    "shopping",
    "healthcare",
    "other",
]

# ── PersonaLedger: merchant_type → category mapping ──────────────────────────
# Based on PersonaLedger's merchant_type values observed in the dataset.
# Covers the most common types; unmapped types fall through to "other".

_PERSONALEDGER_CATEGORY_MAP: dict[str, str] = {
    # dining
    "restaurant":           "dining",
    "fast_food":            "dining",
    "coffee_shop":          "dining",
    "cafe":                 "dining",
    "bar":                  "dining",
    "bakery":               "dining",
    "food_delivery":        "dining",
    "food":                 "dining",
    # groceries
    "grocery":              "groceries",
    "grocery_store":        "groceries",
    "supermarket":          "groceries",
    "farmer_market":        "groceries",
    "wholesale":            "groceries",
    # subscriptions
    "streaming":            "subscriptions",
    "subscription":         "subscriptions",
    "saas":                 "subscriptions",
    "digital_service":      "subscriptions",
    "membership":           "subscriptions",
    # transportation
    "gas_station":          "transportation",
    "fuel":                 "transportation",
    "rideshare":            "transportation",
    "taxi":                 "transportation",
    "public_transit":       "transportation",
    "parking":              "transportation",
    "car_rental":           "transportation",
    "airline":              "transportation",
    "toll":                 "transportation",
    "auto_repair":          "transportation",
    # utilities
    "utility":              "utilities",
    "utilities":            "utilities",
    "electric":             "utilities",
    "gas":                  "utilities",
    "water":                "utilities",
    "internet":             "utilities",
    "telecom":              "utilities",
    "phone":                "utilities",
    "waste":                "utilities",
    # entertainment
    "entertainment":        "entertainment",
    "movie_theater":        "entertainment",
    "theater":              "entertainment",
    "amusement_park":       "entertainment",
    "sports":               "entertainment",
    "gaming":               "entertainment",
    "music":                "entertainment",
    "event":                "entertainment",
    "concert":              "entertainment",
    # shopping
    "retail":               "shopping",
    "online_retail":        "shopping",
    "department_store":     "shopping",
    "clothing":             "shopping",
    "electronics":          "shopping",
    "home_goods":           "shopping",
    "furniture":            "shopping",
    "sporting_goods":       "shopping",
    "bookstore":            "shopping",
    "jewelry":              "shopping",
    "gift_shop":            "shopping",
    # healthcare
    "pharmacy":             "healthcare",
    "medical":              "healthcare",
    "hospital":             "healthcare",
    "dental":               "healthcare",
    "optician":             "healthcare",
    "fitness":              "healthcare",
    "gym":                  "healthcare",
    "health":               "healthcare",
    "doctor":               "healthcare",
    "clinic":               "healthcare",
}


def _map_personaledger_category(merchant_type: str) -> str:
    """Map a PersonaLedger merchant_type string to one of our canonical categories."""
    if not isinstance(merchant_type, str):
        return "other"
    key = merchant_type.lower().strip().replace(" ", "_")
    return _PERSONALEDGER_CATEGORY_MAP.get(key, "other")


# ── MoneyVis: transaction type → category mapping ────────────────────────────
# MoneyVis uses UK banking codes: SO, DEB, BP, CR, etc.

_MONEYVIS_TYPE_MAP: dict[str, str] = {
    "so":    "subscriptions",   # Standing Order (recurring fixed payment)
    "deb":   "shopping",        # Debit card payment (general retail)
    "bp":    "utilities",       # Bill Payment
    "cr":    "other",           # Credit / refund
    "dd":    "utilities",       # Direct Debit (usually bills/subscriptions)
    "trf":   "other",           # Transfer
    "atm":   "other",           # ATM withdrawal
    "chq":   "other",           # Cheque
    "int":   "other",           # Interest
    "fee":   "other",           # Fee
}

_MONEYVIS_DESC_KEYWORDS: dict[str, str] = {
    # dining
    "restaurant":    "dining",  "cafe":          "dining",
    "coffee":        "dining",  "pizza":         "dining",
    "burger":        "dining",  "takeaway":      "dining",
    "food":          "dining",  "eat":           "dining",
    # groceries
    "tesco":         "groceries", "sainsbury":   "groceries",
    "waitrose":      "groceries", "asda":        "groceries",
    "morrisons":     "groceries", "aldi":        "groceries",
    "lidl":          "groceries", "supermarket": "groceries",
    "grocery":       "groceries", "coop":        "groceries",
    # subscriptions
    "netflix":       "subscriptions", "spotify":  "subscriptions",
    "amazon prime":  "subscriptions", "youtube":  "subscriptions",
    "subscription":  "subscriptions", "membership": "subscriptions",
    "gym":           "subscriptions",
    # transportation
    "transport":     "transportation", "rail":    "transportation",
    "train":         "transportation", "bus":     "transportation",
    "uber":          "transportation", "fuel":    "transportation",
    "petrol":        "transportation", "parking": "transportation",
    "oyster":        "transportation",
    # utilities
    "electric":      "utilities", "gas":          "utilities",
    "water":         "utilities", "broadband":    "utilities",
    "bt ":           "utilities", "virgin media": "utilities",
    "sky ":          "utilities", "council tax":  "utilities",
    "insurance":     "utilities", "mortgage":     "utilities",
    "rent":          "utilities",
    # entertainment
    "cinema":        "entertainment", "odeon":    "entertainment",
    "vue":           "entertainment", "theatre":  "entertainment",
    "gaming":        "entertainment", "steam":    "entertainment",
    "amazon video":  "entertainment",
    # shopping
    "amazon":        "shopping", "ebay":         "shopping",
    "argos":         "shopping", "asos":         "shopping",
    "primark":       "shopping", "next":         "shopping",
    "marks":         "shopping", "debenhams":    "shopping",
    # healthcare
    "pharmacy":      "healthcare", "boots":       "healthcare",
    "nhs":           "healthcare", "dentist":     "healthcare",
    "doctor":        "healthcare", "medical":     "healthcare",
    "optician":      "healthcare",
}


def _map_moneyvis_category(tx_type: str, description: str) -> str:
    """Infer category from MoneyVis transaction type + description text."""
    desc_lower = (description or "").lower()
    for keyword, cat in _MONEYVIS_DESC_KEYWORDS.items():
        if keyword in desc_lower:
            return cat
    tx_code = (tx_type or "").lower().strip()
    return _MONEYVIS_TYPE_MAP.get(tx_code, "other")


# ── Shared normaliser ─────────────────────────────────────────────────────────

_COMMON_COLUMNS = [
    "user_id", "date", "amount", "category",
    "merchant", "description", "is_anomaly", "is_bulk_buy",
]


def _enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame has exactly the common columns with correct dtypes."""
    for col in _COMMON_COLUMNS:
        if col not in df.columns:
            if col in ("is_anomaly", "is_bulk_buy"):
                df[col] = False
            elif col == "user_id":
                df[col] = 0
            else:
                df[col] = None

    df["date"]       = pd.to_datetime(df["date"], errors="coerce")
    df["amount"]     = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["user_id"]    = df["user_id"].astype(int)
    df["is_anomaly"] = df["is_anomaly"].astype(bool)
    df["is_bulk_buy"] = df["is_bulk_buy"].astype(bool)
    df["category"]   = df["category"].fillna("other").astype(str)
    df["merchant"]   = df["merchant"].fillna("Unknown").astype(str)
    df["description"] = df["description"].fillna("").astype(str)

    df = df[_COMMON_COLUMNS].dropna(subset=["date"])
    df = df.sort_values(["user_id", "date"]).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1. PersonaLedger loader
# ─────────────────────────────────────────────────────────────────────────────

def load_personaledger(
    config: str = "default",
    split: str = "train",
    max_rows: Optional[int] = 500_000,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load PersonaLedger from HuggingFace and normalise to common schema.

    Parameters
    ----------
    config : str
        Dataset configuration. Options:
          - "default"                        — raw transactions (no labels)
          - "identity_theft_1months"         — 1-month identity theft task
          - "identity_theft_3months"         — 3-month identity theft task
          - "insolvency_prediction_1months"  — 1-month insolvency with labels
          - "insolvency_prediction_3months"  — 3-month insolvency with labels
    split : str
        "train" or "test"
    max_rows : int, optional
        Cap the number of rows loaded (dataset is 117M rows total).
        Set to None to load everything (slow).
    cache_dir : str, optional
        HuggingFace cache directory.

    Returns
    -------
    pd.DataFrame with common schema columns.

    PersonaLedger schema
    --------------------
    timestamp           datetime  2024-01-01 to 2027-11-30
    merchant_name       str
    merchant_type       str       maps to our category
    card_present_or_not str       "yes" / "no"
    amount              float64   negative = credit/refund
    seq_id              int64     sequence id (proxy for user_id in default config)
    """
    logger.info(
        "Loading PersonaLedger (config=%s, split=%s, max_rows=%s) …",
        config, split, max_rows,
    )

    # Read the specific parquet file directly via huggingface_hub to avoid the
    # datasets library generating ALL splits internally.  All test parquets have
    # an extra `is_fraud` column that is absent from train parquets; load_dataset
    # tries to cast every split to the declared features and fails on the schema
    # mismatch even when only the train split is requested.
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError:
        raise ImportError(
            "Install huggingface_hub: pip install huggingface_hub"
        )

    # "default" is not a real config in the repo — it was a virtual config that
    # load_dataset synthesised by combining all available parquets.  We replicate
    # that by reading each config's parquet and concatenating up to max_rows.
    _ALL_CONFIGS = [
        "identity_theft_3months",
        "identity_theft_1months",
        "insolvency_prediction_3months",
        "insolvency_prediction_1months",
    ]

    hf_base: dict = dict(repo_id="capitalone/PersonaLedger", repo_type="dataset")
    if cache_dir:
        hf_base["cache_dir"] = cache_dir

    if config == "default":
        frames = []
        rows_left = max_rows
        for cfg in _ALL_CONFIGS:
            path = hf_hub_download(**hf_base, filename=f"{cfg}/{split}.parquet")
            chunk = pd.read_parquet(path)
            if rows_left is not None:
                chunk = chunk.iloc[:rows_left]
            frames.append(chunk)
            if rows_left is not None:
                rows_left -= len(chunk)
                if rows_left <= 0:
                    break
        df = pd.concat(frames, ignore_index=True)
    else:
        parquet_path = hf_hub_download(**hf_base, filename=f"{config}/{split}.parquet")
        df = pd.read_parquet(parquet_path)
        if max_rows is not None and len(df) > max_rows:
            df = df.iloc[:max_rows].copy()

    # ── Map to common schema ──────────────────────────────────────────────────
    out = pd.DataFrame()

    # seq_id serves as user proxy in the default config;
    # for labelled configs a "user_id" column may appear
    if "user_id" in df.columns:
        out["user_id"] = df["user_id"].astype(int)
    else:
        # Group transactions by seq_id buckets to simulate ~1000 users
        out["user_id"] = (df["seq_id"] % 1000).astype(int)

    out["date"]        = pd.to_datetime(df["timestamp"], errors="coerce")
    out["amount"]      = df["amount"].astype(float)
    out["merchant"]    = df["merchant_name"].fillna("Unknown")
    out["category"]    = df["merchant_type"].apply(_map_personaledger_category)
    out["description"] = (
        df["merchant_name"].fillna("") + " - " + df["merchant_type"].fillna("")
    )

    # Identity theft labels → is_anomaly
    if "is_fraud" in df.columns:
        out["is_anomaly"] = df["is_fraud"].astype(bool)
    elif "label" in df.columns:
        out["is_anomaly"] = df["label"].astype(bool)
    else:
        out["is_anomaly"] = False

    out["is_bulk_buy"] = False  # PersonaLedger has no bulk-buy label

    logger.info("PersonaLedger loaded: %d rows, %d users", len(out), out["user_id"].nunique())
    return _enforce_schema(out)


# ─────────────────────────────────────────────────────────────────────────────
# 2. MoneyVis loader
# ─────────────────────────────────────────────────────────────────────────────

_MONEYVIS_RAW_URL = (
    "https://raw.githubusercontent.com/thevisgroup/MoneyVis/master/data.json"
)


def load_moneyvis(
    path: Optional[str] = None,
    url: str = _MONEYVIS_RAW_URL,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Load the MoneyVis open bank transaction dataset and normalise to common schema.

    Data can be provided as a local file path or fetched directly from GitHub.

    MoneyVis schema (from data.json)
    ---------------------------------
    The JSON structure is a list of transaction objects. Observed keys:
      date          str   "YYYY-MM-DD" or "DD/MM/YYYY"
      type          str   UK bank code: SO, DEB, BP, CR, DD, TRF, ATM …
      description   str   free-text merchant/payee name
      amount        float positive = credit in, negative = debit out
      balance       float running account balance after transaction

    Note: MoneyVis is a single-account dataset (one user, 7 years of data).

    Parameters
    ----------
    path : str, optional
        Local path to a JSON file (list of transaction dicts).
        If None, fetches from GitHub.
    url : str
        GitHub raw URL for data.json (default is the official repo).
    timeout : int
        HTTP timeout in seconds for the GitHub fetch.

    Returns
    -------
    pd.DataFrame with common schema columns.
    """
    # ── Load raw JSON ─────────────────────────────────────────────────────────
    if path is not None:
        logger.info("Loading MoneyVis from local file: %s", path)
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    else:
        logger.info("Fetching MoneyVis from GitHub: %s", url)
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        raw = resp.json()

    # raw may be a list of dicts or a dict with a transactions key
    if isinstance(raw, dict):
        # Try common wrapper keys first
        unwrapped = False
        for key in ("transactions", "data", "records"):
            if key in raw:
                raw = raw[key]
                unwrapped = True
                break

        if not unwrapped:
            # Keys may be dates (MoneyVis format: {date: [txn, ...]}).
            # Inject the key as "date" on each child record so we don't lose it.
            records: list[dict] = []
            for k, v in raw.items():
                if not isinstance(v, list):
                    continue
                try:
                    pd.to_datetime(k, dayfirst=True)   # validate it looks like a date
                    date_key = str(k)
                    for rec in v:
                        rec = dict(rec)           # shallow copy — don't mutate original
                        rec.setdefault("date", date_key)
                        records.append(rec)
                except (ValueError, TypeError):
                    records.extend(v)             # non-date key: flatten as-is
            raw = records

    df_raw = pd.DataFrame(raw)
    logger.info("MoneyVis raw rows: %d, columns: %s", len(df_raw), list(df_raw.columns))

    # ── Normalise column names (case-insensitive, handle variants) ───────────
    df_raw.columns = [c.lower().strip() for c in df_raw.columns]

    col_aliases = {
        "transaction_date": "date", "tx_date": "date",
        "transaction_type": "type", "tx_type": "type", "category_type": "type",
        "transaction_description": "description",
        "tx_description": "description", "payee": "description",
        "transaction_amount": "amount", "tx_amount": "amount",
        "debit":        "amount",        # some exports split debit/credit
        "debit_amount": "amount",        # MoneyVis actual column name
        "credit":       "credit_amount",
    }
    df_raw.rename(columns=col_aliases, inplace=True)

    # If separate debit/credit columns, combine into signed amount
    if "credit_amount" in df_raw.columns and "amount" in df_raw.columns:
        credit = pd.to_numeric(df_raw["credit_amount"], errors="coerce").fillna(0.0)
        debit  = pd.to_numeric(df_raw["amount"],        errors="coerce").fillna(0.0)
        # Debits are spends (negative), credits are income (positive)
        df_raw["amount"] = credit - debit.abs()

    # ── Parse dates ───────────────────────────────────────────────────────────
    if "date" not in df_raw.columns:
        raise ValueError(
            "MoneyVis data has no recognisable date column. "
            f"Found columns: {list(df_raw.columns)}"
        )

    df_raw["date"] = pd.to_datetime(
        df_raw["date"], dayfirst=True, errors="coerce"   # UK dates: DD/MM/YYYY
    )

    # ── Build output ──────────────────────────────────────────────────────────
    out = pd.DataFrame()
    out["user_id"]     = 0  # single account; treat as one user
    out["date"]        = df_raw["date"]
    out["amount"]      = pd.to_numeric(df_raw.get("amount"), errors="coerce").fillna(0.0)
    out["merchant"]    = df_raw.get("description", pd.Series(["Unknown"] * len(df_raw)))
    out["description"] = df_raw.get("description", pd.Series([""] * len(df_raw)))

    tx_type = df_raw.get("type", pd.Series([""] * len(df_raw)))
    out["category"] = [
        _map_moneyvis_category(t, d)
        for t, d in zip(tx_type.fillna(""), out["description"].fillna(""))
    ]

    out["is_anomaly"]  = False
    out["is_bulk_buy"] = False

    # ── Filter: only keep spending transactions (negative amounts) ────────────
    # MoneyVis convention: negative = money out of account
    # We flip sign so amounts are positive spend values (matching PersonaLedger)
    spending_mask = out["amount"] < 0
    out = out[spending_mask].copy()
    out["amount"] = out["amount"].abs()

    logger.info(
        "MoneyVis loaded: %d spend transactions, date range %s → %s",
        len(out),
        out["date"].min().date() if not out.empty else "N/A",
        out["date"].max().date() if not out.empty else "N/A",
    )
    return _enforce_schema(out)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Unified load function
# ─────────────────────────────────────────────────────────────────────────────

def load_from_db(db_path: Optional[str] = None, min_rows: int = 50) -> pd.DataFrame:
    """
    Load all transactions from the local SQLite database (api/db.py schema).

    Returns a DataFrame in the canonical training schema.  Raises ValueError
    if the DB has fewer than `min_rows` rows (not enough to train on).
    """
    from pathlib import Path as _Path
    default_db = _Path(__file__).parent.parent / "data" / "budgetml.db"
    path = _Path(db_path) if db_path else default_db
    if not path.exists():
        raise ValueError(f"DB not found at {path}")

    import sqlite3
    con = sqlite3.connect(str(path))
    df = pd.read_sql_query(
        "SELECT user_id, date, amount, category, merchant, description FROM transactions",
        con,
    )
    con.close()

    if len(df) < min_rows:
        raise ValueError(
            f"DB only has {len(df)} rows — need at least {min_rows} to train. "
            "Use /analyze more to accumulate transactions, or use --source auto."
        )

    df["date"]        = pd.to_datetime(df["date"])
    df["amount"]      = df["amount"].astype(float)
    df["is_anomaly"]  = False
    df["is_bulk_buy"] = False
    # Ensure categories are valid
    valid = set(CATEGORIES)
    df["category"] = df["category"].where(df["category"].isin(valid), "other")
    df["merchant"]  = df["merchant"].fillna("<UNK>")
    df["description"] = df["description"].fillna("")
    df["user_id"]   = df["user_id"].astype(str)

    logger.info("DB source: %d rows, %d users", len(df), df["user_id"].nunique())
    return df


def load_transactions(
    source: str = "auto",
    personaledger_config: str = "identity_theft_3months",
    personaledger_split: str = "test",
    personaledger_max_rows: int = 500_000,
    moneyvis_path: Optional[str] = None,
    db_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load transaction data from the best available source.

    Parameters
    ----------
    source : str
        "auto"          — try PersonaLedger → MoneyVis → synthetic (in order)
        "personaledger" — PersonaLedger only (raises if unavailable)
        "moneyvis"      — MoneyVis only
        "db"            — SQLite DB (real user transactions from /analyze calls)
        "db+auto"       — DB transactions merged with PersonaLedger/MoneyVis/synthetic
        "synthetic"     — always use synthetic data
    personaledger_config : str
        PersonaLedger config (see load_personaledger).
        Default "identity_theft_3months" gives labelled anomaly data.
        Use "default" to combine all available configs (matches old behaviour).
    personaledger_split : str
        "train" or "test".
    personaledger_max_rows : int
        Row cap for PersonaLedger (default 500k keeps memory manageable).
    moneyvis_path : str, optional
        Local path to MoneyVis data.json (skips GitHub fetch if provided).
    cache_dir : str, optional
        HuggingFace cache directory.
    verbose : bool
        Log progress to stdout.

    Returns
    -------
    pd.DataFrame with columns: user_id, date, amount, category,
                                merchant, description, is_anomaly, is_bulk_buy
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    if source == "synthetic":
        return _load_synthetic()

    if source == "db":
        return load_from_db(db_path=db_path)

    if source == "db+auto":
        # Merge real DB transactions with a background dataset for volume.
        # DB rows take precedence — they represent the real user's spending.
        frames: list[pd.DataFrame] = []
        try:
            frames.append(load_from_db(db_path=db_path))
        except ValueError as e:
            logger.warning("DB load skipped: %s", e)
        bg = load_transactions(
            source="auto",
            personaledger_config=personaledger_config,
            personaledger_split=personaledger_split,
            personaledger_max_rows=personaledger_max_rows,
            moneyvis_path=moneyvis_path,
            cache_dir=cache_dir,
            verbose=False,
        )
        frames.append(bg)
        if not frames:
            return _load_synthetic()
        merged = pd.concat(frames, ignore_index=True)
        logger.info("db+auto combined: %d rows", len(merged))
        return merged

    if source == "personaledger":
        return load_personaledger(
            config=personaledger_config,
            split=personaledger_split,
            max_rows=personaledger_max_rows,
            cache_dir=cache_dir,
        )

    if source == "moneyvis":
        return load_moneyvis(path=moneyvis_path)

    # source == "auto": try in order
    errors: list[str] = []

    try:
        return load_personaledger(
            config=personaledger_config,
            split=personaledger_split,
            max_rows=personaledger_max_rows,
            cache_dir=cache_dir,
        )
    except Exception as exc:
        errors.append(f"PersonaLedger failed: {exc}")
        logger.warning(errors[-1])

    try:
        return load_moneyvis(path=moneyvis_path)
    except Exception as exc:
        errors.append(f"MoneyVis failed: {exc}")
        logger.warning(errors[-1])

    logger.warning("All real datasets failed — falling back to synthetic data.\n%s", "\n".join(errors))
    return _load_synthetic()


def _load_synthetic() -> pd.DataFrame:
    """Import and run the synthetic generator as a last-resort fallback."""
    from data.synthetic import generate_synthetic_data  # type: ignore
    logger.info("Generating synthetic transaction data …")
    df = generate_synthetic_data(num_users=200)
    logger.info("Synthetic data: %d rows", len(df))
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and preview transaction data")
    parser.add_argument(
        "--source",
        choices=["auto", "personaledger", "moneyvis", "synthetic", "db", "db+auto"],
        default="auto",
        help="Data source (default: auto)",
    )
    parser.add_argument(
        "--config",
        default="default",
        help="PersonaLedger config name",
    )
    parser.add_argument(
        "--max-rows", type=int, default=100_000,
        help="Max rows to load from PersonaLedger",
    )
    parser.add_argument(
        "--moneyvis-path", default=None,
        help="Local path to MoneyVis data.json",
    )
    args = parser.parse_args()

    df = load_transactions(
        source=args.source,
        personaledger_config=args.config,
        personaledger_max_rows=args.max_rows,
        moneyvis_path=args.moneyvis_path,
    )

    print(f"\nLoaded {len(df):,} transactions")
    print(f"Users  : {df['user_id'].nunique()}")
    print(f"Range  : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Categories:\n{df['category'].value_counts().to_string()}")
    print(f"\nAnomaly rate : {df['is_anomaly'].mean():.2%}")
    print(f"Bulk-buy rate: {df['is_bulk_buy'].mean():.2%}")
    print(f"\nSample:\n{df.head(10).to_string()}")
