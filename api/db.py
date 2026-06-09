"""
api/db.py — SQLite persistence layer for BudgetML user data.

Schema
------
  users             : user_id, created_at, settings_json
  transactions      : id, user_id, date, amount, category, merchant, description, inserted_at
  suggestions       : id, user_id, generated_at, type, category, message, confidence,
                      amount_impact, details_json
  user_anomaly_history : user_id, date, anomaly_score
    auto_category_feedback : id, description, category, confidence, source, created_at
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "budgetml.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _conn():
    """Context manager: open connection, yield cursor, commit, close."""
    con = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    """Create tables if they don't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _conn() as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_id      TEXT PRIMARY KEY,
                created_at   TEXT NOT NULL,
                settings_json TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS transactions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      TEXT NOT NULL,
                date         TEXT NOT NULL,
                amount       REAL NOT NULL,
                category     TEXT DEFAULT 'other',
                merchant     TEXT DEFAULT 'Unknown',
                description  TEXT DEFAULT '',
                inserted_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_txn_user_date
                ON transactions(user_id, date);

            CREATE TABLE IF NOT EXISTS suggestions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      TEXT NOT NULL,
                generated_at TEXT NOT NULL,
                type         TEXT NOT NULL,
                category     TEXT NOT NULL,
                message      TEXT NOT NULL,
                confidence   REAL NOT NULL,
                amount_impact REAL NOT NULL DEFAULT 0.0,
                details_json TEXT DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_sug_user ON suggestions(user_id, generated_at);

            CREATE TABLE IF NOT EXISTS user_anomaly_history (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      TEXT NOT NULL,
                date         TEXT NOT NULL,
                anomaly_score REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_anom_user ON user_anomaly_history(user_id);

            CREATE TABLE IF NOT EXISTS auto_category_feedback (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                description  TEXT NOT NULL,
                category     TEXT NOT NULL,
                confidence   REAL NOT NULL,
                source       TEXT NOT NULL DEFAULT 'web_fallback',
                created_at   TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_autocat_feedback_created
                ON auto_category_feedback(created_at);

            CREATE TABLE IF NOT EXISTS user_categories (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      TEXT NOT NULL,
                name         TEXT NOT NULL,
                description  TEXT DEFAULT '',
                color        TEXT DEFAULT '#4a6580',
                icon         TEXT DEFAULT '📦',
                created_at   TEXT NOT NULL,
                UNIQUE(user_id, name)
            );
            CREATE INDEX IF NOT EXISTS idx_user_cat_user
                ON user_categories(user_id);

            CREATE TABLE IF NOT EXISTS user_category_examples (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      TEXT NOT NULL,
                category     TEXT NOT NULL,
                merchant     TEXT NOT NULL,
                description  TEXT DEFAULT '',
                created_at   TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_user_cat_ex_user
                ON user_category_examples(user_id, category);

            CREATE TABLE IF NOT EXISTS user_category_preferences (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       TEXT NOT NULL,
                merchant_norm TEXT NOT NULL,
                category      TEXT NOT NULL,
                confirmed_at  TEXT NOT NULL,
                UNIQUE(user_id, merchant_norm)
            );
            CREATE INDEX IF NOT EXISTS idx_user_cat_pref_user
                ON user_category_preferences(user_id);

            CREATE TABLE IF NOT EXISTS user_vae_snapshots (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id        TEXT NOT NULL,
                snapshot_month TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                archetype      TEXT NOT NULL DEFAULT 'Unknown',
                captured_at    TEXT NOT NULL,
                UNIQUE(user_id, snapshot_month)
            );
            CREATE INDEX IF NOT EXISTS idx_vae_snap_user
                ON user_vae_snapshots(user_id, snapshot_month);
        """)


def list_users() -> list[dict]:
    """Return all users with transaction counts and date ranges."""
    with _conn() as con:
        rows = con.execute("""
            SELECT u.user_id, u.created_at,
                   COUNT(t.id)  AS tx_count,
                   MIN(t.date)  AS first_tx,
                   MAX(t.date)  AS last_tx
            FROM users u
            LEFT JOIN transactions t ON t.user_id = u.user_id
            GROUP BY u.user_id
            ORDER BY u.created_at DESC
        """).fetchall()
    return [dict(r) for r in rows]


def upsert_user(user_id: str) -> None:
    """Create user record if it doesn't exist."""
    with _conn() as con:
        con.execute(
            "INSERT OR IGNORE INTO users(user_id, created_at) VALUES (?, ?)",
            (user_id, _now()),
        )


def upsert_transactions(user_id: str, transactions: list[dict]) -> None:
    """
    Insert transactions for a user (deduplication by date+amount+merchant).
    New transactions take precedence: we insert on conflict-ignore by composite
    unique key (user_id, date, amount, merchant).
    """
    upsert_user(user_id)
    now = _now()
    with _conn() as con:
        # Create composite unique index if not exists (can't be in CREATE TABLE for ALTER compat)
        try:
            con.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_txn_dedup
                    ON transactions(user_id, date, amount, merchant)
            """)
        except sqlite3.OperationalError:
            pass

        con.executemany(
            """INSERT OR IGNORE INTO transactions
               (user_id, date, amount, category, merchant, description, inserted_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    user_id,
                    t.get("date", ""),
                    float(t.get("amount", 0)),
                    t.get("category", "other"),
                    t.get("merchant", "Unknown"),
                    t.get("description", ""),
                    now,
                )
                for t in transactions
            ],
        )


def update_transaction_categories(user_id: str, updates: list[dict]) -> int:
    """
    Update categories for existing user transactions.

    Rows are matched by the same dedup identity used on insert:
    (user_id, date, amount, merchant).
    Returns the number of updated rows.
    """
    if not updates:
        return 0

    with _conn() as con:
        before = con.total_changes
        con.executemany(
            """UPDATE transactions
               SET category = ?
               WHERE user_id = ?
                 AND date = ?
                 AND merchant = ?
                 AND ABS(amount - ?) < 1e-6""",
            [
                (
                    str(u.get("category", "other")),
                    user_id,
                    str(u.get("date", "")),
                    str(u.get("merchant", "Unknown")),
                    float(u.get("amount", 0.0)),
                )
                for u in updates
            ],
        )
        return con.total_changes - before


def get_user_transactions(user_id: str, limit_days: int = 180) -> list[dict]:
    """Return the last `limit_days` days of transactions for a user."""
    with _conn() as con:
        rows = con.execute(
            """SELECT date, amount, category, merchant, description
               FROM transactions
               WHERE user_id = ?
                 AND date >= date('now', ?)
               ORDER BY date ASC""",
            (user_id, f"-{limit_days} days"),
        ).fetchall()
    return [dict(r) for r in rows]


def save_suggestions(user_id: str, suggestions: list[dict]) -> None:
    """Persist a list of suggestions for a user."""
    upsert_user(user_id)
    now = _now()
    with _conn() as con:
        con.executemany(
            """INSERT INTO suggestions
               (user_id, generated_at, type, category, message, confidence, amount_impact, details_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    user_id,
                    now,
                    s.get("type", ""),
                    s.get("category", ""),
                    s.get("message", ""),
                    float(s.get("confidence", 0)),
                    float(s.get("amount_impact", 0)),
                    json.dumps(s.get("details", {})),
                )
                for s in suggestions
            ],
        )


def get_user_suggestions(user_id: str, limit: int = 30) -> list[dict]:
    """Return the last `limit` suggestions for a user."""
    with _conn() as con:
        rows = con.execute(
            """SELECT generated_at, type, category, message, confidence, amount_impact, details_json
               FROM suggestions
               WHERE user_id = ?
               ORDER BY generated_at DESC
               LIMIT ?""",
            (user_id, limit),
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["details"] = json.loads(d.pop("details_json", "{}"))
        result.append(d)
    return result


def get_user_anomaly_history(user_id: str) -> list[float]:
    """Return all stored anomaly scores for a user (for per-user z-score)."""
    with _conn() as con:
        rows = con.execute(
            "SELECT anomaly_score FROM user_anomaly_history WHERE user_id = ? ORDER BY date ASC",
            (user_id,),
        ).fetchall()
    return [float(r["anomaly_score"]) for r in rows]


def append_anomaly_score(user_id: str, score: float) -> None:
    """Append a new anomaly score observation for a user."""
    upsert_user(user_id)
    with _conn() as con:
        con.execute(
            "INSERT INTO user_anomaly_history(user_id, date, anomaly_score) VALUES (?, ?, ?)",
            (user_id, _now(), float(score)),
        )


def save_vae_snapshot(
    user_id:   str,
    month:     str,          # "YYYY-MM"
    embedding: list[float],
    archetype: str,
) -> None:
    """Upsert a monthly VAE embedding snapshot (overwrites if same month)."""
    upsert_user(user_id)
    with _conn() as con:
        con.execute(
            """INSERT INTO user_vae_snapshots
               (user_id, snapshot_month, embedding_json, archetype, captured_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(user_id, snapshot_month) DO UPDATE SET
                 embedding_json = excluded.embedding_json,
                 archetype      = excluded.archetype,
                 captured_at    = excluded.captured_at""",
            (user_id, month, json.dumps(embedding), archetype, _now()),
        )


def get_vae_snapshots(user_id: str) -> list[dict]:
    """Return all monthly VAE snapshots for a user, oldest first."""
    with _conn() as con:
        rows = con.execute(
            """SELECT snapshot_month, embedding_json, archetype, captured_at
               FROM user_vae_snapshots
               WHERE user_id = ?
               ORDER BY snapshot_month ASC""",
            (user_id,),
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["embedding"] = json.loads(d.pop("embedding_json"))
        result.append(d)
    return result


def clear_user_transactions(user_id: str) -> int:
    """Delete all transactions for a user without removing the user record. Returns row count."""
    with _conn() as con:
        cur = con.execute("DELETE FROM transactions WHERE user_id = ?", (user_id,))
        return cur.rowcount


def delete_user(user_id: str) -> None:
    """Delete all data for a user (GDPR compliance)."""
    with _conn() as con:
        for table in (
            "transactions", "suggestions", "user_anomaly_history",
            "user_categories", "user_category_examples",
            "user_category_preferences", "user_vae_snapshots", "users",
        ):
            con.execute(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))


def save_auto_category_feedback(items: list[dict]) -> None:
    """Persist weak labels produced by category fallback logic (e.g., web lookup)."""
    if not items:
        return
    now = _now()
    with _conn() as con:
        con.executemany(
            """INSERT INTO auto_category_feedback
               (description, category, confidence, source, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            [
                (
                    str(i.get("description", "")).strip(),
                    str(i.get("category", "other")),
                    float(i.get("confidence", 0.0)),
                    str(i.get("source", "web_fallback")),
                    now,
                )
                for i in items
                if str(i.get("description", "")).strip() and str(i.get("category", "other"))
            ],
        )


def get_auto_category_feedback(limit: int = 5000) -> list[dict]:
    """Return recent fallback-derived labels for optional classifier fine-tuning."""
    with _conn() as con:
        rows = con.execute(
            """SELECT description, category, confidence, source, created_at
               FROM auto_category_feedback
               ORDER BY created_at DESC
               LIMIT ?""",
            (int(limit),),
        ).fetchall()
    return [dict(r) for r in rows]


# ── Custom user categories ────────────────────────────────────────────────────

def get_user_categories(user_id: str) -> list[dict]:
    """Return all custom categories defined by this user."""
    with _conn() as con:
        rows = con.execute(
            "SELECT name, description, color, icon, created_at FROM user_categories "
            "WHERE user_id = ? ORDER BY name",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def upsert_user_category(
    user_id: str,
    name: str,
    description: str = "",
    color: str = "#4a6580",
    icon: str = "📦",
) -> None:
    """Create or update a custom category for a user."""
    upsert_user(user_id)
    with _conn() as con:
        con.execute(
            """INSERT INTO user_categories(user_id, name, description, color, icon, created_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(user_id, name) DO UPDATE SET
                 description = excluded.description,
                 color       = excluded.color,
                 icon        = excluded.icon""",
            (user_id, name.strip().lower(), description, color, icon, _now()),
        )


def delete_user_category(user_id: str, name: str) -> None:
    """Delete a custom category and its examples."""
    with _conn() as con:
        con.execute(
            "DELETE FROM user_categories WHERE user_id = ? AND name = ?",
            (user_id, name.strip().lower()),
        )
        con.execute(
            "DELETE FROM user_category_examples WHERE user_id = ? AND category = ?",
            (user_id, name.strip().lower()),
        )


def add_category_example(
    user_id: str,
    category: str,
    merchant: str,
    description: str = "",
) -> None:
    """Add a labeled example merchant/description to a custom (or built-in) category."""
    upsert_user(user_id)
    with _conn() as con:
        con.execute(
            """INSERT INTO user_category_examples(user_id, category, merchant, description, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (user_id, category.strip().lower(), merchant.strip(), description, _now()),
        )


def get_category_examples(user_id: str, category: str | None = None) -> list[dict]:
    """Return labeled examples for this user, optionally filtered by category."""
    with _conn() as con:
        if category:
            rows = con.execute(
                "SELECT category, merchant, description, created_at FROM user_category_examples "
                "WHERE user_id = ? AND category = ? ORDER BY created_at DESC",
                (user_id, category.strip().lower()),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT category, merchant, description, created_at FROM user_category_examples "
                "WHERE user_id = ? ORDER BY category, created_at DESC",
                (user_id,),
            ).fetchall()
    return [dict(r) for r in rows]


# ── Category preferences (user-confirmed corrections) ─────────────────────────

def save_category_preference(user_id: str, merchant: str, category: str) -> None:
    """Upsert a user's confirmed merchant→category mapping (last write wins)."""
    upsert_user(user_id)
    norm = merchant.strip().lower()
    if not norm:
        return
    with _conn() as con:
        con.execute(
            """INSERT INTO user_category_preferences(user_id, merchant_norm, category, confirmed_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(user_id, merchant_norm) DO UPDATE SET
                 category     = excluded.category,
                 confirmed_at = excluded.confirmed_at""",
            (user_id, norm, category.strip().lower(), _now()),
        )


def get_category_preferences(user_id: str) -> dict[str, str]:
    """Return {merchant_norm: category} for all confirmed preferences of a user."""
    with _conn() as con:
        rows = con.execute(
            "SELECT merchant_norm, category FROM user_category_preferences WHERE user_id = ?",
            (user_id,),
        ).fetchall()
    return {r["merchant_norm"]: r["category"] for r in rows}
