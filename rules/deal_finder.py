"""
rules/deal_finder.py — Web deal search driven by analysis output.

Generates targeted DuckDuckGo queries based on what the analysis found
(bulk-buy flags, impulse spending, subscription traps, forecast overruns,
price intelligence hits), fetches real results, and caches them in SQLite
for 24 hours so repeated /analyze calls don't hammer the search API.

Requires: duckduckgo-search (pip install duckduckgo-search)
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_QUERY_NOISE_RE = re.compile(
    r"\b(?:VISA|MC|MASTERCARD|DEBIT|CREDIT|DDA|PUR|POS|ACH|AUTH|PAYMENT|"
    r"TRANSFER|PURCHASE|WITHDRAWAL|XFER|TRNSFR|PENDING|RECURRING|AP|DBT)\b",
    re.IGNORECASE,
)


def _is_noisy_merchant_name(name: str) -> bool:
    """Return True when a merchant string is raw bank statement noise, not a searchable brand."""
    if not name or name.lower() in ("unknown", "your usual store", ""):
        return True
    digit_ratio = sum(c.isdigit() for c in name) / max(len(name), 1)
    if digit_ratio > 0.30:
        return True
    return bool(_QUERY_NOISE_RE.search(name))

_DB_PATH       = Path(__file__).parent.parent / "data" / "budgetml.db"
_CACHE_TTL_H   = 24
_MAX_RESULTS   = 3    # results per query
_MAX_QUERIES   = 5    # cap searches per /analyze call
_OLLAMA_HOST   = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
_OLLAMA_MODEL  = os.environ.get("OLLAMA_MODEL", "llama3")


# ── SQLite deal cache ─────────────────────────────────────────────────────────

def _ensure_cache_table() -> None:
    if not _DB_PATH.exists():
        return
    try:
        con = sqlite3.connect(str(_DB_PATH))
        con.execute("""
            CREATE TABLE IF NOT EXISTS deal_cache (
                query        TEXT PRIMARY KEY,
                results_json TEXT NOT NULL,
                fetched_at   TEXT NOT NULL
            )
        """)
        con.commit()
        con.close()
    except Exception:
        pass


def _cache_get(query: str) -> list[dict] | None:
    if not _DB_PATH.exists():
        return None
    try:
        con = sqlite3.connect(str(_DB_PATH))
        row = con.execute(
            "SELECT results_json, fetched_at FROM deal_cache WHERE query = ?",
            (query,),
        ).fetchone()
        con.close()
        if row is None:
            return None
        fetched   = datetime.fromisoformat(row[1][:19])
        age_hours = (datetime.now(timezone.utc).replace(tzinfo=None) - fetched).total_seconds() / 3600
        return json.loads(row[0]) if age_hours <= _CACHE_TTL_H else None
    except Exception:
        return None


def _cache_set(query: str, results: list[dict]) -> None:
    if not _DB_PATH.exists():
        return
    try:
        _ensure_cache_table()
        con = sqlite3.connect(str(_DB_PATH))
        con.execute(
            "INSERT OR REPLACE INTO deal_cache(query, results_json, fetched_at) VALUES (?, ?, ?)",
            (query, json.dumps(results), datetime.now(timezone.utc).replace(tzinfo=None).isoformat()),
        )
        con.commit()
        con.close()
    except Exception:
        pass


# ── DuckDuckGo search ─────────────────────────────────────────────────────────

def _brave_search(query: str, api_key: str) -> list[dict]:
    """Search via Brave Search API (2 000 free queries/month)."""
    import requests
    resp = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": _MAX_RESULTS},
        headers={"Accept": "application/json", "X-Subscription-Token": api_key},
        timeout=10,
    )
    resp.raise_for_status()
    web = resp.json().get("web", {}).get("results", [])
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("description", "")}
        for r in web
    ]


def _ddg_search_raw(query: str) -> list[dict]:
    """Scrape DuckDuckGo HTML search — no API key, no VQD token."""
    import requests
    from bs4 import BeautifulSoup

    resp = requests.post(
        "https://html.duckduckgo.com/html",
        data={"q": query, "b": ""},
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"},
        timeout=10,
    )
    resp.raise_for_status()
    soup    = BeautifulSoup(resp.text, "html.parser")
    anchors  = soup.select(".result__title a")[:_MAX_RESULTS + 5]   # grab extra to account for filtered ads
    snippets = soup.select(".result__snippet")

    from urllib.parse import urlparse, parse_qs
    results = []
    for i, a in enumerate(anchors):
        title = a.get_text(strip=True)
        url   = a.get("href", "")
        if "//duckduckgo.com/l/" in url:
            qs  = parse_qs(urlparse(url).query)
            url = qs.get("uddg", [url])[0]
        # Skip ads (duckduckgo.com URLs) and empty/useless results
        if "duckduckgo.com" in url or not title or title.lower() in ("more info", ""):
            continue
        snippet = snippets[i].get_text(strip=True) if i < len(snippets) else ""
        results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= _MAX_RESULTS:
            break
    return results


def _clean(results: list[dict]) -> list[dict]:
    """Remove ad/internal URLs and empty titles."""
    return [
        r for r in results
        if r.get("url") and r.get("title")
        and "duckduckgo.com" not in r["url"]
        and r["title"].lower() not in ("more info", "")
    ]


def _search(query: str) -> list[dict]:
    """
    Return [{title, url, snippet}], served from cache when fresh.

    Backend priority:
      1. Brave Search API  — if BRAVE_API_KEY env var is set (recommended on HPC/servers)
      2. DuckDuckGo HTML   — free, no API key required
    """
    cached = _cache_get(query)
    if cached is not None:
        return _clean(cached)

    results: list[dict] = []
    try:
        brave_key = os.environ.get("BRAVE_API_KEY", "")
        if brave_key:
            results = _brave_search(query, brave_key)
        else:
            results = _ddg_search_raw(query)
    except Exception:
        results = []

    if results:   # never cache empty — could be a transient rate-limit
        _cache_set(query, results)
    return _clean(results)


# ── LLM query rewriter ────────────────────────────────────────────────────────

def _llm_available() -> bool:
    """Return True if any local LLM backend is ready (Ollama or llama-cpp-python)."""
    # Try Ollama first (fast probe)
    try:
        import urllib.request as _ur
        with _ur.urlopen(f"{_OLLAMA_HOST}/api/tags", timeout=2):
            return True
    except Exception:
        pass
    # Fall back to in-process llama-cpp-python
    try:
        from models.intelligence import local_llm  # noqa: PLC0415
        return local_llm.is_available()
    except Exception:
        return False


def _rewrite_query(fallback: str, context: dict, *, ollama_active: bool = False) -> str:
    """
    Use a local LLM to generate a more targeted search query than the template.

    Tries Ollama (if ollama_active=True), then llama-cpp-python.  Falls back to
    the template query on any error or if no LLM is available.

    Pass ollama_active from the caller's _llm_available() check to skip a
    redundant HTTP probe on every per-query call.

    context keys: merchant, category, monthly_spend, frequency (visits/60 days)
    """
    parts: list[str] = []
    if context.get("merchant") and context["merchant"] not in ("Unknown", "your usual store", ""):
        parts.append(f"merchant: {context['merchant']}")
    if context.get("monthly_spend", 0) > 0:
        parts.append(f"${context['monthly_spend']:.0f}/month")
    if context.get("frequency", 0) > 0:
        parts.append(f"{context['frequency']:.0f} purchases/60 days")
    if context.get("category"):
        parts.append(f"category: {context['category']}")
    ctx_str = ", ".join(parts) or fallback

    prompt = (
        "You are a search query optimizer for a personal finance savings app.\n"
        f"User context: {ctx_str}\n"
        "Write a single Google search query that would return a page with real, "
        "actionable savings: a coupon, cheaper alternative, loyalty discount, "
        "or cancellation guide specific to this user.\n"
        "Rules: max 10 words, no quotes, no punctuation, current year.\n"
        "Return ONLY the query, nothing else.\n"
    )

    result: str | None = None

    # ── Ollama (caller already confirmed it's up — skip the probe) ────────────
    if ollama_active:
        try:
            import urllib.request as _ur, json as _j
            payload = _j.dumps({"model": _OLLAMA_MODEL, "prompt": prompt, "stream": False}).encode()
            req = _ur.Request(
                f"{_OLLAMA_HOST}/api/generate", data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with _ur.urlopen(req, timeout=30) as resp:
                result = str(_j.loads(resp.read()).get("response", "")).strip()
        except Exception:
            pass

    # ── llama-cpp-python fallback ─────────────────────────────────────────────
    if not result:
        try:
            from models.intelligence import local_llm  # noqa: PLC0415
            result = local_llm.generate(prompt, max_tokens=20, temperature=0.3)
        except Exception:
            pass

    if result:
        result = result.split("\n")[0].strip().strip('"').strip("'")
        if 3 <= len(result.split()) <= 12:
            return result
    return fallback


# ── Query generation ──────────────────────────────────────────────────────────

def _build_queries(
    df: pd.DataFrame,
    suggestions: list[dict],
    model_outputs: dict,
) -> list[dict]:
    """
    Extract search targets from analysis output and rank by priority.
    Returns list of {topic, query, category} — at most _MAX_QUERIES entries.
    """
    seen:    set[str]  = set()
    targets: list[dict] = []

    def _add(
        topic: str, query: str, category: str, priority: int,
        savings: float = 0.0, ctx: dict | None = None,
    ) -> None:
        key = category if priority >= 3 else f"{category}:{topic}"
        if key not in seen:
            seen.add(key)
            targets.append({
                "topic": topic, "query": query, "category": category,
                "_pri": priority, "potential_savings": savings,
                "_ctx": ctx or {},
            })

    for s in suggestions:
        stype   = s.get("type", "")
        cat     = s.get("category", "other")
        details = s.get("details", {})
        msg     = s.get("message", "")
        savings = abs(float(s.get("amount_impact", 0)))

        if stype == "bulk_buy_opportunity":
            merchant = details.get("top_merchant", "")
            freq     = details.get("purchase_frequency", 0)
            spend    = details.get("avg_transaction", 0) * max(freq / 2, 1)
            if merchant and not _is_noisy_merchant_name(merchant):
                q = f"{merchant} bulk buy deal cheap {datetime.now().year}"
            else:
                q = f"buy {cat} in bulk cheap deals {datetime.now().year}"
                merchant = ""  # don't pass noisy name into Ollama rewriter either
            _add(f"Bulk {cat} deals", q, cat, priority=1, savings=savings,
                 ctx={"merchant": merchant, "category": cat,
                      "monthly_spend": spend, "frequency": freq})

        elif stype == "subscription_trap":
            merchant = details.get("merchant", "")
            monthly  = details.get("monthly_cost", savings)
            if merchant:
                q = f"{merchant} cheaper plan cancel subscription alternative {datetime.now().year}"
                _add(f"{merchant} — cheaper plan", q, "subscriptions", priority=2, savings=savings,
                     ctx={"merchant": merchant, "category": "subscriptions",
                          "monthly_spend": monthly, "frequency": 1})

        elif stype == "price_intelligence":
            cheap   = details.get("cheap_merchant", "")
            exp_m   = details.get("expensive_merchant", "")
            monthly = details.get("est_monthly_savings", savings)
            if cheap:
                q = f"{cheap} deals coupons promo code {cat} {datetime.now().year}"
                _add(f"Deals at {cheap}", q, cat, priority=2, savings=savings,
                     ctx={"merchant": exp_m or cheap, "category": cat,
                          "monthly_spend": monthly, "frequency": 0})

        elif stype == "behavioral_bias" and "ImpulseBuy" in msg:
            q = f"stop impulse buying save money {cat} tips tricks {datetime.now().year}"
            _add(f"Curb {cat} impulse spend", q, cat, priority=3, savings=savings,
                 ctx={"merchant": "", "category": cat, "monthly_spend": savings, "frequency": 0})

        elif stype == "forecast_warning":
            if savings > 50:
                q = f"save money {cat} best deals tips {datetime.now().year}"
                _add(f"Save on over-budget {cat}", q, cat, priority=4, savings=savings,
                     ctx={"merchant": "", "category": cat, "monthly_spend": savings, "frequency": 0})

    # Fallback: highest-spend uncovered categories
    if not df.empty:
        cat_spend = (
            df.groupby("category")["amount"].sum().sort_values(ascending=False)
        )
        for cat, spend in cat_spend.items():
            if cat != "other" and cat not in seen:
                q = f"best deals save money on {cat} {datetime.now().year}"
                _add(f"Save on {cat}", q, str(cat), priority=6,
                     ctx={"merchant": "", "category": str(cat), "monthly_spend": float(spend), "frequency": 0})

    targets.sort(key=lambda t: t["_pri"])
    return targets[:_MAX_QUERIES]


# ── Public API ────────────────────────────────────────────────────────────────

class DealFinder:
    """
    Web deal search driven by analysis output.

    find() returns a list of deal groups — each with a human-readable topic,
    the query that was run, and up to 3 real search results from DuckDuckGo
    (cached 24 h in SQLite so repeated /analyze calls are free).

    Degrades gracefully: returns [] if search fails or no network.

    On HPC / shared-IP servers, set BRAVE_API_KEY in .env to use the
    Brave Search API (2 000 free queries/month at api.search.brave.com)
    — DuckDuckGo rate-limits server IPs aggressively.
    """

    def find(
        self,
        df: pd.DataFrame,
        suggestions: list[dict],
        model_outputs: dict,
    ) -> list[dict]:
        _ensure_cache_table()

        # Compute actual monthly spend per category from the user's transactions
        monthly_spend: dict[str, float] = {}
        if not df.empty and "date" in df.columns and "category" in df.columns:
            date_range_days = max(1, (df["date"].max() - df["date"].min()).days + 1)
            for cat, grp in df.groupby("category"):
                monthly_spend[str(cat)] = round(float(grp["amount"].sum()) * 30 / date_range_days, 2)

        targets       = _build_queries(df, suggestions, model_outputs)
        ollama_active = _llm_available()

        # Filter targets before launching threads
        work: list[tuple[dict, float]] = []
        for t in targets:
            cat          = t["category"]
            actual_spend = monthly_spend.get(cat, 0.0)
            if actual_spend < 1.0 or t.get("potential_savings", 0.0) <= 0:
                continue
            ctx = t.get("_ctx", {})
            if ctx.get("monthly_spend", 0) == 0:
                ctx["monthly_spend"] = actual_spend
            work.append((t, actual_spend))

        def _fetch_one(t: dict, actual_spend: float) -> dict | None:
            ctx   = t.get("_ctx", {})
            query = _rewrite_query(t["query"], ctx, ollama_active=ollama_active)
            results = _search(query)
            if not results:
                return None
            return {
                "topic":              t["topic"],
                "query":              query,
                "category":          t["category"],
                "your_monthly_spend": actual_spend,
                "potential_savings":  round(t.get("potential_savings", 0.0), 2),
                "results":            results,
            }

        # Run all searches concurrently — wall time = slowest single request
        deal_groups: list[dict | None] = [None] * len(work)
        with ThreadPoolExecutor(max_workers=min(len(work), 4) or 1) as pool:
            futures = {pool.submit(_fetch_one, t, sp): i for i, (t, sp) in enumerate(work)}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    deal_groups[idx] = fut.result()
                except Exception:
                    pass
        return [g for g in deal_groups if g is not None]
