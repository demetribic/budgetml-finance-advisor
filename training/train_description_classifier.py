"""
training/train_description_classifier.py — Train the transaction auto-classifier.

Two backends are supported:

  TF-IDF + Logistic Regression (default, no GPU, trains in <2 s):
    Saves to  models/saved/description_classifier.pkl

  SetFit  (--setfit, fine-tunable on user labels, trains in ~5 min on CPU/GPU):
    Saves to  models/saved/setfit_description_classifier/
    Requires: pip install setfit datasets
    The API loads SetFit automatically when the directory exists.

Data sources (all additive to the ~280 seed examples):
  --csv PATH        Labeled CSV with columns: category plus description and/or merchant
  --gold-csv PATH   Additional high-trust regression examples (same schema as --csv)
  --from-db         Pull labeled transactions from SQLite (requires run API once)
  --from-feedback   Include pseudo-labels from web-search fallback
  --hf-dataset ID   Online HF dataset (e.g. rajeshradhakrishnan/fin-transaction-category)

Usage
-----
  # Default (TF-IDF, seed only):
  python training/train_description_classifier.py

  # SetFit, seed only:
  python training/train_description_classifier.py --setfit

  # SetFit, fine-tuned on your own labeled DB transactions:
  python training/train_description_classifier.py --setfit --from-db

  # SetFit with a custom backbone:
  python training/train_description_classifier.py --setfit --setfit-model BAAI/bge-small-en-v1.5
"""

from __future__ import annotations

import argparse
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from config import Settings
from models.embeddings.description_classifier import (
    DescriptionClassifier, SetFitDescriptionClassifier, CATEGORIES, SEED_DATA,
    combine_transaction_text,
)


HF_CATEGORY_MAP: dict[str, str] = {
    "shopping": "shopping",
    "retail": "shopping",
    "restaurants": "dining",
    "dining out": "dining",
    "groceries": "groceries",
    "entertainment": "entertainment",
    "transportation": "transportation",
    "utilities": "utilities",
    "subscription": "subscriptions",
    "service subscription": "subscriptions",
    "service subscriptions": "subscriptions",
}

_EXTERNAL_LABEL_KEYWORDS: list[tuple[tuple[str, ...], str]] = [
    (("grocery", "supermarket", "market"), "groceries"),
    (("restaurant", "dining", "food", "coffee", "cafe", "bar"), "dining"),
    (("subscription", "membership", "streaming", "software", "saas", "cloud"), "subscriptions"),
    (("utility", "internet", "wireless", "phone", "electric", "water", "gas bill", "cable"), "utilities"),
    (("transport", "transit", "fuel", "gas station", "parking", "toll", "rideshare", "auto"), "transportation"),
    (("movie", "music", "game", "gaming", "entertainment", "ticket", "concert", "sport"), "entertainment"),
    (("medical", "health", "healthcare", "pharmacy", "doctor", "dental", "vision"), "healthcare"),
    (("shopping", "retail", "clothing", "apparel", "book", "education", "school", "office", "pet", "beauty"), "shopping"),
]

_EXTERNAL_OTHER_KEYWORDS = (
    "housing", "mortgage", "rent", "loan", "debt", "income", "payroll",
    "transfer", "payment", "credit", "tax", "insurance", "fee", "fees",
)


def _normalize_external_label(label: str) -> str | None:
    key = str(label).strip().lower()
    if key in CATEGORIES:
        return key
    mapped = HF_CATEGORY_MAP.get(key)
    if mapped:
        return mapped

    for keywords, category in _EXTERNAL_LABEL_KEYWORDS:
        if any(keyword in key for keyword in keywords):
            return category

    if any(keyword in key for keyword in _EXTERNAL_OTHER_KEYWORDS):
        return "other"
    return None


def _load_hf_labeled_pairs(repo_id: str, include_test: bool) -> tuple[list[str], list[str]]:
    """Load merchant/category pairs from a HF dataset repo containing parquet files."""
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    parquet_files = [f for f in files if f.endswith(".parquet")]

    if not include_test:
        parquet_files = [f for f in parquet_files if "/train" in f or f.startswith("train")]

    if not parquet_files:
        return [], []

    descriptions: list[str] = []
    labels: list[str] = []

    for file_name in parquet_files:
        local_path = hf_hub_download(repo_id=repo_id, filename=file_name, repo_type="dataset")
        df = pd.read_parquet(local_path)

        if "category" not in df.columns:
            continue

        has_merchant = "merchant" in df.columns
        has_description = "description" in df.columns

        text_col = None
        if not (has_merchant or has_description):
            for candidate in ("text", "transaction", "memo"):
                if candidate in df.columns:
                    text_col = candidate
                    break
        if not (has_merchant or has_description or text_col):
            continue

        if has_merchant or has_description:
            text_iter = [
                combine_transaction_text(
                    merchant=str(row.get("merchant", "") or ""),
                    description=str(row.get("description", "") or ""),
                )
                for _, row in df.iterrows()
            ]
        else:
            text_iter = [str(v or "").strip() for v in df[text_col].tolist()]

        for raw_text, raw_label in zip(text_iter, df["category"].tolist()):
            if not raw_text:
                continue

            mapped: str | None = None
            if isinstance(raw_label, (int, float)):
                # Specific mapping for fin-transaction-category class ids.
                int_map = {
                    0: "shopping",
                    1: "dining",
                    2: "entertainment",
                    3: "transportation",
                    4: "other",
                    5: "other",
                    6: "utilities",
                    7: "subscriptions",
                }
                mapped = int_map.get(int(raw_label))
            else:
                mapped = _normalize_external_label(str(raw_label))

            if mapped in CATEGORIES:
                descriptions.append(str(raw_text))
                labels.append(mapped)

    return descriptions, labels


def _duplicate_examples(
    texts: list[str],
    labels: list[str],
    *,
    factor: int,
) -> tuple[list[str], list[str]]:
    if factor <= 1:
        return texts, labels
    out_texts: list[str] = []
    out_labels: list[str] = []
    for text, label in zip(texts, labels):
        out_texts.extend([text] * factor)
        out_labels.extend([label] * factor)
    return out_texts, out_labels


def _load_csv_pairs(path: str | Path) -> tuple[list[str], list[str]]:
    df = pd.read_csv(path)
    if "category" not in df.columns:
        raise ValueError(f"CSV must contain column 'category'. Found: {set(df.columns)}")
    if "merchant" not in df.columns and "description" not in df.columns and "text" not in df.columns:
        raise ValueError(
            "CSV must contain at least one of: merchant, description, text"
        )

    texts: list[str] = []
    labels: list[str] = []
    for _, row in df.iterrows():
        label = str(row.get("category", "")).strip().lower()
        if label not in CATEGORIES:
            continue
        if "text" in df.columns and str(row.get("text", "")).strip():
            text = str(row.get("text", "")).strip()
        else:
            text = combine_transaction_text(
                merchant=str(row.get("merchant", "") or ""),
                description=str(row.get("description", "") or ""),
            )
        if not text:
            continue
        texts.append(text)
        labels.append(label)
    return texts, labels


def _load_db_pairs() -> tuple[list[str], list[str]]:
    from api.db import DB_PATH  # noqa: PLC0415

    if not DB_PATH.exists():
        print(f"  DB not found at {DB_PATH} — skipping.")
        return [], []

    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    try:
        tx_rows = con.execute(
            """
            SELECT merchant, description, category
            FROM transactions
            WHERE category != 'other'
              AND (merchant != '' OR description != '')
            """
        ).fetchall()
        example_rows = con.execute(
            """
            SELECT merchant, description, category
            FROM user_category_examples
            WHERE category != 'other'
            """
        ).fetchall()
        pref_rows = con.execute(
            """
            SELECT merchant_norm AS merchant, '' AS description, category
            FROM user_category_preferences
            WHERE category != 'other'
            """
        ).fetchall()
    finally:
        con.close()

    tx_texts = [
        combine_transaction_text(merchant=str(r["merchant"] or ""), description=str(r["description"] or ""))
        for r in tx_rows
        if str(r["category"]).strip().lower() in CATEGORIES
    ]
    tx_labels = [
        str(r["category"]).strip().lower()
        for r in tx_rows
        if str(r["category"]).strip().lower() in CATEGORIES
    ]

    example_texts = [
        combine_transaction_text(merchant=str(r["merchant"] or ""), description=str(r["description"] or ""))
        for r in example_rows
        if str(r["category"]).strip().lower() in CATEGORIES
    ]
    example_labels = [
        str(r["category"]).strip().lower()
        for r in example_rows
        if str(r["category"]).strip().lower() in CATEGORIES
    ]

    pref_texts = [
        combine_transaction_text(merchant=str(r["merchant"] or ""), description="")
        for r in pref_rows
        if str(r["category"]).strip().lower() in CATEGORIES
    ]
    pref_labels = [
        str(r["category"]).strip().lower()
        for r in pref_rows
        if str(r["category"]).strip().lower() in CATEGORIES
    ]

    example_texts, example_labels = _duplicate_examples(example_texts, example_labels, factor=3)
    pref_texts, pref_labels = _duplicate_examples(pref_texts, pref_labels, factor=4)

    texts = [t for t in tx_texts + example_texts + pref_texts if t]
    labels = (
        tx_labels
        + example_labels
        + pref_labels
    )
    return texts, labels


def _cap_examples_per_label(
    texts: list[str],
    labels: list[str],
    *,
    max_per_label: int,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    if max_per_label <= 0:
        return texts, labels

    grouped: dict[str, list[str]] = defaultdict(list)
    for text, label in zip(texts, labels):
        grouped[label].append(text)

    rng = random.Random(seed)
    out_texts: list[str] = []
    out_labels: list[str] = []
    for label in sorted(grouped):
        examples = grouped[label]
        if len(examples) > max_per_label:
            examples = rng.sample(examples, max_per_label)
        out_texts.extend(examples)
        out_labels.extend([label] * len(examples))
    return out_texts, out_labels


def train(args) -> None:
    cfg = Settings.load()
    cfg.save_dir.mkdir(parents=True, exist_ok=True)

    use_setfit = getattr(args, "setfit", False)
    if use_setfit:
        save_path = cfg.save_dir / "setfit_description_classifier"
    else:
        save_path = cfg.save_dir / "description_classifier.pkl"

    descriptions: list[str] = []
    labels:       list[str] = []

    # ── Optional: labeled CSV ─────────────────────────────────────────────────
    if args.csv:
        csv_texts, csv_labels = _load_csv_pairs(args.csv)
        descriptions += csv_texts
        labels += csv_labels
        print(f"  CSV: loaded {len(csv_texts):,} labeled rows.")

    if args.gold_csv:
        gold_texts, gold_labels = _load_csv_pairs(args.gold_csv)
        gold_texts, gold_labels = _duplicate_examples(gold_texts, gold_labels, factor=4)
        descriptions += gold_texts
        labels += gold_labels
        print(f"  Gold CSV: loaded {len(gold_texts):,} weighted rows.")

    # ── Optional: DB transactions ─────────────────────────────────────────────
    if args.from_db:
        db_texts, db_labels = _load_db_pairs()
        descriptions += db_texts
        labels += db_labels
        print(f"  DB: loaded {len(db_texts):,} weighted labeled rows.")

    # ── Optional: feedback from web fallback pseudo-labels ───────────────────
    if args.from_feedback:
        try:
            from api.db import init_db, get_auto_category_feedback  # noqa: PLC0415
            init_db()
            rows = get_auto_category_feedback(limit=args.feedback_limit)
            fb_descs = [r["description"] for r in rows if r.get("category") in CATEGORIES]
            fb_labels = [r["category"] for r in rows if r.get("category") in CATEGORIES]
            descriptions += fb_descs
            labels += fb_labels
            print(f"  Feedback: loaded {len(fb_descs):,} pseudo-labeled rows.")
        except Exception as e:
            print(f"  Feedback load failed: {e}")

    # ── Optional: online HF dataset ───────────────────────────────────────────
    hf_repo_ids = [s.strip() for s in str(args.hf_dataset or "").split(",") if s.strip()]
    for repo_id in hf_repo_ids:
        hf_descs, hf_labels = _load_hf_labeled_pairs(
            repo_id=repo_id,
            include_test=args.hf_include_test,
        )
        hf_descs, hf_labels = _cap_examples_per_label(
            hf_descs,
            hf_labels,
            max_per_label=args.hf_max_per_category,
        )
        descriptions += hf_descs
        labels += hf_labels
        print(
            f"  HF: loaded {len(hf_descs):,} capped labeled rows from {repo_id} "
            f"(max_per_category={args.hf_max_per_category})."
        )

    total_extra = len(descriptions)
    print(f"\nTraining on {len(SEED_DATA)} seed + {total_extra} user-supplied examples …")

    if use_setfit:
        setfit_model_id = getattr(args, "setfit_model", "sentence-transformers/all-MiniLM-L6-v2")
        print(f"Backend: SetFit  (backbone: {setfit_model_id})")
        clf = SetFitDescriptionClassifier(model_id=setfit_model_id)
        clf.fit(
            descriptions or None,
            labels or None,
            num_iterations=getattr(args, "setfit_iterations", 20),
            num_epochs=getattr(args, "setfit_epochs", 1),
        )
    else:
        print("Backend: TF-IDF + Logistic Regression")
        clf = DescriptionClassifier()
        clf.fit(descriptions or None, labels or None)

    clf.save(save_path)
    print(f"Saved → {save_path}")
    print(f"Parameters: {clf.count_parameters():,}")

    # ── Quick accuracy check on held-out seed sample ──────────────────────────
    random.seed(42)
    test_n    = min(50, len(SEED_DATA))
    test_data = random.sample(SEED_DATA, test_n)
    test_d, test_l = zip(*test_data)
    preds = clf.predict(list(test_d))
    acc   = sum(p == l for p, l in zip(preds, test_l)) / test_n
    print(f"\nSeed-data held-out accuracy (n={test_n}): {acc:.1%}")

    # Per-category breakdown
    from collections import defaultdict
    per_cat: dict[str, list[bool]] = defaultdict(list)
    for pred, true in zip(preds, test_l):
        per_cat[true].append(pred == true)
    for cat, cat_results in sorted(per_cat.items()):
        n_correct = sum(cat_results)
        print(f"  {cat:<15} {n_correct}/{len(cat_results)}  ({100*n_correct/len(cat_results):.0f}%)")


def parse_args():
    p = argparse.ArgumentParser(description="Train transaction description classifier")
    p.add_argument("--csv",      default=None,
                   help="Path to labeled CSV with columns: category plus merchant/description/text")
    p.add_argument("--gold-csv", default=None,
                   help="Path to high-trust gold CSV with columns: category plus merchant/description/text")
    p.add_argument("--from-db",  action="store_true",
                   help="Include labeled transactions from the SQLite DB")
    p.add_argument("--from-feedback", action="store_true",
                   help="Include pseudo-labels generated by /categorize web fallback")
    p.add_argument("--feedback-limit", type=int, default=5000,
                   help="Max pseudo-labeled feedback rows to load when --from-feedback is set")
    p.add_argument("--hf-dataset", default=None,
                   help="Optional comma-separated HF dataset repo ids with text/category labels")
    p.add_argument("--hf-include-test", action="store_true",
                   help="Include test split parquet files from --hf-dataset")
    p.add_argument("--hf-max-per-category", type=int, default=1200,
                   help="Cap per-category examples loaded from each HF dataset to avoid overfitting synthetic data")
    # ── SetFit options ───────────────────────────────────────────────────────
    p.add_argument("--setfit", action="store_true",
                   help="Use SetFit instead of TF-IDF+LR (requires: pip install setfit datasets)")
    p.add_argument("--setfit-model", default="sentence-transformers/all-MiniLM-L6-v2",
                   help="HuggingFace model ID for the SetFit backbone (default: all-MiniLM-L6-v2)")
    p.add_argument("--setfit-iterations", type=int, default=20,
                   help="Number of contrastive pair iterations per class (default: 20)")
    p.add_argument("--setfit-epochs", type=int, default=1,
                   help="Training epochs for the classification head (default: 1)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
