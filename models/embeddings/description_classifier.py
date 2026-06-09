"""
models/embeddings/description_classifier.py — Auto-categorize transaction descriptions.

TF-IDF char n-gram + Logistic Regression trained on labeled transaction descriptions.
Inference: CPU-only, ~0.1 ms per description.

Usage
-----
  clf = DescriptionClassifier().fit()           # seed data only
  clf.fit(my_descs, my_labels)                  # seed + user data
  clf.save("models/saved/description_classifier.pkl")

  clf = DescriptionClassifier.load(path)
  cats = clf.predict(["STARBUCKS", "UBER X"])   # → ["dining", "transportation"]
  probs = clf.predict_proba(["AMAZON"])          # → [{"shopping": 0.92, ...}]
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import ClassVar

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

CATEGORIES = [
    "dining", "groceries", "subscriptions", "transportation",
    "utilities", "entertainment", "shopping", "healthcare", "other",
]

_NOISE_TOKENS = {
    "VISA", "MASTERCARD", "DEBIT", "CREDIT", "CARD", "PUR", "PURCHASE",
    "PAYMENT", "POS", "ACH", "TRANSFER", "TRNSFR", "WITHDRAWAL", "DDA",
    "AP", "DBT", "CHECK", "CHK", "ONLINE", "AUTH", "PENDING", "POSTED",
    "NUM", "NO", "REF", "ID", "DESC", "DETAIL", "ENTRY", "RECURRING",
}

_STATE_CODES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI",
    "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI",
    "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC",
    "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT",
    "VT", "VA", "WA", "WV", "WI", "WY",
}


def _is_mostly_numeric(token: str) -> bool:
    digits = sum(ch.isdigit() for ch in token)
    return digits >= 3 or (digits > 0 and digits >= len(token) - 1)


def _clean_tokens(text: str) -> list[str]:
    clean = re.sub(r"[^A-Z0-9 ]", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    raw_tokens = clean.split(" ") if clean else []

    filtered: list[str] = []
    for tok in raw_tokens:
        if not tok:
            continue
        if tok in _NOISE_TOKENS or tok in _STATE_CODES:
            continue
        if _is_mostly_numeric(tok):
            continue
        if len(tok) <= 1:
            continue
        filtered.append(tok)
    return filtered


def _augment_with_statement_noise(text: str) -> list[str]:
    """Generate deterministic noisy statement-style variants for robustness."""
    base = re.sub(r"\s+", " ", str(text).upper()).strip()
    if not base:
        return []
    return [
        base,
        f"VISA DDA PUR AP {base}",
        f"POS DEBIT {base} AUTH",
        f"CARD PURCHASE {base} REF 123456",
        f"{base} PAYMENT",
    ]


def _normalize(text: str) -> str:
    text = str(text).upper()
    filtered = _clean_tokens(text)
    if filtered:
        return " ".join(filtered)
    text = re.sub(r"\b\d{4,}\b", "", text)
    text = re.sub(r"\b[A-Z]{2}\d+\b", "", text)
    text = re.sub(r"[^A-Z ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def combine_transaction_text(
    merchant: str = "",
    description: str = "",
    *,
    repeat_merchant: int = 2,
) -> str:
    """
    Build a single classifier string from merchant + description.

    Repeating the merchant slightly gives statement-text models a stronger signal
    about the entity being charged without fully discarding the description.
    """
    merchant_norm = _normalize(merchant)
    description_norm = _normalize(description)

    if merchant_norm and description_norm:
        if description_norm == merchant_norm:
            return merchant_norm
        return " ".join(
            [merchant_norm] * max(1, repeat_merchant) + [description_norm]
        ).strip()
    return merchant_norm or description_norm


# ── Seed training data ────────────────────────────────────────────────────────
# ~180 labeled examples covering the most common US merchant patterns.

SEED_DATA: list[tuple[str, str]] = [
    # dining
    ("STARBUCKS", "dining"), ("MCDONALDS", "dining"), ("CHIPOTLE", "dining"),
    ("SUBWAY SANDWICHES", "dining"), ("DUNKIN DONUTS", "dining"), ("PANERA BREAD", "dining"),
    ("CHICK FIL A", "dining"), ("TACO BELL", "dining"), ("WENDYS", "dining"),
    ("BURGER KING", "dining"), ("SONIC DRIVE IN", "dining"), ("FIVE GUYS", "dining"),
    ("DOMINOS PIZZA", "dining"), ("PIZZA HUT", "dining"), ("PAPA JOHNS", "dining"),
    ("GRUBHUB", "dining"), ("DOORDASH", "dining"), ("UBER EATS", "dining"),
    ("POSTMATES", "dining"), ("SEAMLESS", "dining"), ("INSTACART MEALS", "dining"),
    ("RESTAURANT", "dining"), ("CAFE", "dining"), ("DINER", "dining"),
    ("SUSHI BAR", "dining"), ("BURRITO", "dining"), ("THAI KITCHEN", "dining"),
    ("ITALIAN RISTORANTE", "dining"), ("MEXICAN GRILL", "dining"),
    ("CHINESE KITCHEN", "dining"), ("COFFEE SHOP", "dining"),
    ("BAKERY", "dining"), ("JUICE BAR", "dining"), ("SMOOTHIE KING", "dining"),
    # groceries
    ("WHOLE FOODS MARKET", "groceries"), ("TRADER JOES", "groceries"),
    ("KROGER", "groceries"), ("SAFEWAY", "groceries"), ("PUBLIX", "groceries"),
    ("ALDI", "groceries"), ("COSTCO WHOLESALE", "groceries"), ("SAMS CLUB", "groceries"),
    ("WALMART GROCERY", "groceries"), ("FOOD LION", "groceries"),
    ("WEGMANS", "groceries"), ("HEB GROCERY", "groceries"),
    ("STOP SHOP", "groceries"), ("GIANT FOOD", "groceries"), ("MEIJER", "groceries"),
    ("WINN DIXIE", "groceries"), ("HARRIS TEETER", "groceries"),
    ("SPROUTS MARKET", "groceries"), ("GROCERY STORE", "groceries"),
    ("SUPERMARKET", "groceries"), ("FRESH MARKET", "groceries"),
    ("WAWA MARKET", "groceries"), ("WAWA FOOD", "groceries"),
    # subscriptions
    ("NETFLIX", "subscriptions"), ("SPOTIFY", "subscriptions"),
    ("HULU", "subscriptions"), ("DISNEY PLUS", "subscriptions"),
    ("HBO MAX", "subscriptions"), ("APPLE TV PLUS", "subscriptions"),
    ("AMAZON PRIME", "subscriptions"), ("YOUTUBE PREMIUM", "subscriptions"),
    ("PARAMOUNT PLUS", "subscriptions"), ("PEACOCK TV", "subscriptions"),
    ("ADOBE CREATIVE CLOUD", "subscriptions"), ("MICROSOFT OFFICE", "subscriptions"),
    ("DROPBOX", "subscriptions"), ("GOOGLE ONE STORAGE", "subscriptions"),
    ("ICLOUD STORAGE", "subscriptions"), ("AUDIBLE", "subscriptions"),
    ("LINKEDIN PREMIUM", "subscriptions"), ("NYTIMES DIGITAL", "subscriptions"),
    ("PLANET FITNESS", "subscriptions"), ("ANYTIME FITNESS", "subscriptions"),
    ("YMCA MEMBERSHIP", "subscriptions"), ("GYM MEMBERSHIP", "subscriptions"),
    ("EQUINOX", "subscriptions"), ("CRUNCH FITNESS", "subscriptions"),
    ("MONTHLY SUBSCRIPTION", "subscriptions"), ("ANNUAL SUBSCRIPTION", "subscriptions"),
    ("SOFTWARE SUBSCRIPTION", "subscriptions"), ("CLOUD SUBSCRIPTION", "subscriptions"),
    ("AI SUBSCRIPTION", "subscriptions"), ("ANTHROPIC SUBSCRIPTION", "subscriptions"),
    ("OPENAI SUBSCRIPTION", "subscriptions"),
    # transportation
    ("UBER TRIP", "transportation"), ("LYFT RIDE", "transportation"),
    ("SHELL OIL", "transportation"), ("CHEVRON STATION", "transportation"),
    ("EXXON MOBIL", "transportation"), ("BP GAS", "transportation"),
    ("SUNOCO", "transportation"), ("SPEEDWAY GAS", "transportation"),
    ("CIRCLE K GAS", "transportation"), ("MARATHON GAS", "transportation"),
    ("METRO TRANSIT", "transportation"), ("AMTRAK TRAIN", "transportation"),
    ("AMERICAN AIRLINES", "transportation"), ("DELTA AIR LINES", "transportation"),
    ("UNITED AIRLINES", "transportation"), ("SOUTHWEST AIRLINES", "transportation"),
    ("SPIRIT AIRLINES", "transportation"), ("JETBLUE AIRWAYS", "transportation"),
    ("EZ PASS TOLL", "transportation"), ("PARKING METER", "transportation"),
    ("CAR WASH", "transportation"), ("JIFFY LUBE", "transportation"),
    ("AUTOZONE", "transportation"), ("PEP BOYS", "transportation"),
    ("ENTERPRISE RENT", "transportation"), ("HERTZ CAR", "transportation"),
    ("WAWA GAS", "transportation"), ("WAWA FUEL", "transportation"),
    ("SHEETZ", "transportation"), ("QUIKTRIP", "transportation"),
    ("RACETRAC", "transportation"), ("CUMBERLAND FARMS", "transportation"),
    ("MURPHY USA", "transportation"), ("PILOT TRAVEL CENTER", "transportation"),
    ("LOVE S TRAVEL STOP", "transportation"), ("MORRISVILLE FUELS", "transportation"),
    ("SPEEDWAY FUEL", "transportation"),
    ("SEVEN ELEVEN GAS", "transportation"), ("SUNOCO GAS", "transportation"),
    # utilities
    ("ELECTRIC BILL", "utilities"), ("GAS UTILITY", "utilities"),
    ("WATER BILL", "utilities"), ("COMCAST XFINITY", "utilities"),
    ("VERIZON WIRELESS", "utilities"), ("ATT WIRELESS", "utilities"),
    ("TMOBILE", "utilities"), ("SPECTRUM CABLE", "utilities"),
    ("COX COMMUNICATIONS", "utilities"), ("CENTURYLINK", "utilities"),
    ("NATIONAL GRID ENERGY", "utilities"), ("DUKE ENERGY", "utilities"),
    ("DOMINION ENERGY", "utilities"), ("SOUTHERN COMPANY", "utilities"),
    ("INTERNET SERVICE", "utilities"), ("PHONE BILL", "utilities"),
    ("WASTE MANAGEMENT", "utilities"), ("SEWAGE BILL", "utilities"),
    # entertainment
    ("AMC THEATERS", "entertainment"), ("REGAL CINEMAS", "entertainment"),
    ("CINEMARK", "entertainment"), ("TICKETMASTER", "entertainment"),
    ("STUBHUB TICKETS", "entertainment"), ("LIVE NATION", "entertainment"),
    ("STEAM PURCHASE", "entertainment"), ("PLAYSTATION NETWORK", "entertainment"),
    ("XBOX LIVE GOLD", "entertainment"), ("NINTENDO ESHOP", "entertainment"),
    ("APPLE ARCADE", "entertainment"), ("TWITCH", "entertainment"),
    ("BOWLING ALLEY", "entertainment"), ("ESCAPE ROOM", "entertainment"),
    ("MUSEUM ADMISSION", "entertainment"), ("ZOO TICKET", "entertainment"),
    ("GOLF COURSE", "entertainment"), ("MINI GOLF", "entertainment"),
    ("LASER TAG", "entertainment"), ("TRAMPOLINE PARK", "entertainment"),
    # shopping
    ("AMAZON PURCHASE", "shopping"), ("EBAY PURCHASE", "shopping"),
    ("TARGET STORE", "shopping"), ("WALMART STORE", "shopping"),
    ("BEST BUY", "shopping"), ("HOME DEPOT", "shopping"), ("LOWES", "shopping"),
    ("MACYS", "shopping"), ("NORDSTROM", "shopping"), ("BLOOMINGDALES", "shopping"),
    ("ZARA", "shopping"), ("HM CLOTHING", "shopping"), ("GAP STORE", "shopping"),
    ("OLD NAVY", "shopping"), ("BANANA REPUBLIC", "shopping"),
    ("NIKE STORE", "shopping"), ("ADIDAS", "shopping"), ("UNDER ARMOUR", "shopping"),
    ("APPLE STORE", "shopping"), ("IKEA", "shopping"), ("MARSHALLS", "shopping"),
    ("TJ MAXX", "shopping"), ("ROSS STORES", "shopping"), ("DOLLAR TREE", "shopping"),
    ("DOLLAR GENERAL", "shopping"), ("FIVE BELOW", "shopping"),
    # healthcare
    ("WALGREENS PHARMACY", "healthcare"), ("CVS PHARMACY", "healthcare"),
    ("RITE AID", "healthcare"), ("HOSPITAL PAYMENT", "healthcare"),
    ("URGENT CARE CENTER", "healthcare"), ("DOCTORS OFFICE", "healthcare"),
    ("DENTAL OFFICE", "healthcare"), ("VISION CENTER", "healthcare"),
    ("OPTOMETRY CLINIC", "healthcare"), ("PHYSICAL THERAPY", "healthcare"),
    ("CHIROPRACTOR", "healthcare"), ("LABCORP", "healthcare"),
    ("QUEST DIAGNOSTICS", "healthcare"), ("HEALTH INSURANCE", "healthcare"),
    ("PRESCRIPTION RX", "healthcare"), ("PLANNED PARENTHOOD", "healthcare"),
    # other
    ("ATM WITHDRAWAL", "other"), ("BANK SERVICE FEE", "other"),
    ("WIRE TRANSFER", "other"), ("VENMO PAYMENT", "other"),
    ("ZELLE TRANSFER", "other"), ("PAYPAL TRANSFER", "other"),
    ("CASHAPP", "other"), ("LOAN PAYMENT", "other"),
    ("MORTGAGE PAYMENT", "other"), ("RENT PAYMENT", "other"),
    ("INSURANCE PREMIUM", "other"), ("TAX PAYMENT IRS", "other"),
    # additional dining
    ("WHATABURGER", "dining"), ("IN N OUT BURGER", "dining"),
    ("RAISING CANES", "dining"), ("SHAKE SHACK", "dining"),
    ("WINGSTOP", "dining"), ("PANDA EXPRESS", "dining"),
    ("OLIVE GARDEN", "dining"), ("APPLEBEES", "dining"),
    ("CRACKER BARREL", "dining"), ("IHOP", "dining"),
    ("WAFFLE HOUSE", "dining"), ("DENNY S", "dining"),
    ("JERSEY MIKE S SUBS", "dining"), ("JIMMY JOHN S", "dining"),
    ("POTBELLY", "dining"), ("FIREHOUSE SUBS", "dining"),
    # additional groceries
    ("INSTACART GROCERIES", "groceries"), ("INSTACART MARKET", "groceries"),
    ("LIDL", "groceries"), ("ALDI FOODS", "groceries"),
    ("BJS WHOLESALE", "groceries"), ("MARKET BASKET", "groceries"),
    ("PIGGLY WIGGLY", "groceries"), ("STATER BROS", "groceries"),
    ("LUCKY SUPERMARKETS", "groceries"), ("RALEY S", "groceries"),
    ("SHOPRITE", "groceries"), ("ACME MARKETS", "groceries"),
    ("FOOD DEPOT", "groceries"), ("WINCO FOODS", "groceries"),
    ("SEVEN ELEVEN", "groceries"), ("7 ELEVEN", "groceries"),
    ("7 11", "groceries"), ("711", "groceries"),
    # additional subscriptions
    ("APPLE ONE", "subscriptions"), ("APPLE MUSIC", "subscriptions"),
    ("DUOLINGO PLUS", "subscriptions"), ("HEADSPACE", "subscriptions"),
    ("CALM APP", "subscriptions"), ("NOOM", "subscriptions"),
    ("WEIGHT WATCHERS", "subscriptions"), ("MASTERCLASS", "subscriptions"),
    ("SKILLSHARE", "subscriptions"), ("COURSERA PLUS", "subscriptions"),
    ("NOTION", "subscriptions"), ("SLACK", "subscriptions"),
    ("ZOOM SUBSCRIPTION", "subscriptions"), ("GITHUB COPILOT", "subscriptions"),
    ("APPLE COM BILL", "subscriptions"), ("APPLE COM BILL ITUNES", "subscriptions"),
    ("APPLE SERVICES", "subscriptions"), ("CLAUDE SUBSCRIPTION ANTHROPIC", "subscriptions"),
    ("ANTHROPIC PBC", "subscriptions"),
    # additional transportation
    ("TESLA SUPERCHARGER", "transportation"), ("BLINK CHARGING", "transportation"),
    ("CHARGEPOINT", "transportation"), ("EVGO", "transportation"),
    ("GREYHOUND BUS", "transportation"), ("MEGABUS", "transportation"),
    ("ZIPCAR", "transportation"), ("BIRD SCOOTER", "transportation"),
    ("LIME SCOOTER", "transportation"), ("CITI BIKE", "transportation"),
    ("NJ TRANSIT", "transportation"), ("MTA SUBWAY", "transportation"),
    ("MILEAGE PLUS", "transportation"),
    # additional utilities
    ("GOOGLE FI", "utilities"), ("MINT MOBILE", "utilities"),
    ("VISIBLE WIRELESS", "utilities"), ("CRICKET WIRELESS", "utilities"),
    ("BOOST MOBILE", "utilities"), ("DIRECTV", "utilities"),
    ("DISH NETWORK", "utilities"), ("FRONTIER COMM", "utilities"),
    ("LUMEN TECHNOLOGIES", "utilities"), ("OPTIMUM CABLE", "utilities"),
    # additional entertainment
    ("DAVE BUSTER S", "entertainment"), ("MAIN EVENT", "entertainment"),
    ("ROUND ONE ENTERTAINMENT", "entertainment"), ("TOPGOLF", "entertainment"),
    ("PELOTON", "entertainment"), ("CLASSPASS", "entertainment"),
    ("FANDANGO", "entertainment"), ("ATOM TICKETS", "entertainment"),
    ("EVENTBRITE", "entertainment"),
    # additional shopping
    ("SHEIN", "shopping"), ("TEMU", "shopping"), ("WISH", "shopping"),
    ("WAYFAIR", "shopping"), ("OVERSTOCK", "shopping"),
    ("CHEWY PET", "shopping"), ("PETSMART", "shopping"), ("PETCO", "shopping"),
    ("MICHAELS CRAFT", "shopping"), ("HOBBY LOBBY", "shopping"),
    ("BED BATH BEYOND", "shopping"), ("CRATE BARREL", "shopping"),
    ("WILLIAMS SONOMA", "shopping"), ("POTTERY BARN", "shopping"),
    ("MACYS COM", "shopping"), ("KOHLS", "shopping"),
    ("SEPHORA", "shopping"), ("ULTA BEAUTY", "shopping"),
    # additional healthcare
    ("GOODRX", "healthcare"), ("HIMS", "healthcare"), ("HERS", "healthcare"),
    ("ROMAN HEALTH", "healthcare"), ("TELADOC", "healthcare"),
    ("NURX", "healthcare"), ("AMAZON PHARMACY", "healthcare"),
    ("COSTCO PHARMACY", "healthcare"), ("KAISER PERMANENTE", "healthcare"),
    ("BLUE CROSS", "healthcare"), ("CIGNA", "healthcare"),
    ("AETNA HEALTH", "healthcare"), ("HUMANA", "healthcare"),
    ("MINDPATH CARE", "healthcare"), ("BETTERHELP", "healthcare"),
    ("TALKSPACE", "healthcare"),
    # additional other
    ("COINBASE", "other"), ("ROBINHOOD", "other"),
    ("FIDELITY INVESTMENTS", "other"), ("VANGUARD", "other"),
    ("CHARLES SCHWAB", "other"), ("ETRADE", "other"),
    ("SOFI TRANSFER", "other"), ("CHIME TRANSFER", "other"),
    ("APPLE PAY CASH", "other"), ("GOOGLE PAY", "other"),
    # liquor / alcohol retail — shopping, not entertainment
    ("LIQUOR STORE", "shopping"), ("WINE SHOP", "shopping"),
    ("TOTAL WINE", "shopping"), ("BEV MO", "shopping"),
    ("ABC FINE WINE", "shopping"), ("SPEC S WINES", "shopping"),
    ("BINNY S BEVERAGE", "shopping"), ("STATE LIQUOR STORE", "shopping"),
    ("BOTTLES AND CANS", "shopping"), ("SPIRITS SHOP", "shopping"),
    # bars → dining (consumed on-premise); nightclubs → entertainment (cover charge model)
    ("BAR AND GRILL", "dining"), ("SPORTS BAR", "dining"),
    ("BREWERY TAP", "dining"), ("CRAFT BREWERY", "dining"),
    ("COCKTAIL LOUNGE", "dining"), ("WINE BAR", "dining"),
    ("NIGHTCLUB", "entertainment"), ("NIGHT CLUB", "entertainment"),
    # international / foreign transaction fees — always other
    ("INTL FEE", "other"), ("INTERNATIONAL FEE", "other"),
    ("FOREIGN TRANSACTION FEE", "other"), ("FX FEE", "other"),
    ("FOREIGN EXCHANGE FEE", "other"), ("CURRENCY CONVERSION FEE", "other"),
    ("CROSS BORDER FEE", "other"), ("INTL TRANSACTION", "other"),
    # bank/card service charges — other (not utilities)
    ("SERVICE CHARGE", "other"), ("MONTHLY SERVICE FEE", "other"),
    ("ACCOUNT FEE", "other"), ("OVERDRAFT FEE", "other"),
    ("LATE FEE", "other"), ("ANNUAL FEE", "other"),
    ("CARD FEE", "other"), ("MAINTENANCE FEE", "other"),
    # airport / travel retail — shopping
    ("AIRPORT SHOP", "shopping"), ("TERMINAL SHOP", "shopping"),
    ("DUTY FREE", "shopping"), ("GLOBAL BAZAAR", "shopping"),
    ("AIRPORT NEWSSTAND", "shopping"), ("HUDSON NEWS", "shopping"),
    ("PARADIES SHOP", "shopping"), ("RELAY TRAVEL", "shopping"),
    # hotel / resort markets — shopping
    ("HOTEL MARKET", "shopping"), ("RESORT SHOP", "shopping"),
    ("HOTEL GIFT SHOP", "shopping"), ("RESORT MARKET", "shopping"),
    ("RIU MARKET", "shopping"), ("MARRIOTT SHOP", "shopping"),
    ("HILTON MARKET", "shopping"), ("HYATT MARKET", "shopping"),
    # hotel stays — other (no travel category; avoids misclassifying as subscriptions)
    ("HOTEL STAY", "other"), ("RESORT STAY", "other"),
    ("AIRBNB", "other"), ("VRBO", "other"),
    ("MARRIOTT HOTEL", "other"), ("HILTON HOTEL", "other"),
    ("HYATT HOTEL", "other"), ("RIU HOTEL", "other"),
    ("IHG HOTEL", "other"), ("WYNDHAM HOTEL", "other"),
    # dining at airports / travel food
    ("AIRPORT FOOD", "dining"), ("TERMINAL FOOD COURT", "dining"),
    ("TRAVEL PLAZA FOOD", "dining"),
    # anthropic / AI subscriptions — make sure "Claude" maps correctly
    ("ANTHROPIC", "subscriptions"), ("CLAUDE SUBSCRIPTION", "subscriptions"),
    ("OPENAI", "subscriptions"), ("CHATGPT PLUS", "subscriptions"),
]


class DescriptionClassifier:
    """
    Classifies transaction descriptions into spending categories.

    Architecture: TF-IDF char-wb (2-4 gram) → Logistic Regression (multinomial).
    No GPU required. Trains in <2 s on CPU, infers in <0.2 ms per description.
    """

    # TF-IDF softmax probabilities are overconfident; 0.40 is a conservative trigger
    # so that Ollama only steps in when the model is genuinely split between classes.
    confidence_threshold: ClassVar[float] = 0.40

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self.confidence_threshold = float(type(self).confidence_threshold)

    def _calibrate_threshold(
        self,
        texts: list[str],
        labels: list[str],
    ) -> None:
        if self._pipeline is None or not texts or len(texts) != len(labels):
            return
        probs = self.predict_proba(texts)
        rows: list[tuple[float, bool]] = []
        for p, true_label in zip(probs, labels):
            pred = max(p, key=p.get)
            conf = float(p[pred])
            rows.append((conf, pred == true_label))

        if not rows:
            return

        best_threshold = 0.40
        best_score = -1.0
        for threshold in np.linspace(0.35, 0.90, 23):
            accepted = [ok for conf, ok in rows if conf >= threshold]
            coverage = len(accepted) / len(rows)
            accuracy = sum(accepted) / len(accepted) if accepted else 0.0
            score = (0.75 * accuracy) + (0.25 * coverage)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)

        self.confidence_threshold = round(best_threshold, 2)

    def fit(
        self,
        descriptions: list[str] | None = None,
        labels:       list[str] | None = None,
    ) -> "DescriptionClassifier":
        """
        Train on seed data plus any user-supplied labeled pairs.

        Parameters
        ----------
        descriptions : list of raw transaction description strings (optional)
        labels       : matching category labels (optional, must pair with descriptions)
        """
        seed_d, seed_l = zip(*SEED_DATA)
        all_d = list(seed_d)
        all_l = list(seed_l)

        if descriptions and labels:
            all_d += [str(d) for d in descriptions]
            all_l += [str(l) for l in labels]

        # Add synthetic statement-noise variants so the model learns to ignore
        # card-rail boilerplate and focus on merchant semantics.
        aug_d: list[str] = []
        aug_l: list[str] = []
        for d, l in zip(all_d, all_l):
            for v in _augment_with_statement_noise(d):
                aug_d.append(v)
                aug_l.append(l)

        all_d = aug_d
        all_l = aug_l

        X = [_normalize(d) for d in all_d]

        self._pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                max_features=20_000,
                sublinear_tf=True,
                min_df=1,
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=2.0,
                solver="lbfgs",
                n_jobs=-1,
            )),
        ])
        self._pipeline.fit(X, all_l)
        self._calibrate_threshold(list(seed_d), list(seed_l))
        return self

    def predict(self, descriptions: list[str]) -> list[str]:
        """Return the most likely category for each description."""
        if self._pipeline is None:
            raise RuntimeError("Call fit() or load() first.")
        probs = self.predict_proba(descriptions)
        return [max(p, key=p.get) for p in probs]

    def predict_proba(self, descriptions: list[str]) -> list[dict[str, float]]:
        """Return per-category probability dict for each description."""
        if self._pipeline is None:
            raise RuntimeError("Call fit() or load() first.")
        norm_descs = [_normalize(d) for d in descriptions]
        proba   = self._pipeline.predict_proba(norm_descs)
        classes = self._pipeline.classes_
        return [
            {str(c): round(float(p), 4) for c, p in zip(classes, row)}
            for row in proba
        ]

    def count_parameters(self) -> int:
        if self._pipeline is None:
            return 0
        vocab = len(self._pipeline.named_steps["tfidf"].vocabulary_)
        return vocab * len(CATEGORIES)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "DescriptionClassifier":
        return joblib.load(str(path))


# ── SetFit-based classifier (fine-tunable, few-shot) ─────────────────────────

_SETFIT_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_SETFIT_MANIFEST = "setfit_manifest.json"


class SetFitDescriptionClassifier:
    """
    Few-shot, fine-tunable transaction description classifier.

    Uses SetFit (contrastive pair fine-tuning on a sentence transformer) to learn
    a per-user category mapping from as few as 8 labeled examples per class.

    Architecture
    ------------
    - Backbone : sentence-transformers/all-MiniLM-L6-v2 (22 M params, CPU-friendly)
    - Head     : logistic regression on 384-dim sentence embeddings
    - Training : SetFit contrastive fine-tuning (generates pairs, runs ~5 min on CPU)

    Inference  : ~5 ms per batch on CPU — identical speed to TF-IDF+LR.

    Interface is identical to DescriptionClassifier so ModelRegistry can use either.
    """

    # Lower threshold (0.50) so SetFit triggers Ollama fallback when uncertain.
    # SetFit struggles with merchant names vs TF-IDF; this ensures high-uncertainty
    # cases get the Ollama LLM. Raised from 0.65 after empirical testing.
    confidence_threshold: ClassVar[float] = 0.50

    def __init__(self, model_id: str = _SETFIT_DEFAULT_MODEL) -> None:
        self.model_id  = model_id
        self._model    = None   # setfit.SetFitModel loaded after fit/load
        self.confidence_threshold = float(type(self).confidence_threshold)

    def _calibrate_threshold(
        self,
        texts: list[str],
        labels: list[str],
    ) -> None:
        if self._model is None or not texts or len(texts) != len(labels):
            return
        probs = self.predict_proba(texts)
        rows: list[tuple[float, bool]] = []
        for p, true_label in zip(probs, labels):
            pred = max(p, key=p.get)
            conf = float(p[pred])
            rows.append((conf, pred == true_label))

        if not rows:
            return

        best_threshold = 0.50
        best_score = -1.0
        for threshold in np.linspace(0.40, 0.90, 21):
            accepted = [ok for conf, ok in rows if conf >= threshold]
            coverage = len(accepted) / len(rows)
            if coverage < 0.50:
                continue
            accuracy = sum(accepted) / len(accepted) if accepted else 0.0
            score = (0.75 * accuracy) + (0.25 * coverage)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)

        self.confidence_threshold = round(best_threshold, 2)

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(
        self,
        descriptions: list[str] | None = None,
        labels:       list[str] | None = None,
        num_iterations: int = 50,
        num_epochs: int = 10,
    ) -> "SetFitDescriptionClassifier":
        """
        Fine-tune on seed data plus any user-supplied pairs.

        Parameters
        ----------
        descriptions  : additional labeled description strings (optional)
        labels        : matching category labels (optional)
        num_iterations: number of contrastive pair iterations per class
        num_epochs    : training epochs for the classification head
        """
        try:
            from setfit import SetFitModel, Trainer, TrainingArguments
            from datasets import Dataset as HFDataset
        except ImportError as e:
            raise ImportError(
                "SetFit dependencies not installed. Run: pip install setfit datasets"
            ) from e

        seed_d, seed_l = zip(*SEED_DATA)
        all_d = list(seed_d)
        all_l = list(seed_l)

        # SetFit learns semantic embeddings directly from merchant names.
        # Skip statement-noise augmentation (used for TF-IDF to ignore card-rail boilerplate).
        # Semantic embeddings should capture merchant semantics, not robustness to noise.
        if descriptions and labels:
            all_d += [str(d) for d in descriptions]
            all_l += [str(l) for l in labels]

        dataset = HFDataset.from_dict({"text": all_d, "label": all_l})

        self._model = SetFitModel.from_pretrained(
            self.model_id,
            labels=CATEGORIES,
        )

        _ckpt_dir = Path(__file__).parent.parent / "saved" / "setfit_checkpoints"
        train_args = TrainingArguments(
            output_dir=str(_ckpt_dir),
            num_iterations=num_iterations,
            num_epochs=num_epochs,
            body_learning_rate=2e-5,
            head_learning_rate=1e-2,
        )

        trainer = Trainer(
            model=self._model,
            args=train_args,
            train_dataset=dataset,
        )
        trainer.train()
        self._calibrate_threshold(list(seed_d), list(seed_l))
        return self

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, descriptions: list[str]) -> list[str]:
        if self._model is None:
            raise RuntimeError("Call fit() or load() first.")
        preds = self._model.predict([_normalize(d) for d in descriptions])
        # SetFit may return numpy arrays or torch tensors — coerce to str list
        return [str(p) for p in preds]

    def predict_proba(self, descriptions: list[str]) -> list[dict[str, float]]:
        if self._model is None:
            raise RuntimeError("Call fit() or load() first.")
        proba = self._model.predict_proba([_normalize(d) for d in descriptions])
        # proba is (N, num_classes) numpy array; model.labels gives class order
        labels = self._model.labels or CATEGORIES
        results = []
        for row in proba:
            results.append(
                {str(lbl): round(float(p), 4) for lbl, p in zip(labels, row)}
            )
        return results

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save to a directory (not a .pkl). Creates <path>/ with model files."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self._model is None:
            raise RuntimeError("Nothing to save — model has not been trained yet.")
        self._model.save_pretrained(str(path))
        (path / _SETFIT_MANIFEST).write_text(
            json.dumps(
                {
                    "model_id": self.model_id,
                    "type": "setfit",
                    "confidence_threshold": self.confidence_threshold,
                }
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> "SetFitDescriptionClassifier":
        """Load from a directory created by save()."""
        try:
            from setfit import SetFitModel
        except ImportError as e:
            raise ImportError(
                "SetFit dependencies not installed. Run: pip install setfit datasets"
            ) from e
        path = Path(path)
        obj  = cls()
        obj._model = SetFitModel.from_pretrained(str(path), local_files_only=True)
        manifest_path = path / _SETFIT_MANIFEST
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                obj.confidence_threshold = float(
                    manifest.get("confidence_threshold", obj.confidence_threshold)
                )
                obj.model_id = str(manifest.get("model_id", obj.model_id))
            except Exception:
                pass
        return obj

    def count_parameters(self) -> int:
        if self._model is None:
            return 0
        try:
            return sum(p.numel() for p in self._model.model_body.parameters())
        except Exception:
            return 0
