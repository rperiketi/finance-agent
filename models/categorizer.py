"""
Phase 2 – Expense Categorization

Tier 1: Rule-based keywords
Tier 2: Word + character TF-IDF + Logistic Regression
Tier 3: Low-confidence → Other
Supports optional CSV `source_category` column mapping onto canonical labels.
"""

import re
import pickle
import warnings
from difflib import get_close_matches
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

CATEGORIES = [
    "Food & Dining", "Transport", "Shopping", "Entertainment",
    "Health & Fitness", "Utilities", "Subscriptions", "Income",
    "ATM / Cash", "Other",
]

# Normalised synonyms from bank exports → canonical bucket
SYNONYMS: dict[str, str] = {
    "groceries": "Food & Dining",
    "grocery": "Food & Dining",
    "food": "Food & Dining",
    "restaurant": "Food & Dining",
    "fuel": "Transport",
    "gas": "Transport",
    "automotive": "Transport",
    "retail": "Shopping",
    "telecom": "Utilities",
    "phone": "Utilities",
    "internet": "Utilities",
    "rent": "Utilities",
    "utilities": "Utilities",
    "transfer": "Other",
    "internal": "Other",
    "travel": "Transport",
}

RULES: dict[str, str] = {
    r"starbucks|coffee|cafe|bakery|pizza|burger|mcdonald|kfc|subway|chipotle"
    r"|restaurant|dining|doordash|grubhub|uber\s*eats|zomato|swiggy|domino|taco": "Food & Dining",
    r"\buber\b(?!\s*eats)|lyft|taxi|cab|ola|rapido|metro|bus|train|airline"
    r"|flight|fuel|gas\s*station|shell|chevron|bp\b|exxon|petrol": "Transport",
    r"amazon|walmart|target|costco|ebay|flipkart|myntra|ikea|h&m|zara"
    r"|online\s*shop|mall|store|market": "Shopping",
    r"netflix|spotify|hulu|disney|prime\s*video|youtube|cinema|movie|theatre"
    r"|concert|gaming|steam|xbox|playstation": "Entertainment",
    r"gym|fitness|yoga|pharmacy|cvs|walgreen|hospital|clinic|doctor|medical"
    r"|health|dental|vision": "Health & Fitness",
    r"electric|electricity|water\s*bill|gas\s*bill|internet|broadband|wifi"
    r"|telephone|mobile\s*bill|utility|rent|mortgage": "Utilities",
    r"subscription|monthly\s*plan|annual\s*plan|membership|adobe|microsoft"
    r"|dropbox|icloud|google\s*one": "Subscriptions",
    r"salary|payroll|direct\s*deposit|income|bonus|dividend|refund|cashback"
    r"|transfer\s*in|credit\s*received|freelance": "Income",
    r"atm|cash\s*withdrawal|withdraw": "ATM / Cash",
}


def rule_based_category(description: str) -> Optional[str]:
    text = str(description).lower()
    for pattern, category in RULES.items():
        if re.search(pattern, text, re.IGNORECASE):
            return category
    return None


def map_source_category_label(raw: str | float | None) -> Optional[str]:
    """Map a CSV category string to canonical CATEGORIES, or None if unknown / unset."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    s = str(raw).strip()
    if not s or s.lower() in {"", "nan", "none"}:
        return None
    low = s.lower()
    if low in SYNONYMS:
        return SYNONYMS[low]
    for cat in CATEGORIES:
        cl = cat.lower()
        tokens = set(cl.replace("&", " ").replace("/", " ").split())
        if low == cl or low in tokens:
            return cat
    cat_keys = [c.lower() for c in CATEGORIES]
    syn_keys = list(SYNONYMS.keys())
    close = get_close_matches(low, cat_keys + syn_keys, n=1, cutoff=0.72)
    if close:
        m = close[0]
        for cat in CATEGORIES:
            if m == cat.lower():
                return cat
        for k, v in SYNONYMS.items():
            if m == k:
                return v
    return None


_SEED_DATA: list[tuple[str, str]] = [
    ("Starbucks Coffee", "Food & Dining"), ("McDonald's Drive Thru", "Food & Dining"),
    ("Pizza Hut Delivery", "Food & Dining"), ("Local Restaurant", "Food & Dining"),
    ("Grocery Store Deli", "Food & Dining"), ("Panera Bread", "Food & Dining"),
    ("Chipotle Mexican Grill", "Food & Dining"), ("Dunkin Donuts", "Food & Dining"),
    ("Whole Foods Market", "Food & Dining"), ("Trader Joe's", "Food & Dining"),
    ("Uber Trip", "Transport"), ("Lyft Ride", "Transport"),
    ("Shell Gas Station", "Transport"), ("ExxonMobil Fuel", "Transport"),
    ("City Metro Pass", "Transport"), ("Delta Airlines", "Transport"),
    ("Chevron Gas", "Transport"), ("Bus Pass Renewal", "Transport"),
    ("Amazon Purchase", "Shopping"), ("Walmart Supercenter", "Shopping"),
    ("Target Store", "Shopping"), ("eBay Online", "Shopping"),
    ("Best Buy Electronics", "Shopping"), ("Costco Wholesale", "Shopping"),
    ("IKEA Home Furnishings", "Shopping"), ("H&M Clothing", "Shopping"),
    ("Netflix Monthly", "Entertainment"), ("Spotify Premium", "Entertainment"),
    ("Hulu Subscription", "Entertainment"), ("AMC Movie Tickets", "Entertainment"),
    ("Steam Game Purchase", "Entertainment"), ("Disney+ Plan", "Entertainment"),
    ("YouTube Premium", "Entertainment"), ("Concert Tickets", "Entertainment"),
    ("Planet Fitness Gym", "Health & Fitness"), ("CVS Pharmacy", "Health & Fitness"),
    ("Walgreens Drug Store", "Health & Fitness"), ("Doctor Visit Copay", "Health & Fitness"),
    ("Yoga Studio Membership", "Health & Fitness"), ("Dental Clinic", "Health & Fitness"),
    ("Electric Company Bill", "Utilities"), ("Water Department", "Utilities"),
    ("Comcast Internet", "Utilities"), ("Verizon Mobile Bill", "Utilities"),
    ("Apartment Rent", "Utilities"), ("Gas Utility Bill", "Utilities"),
    ("Adobe Creative Cloud", "Subscriptions"), ("Microsoft 365", "Subscriptions"),
    ("Dropbox Plus Plan", "Subscriptions"), ("iCloud Storage", "Subscriptions"),
    ("Gym Annual Membership", "Subscriptions"),
    ("Salary Direct Deposit", "Income"), ("Freelance Payment Received", "Income"),
    ("Tax Refund", "Income"), ("Dividend Payment", "Income"),
    ("Cashback Reward", "Income"), ("Transfer In", "Income"),
    ("ATM Withdrawal", "ATM / Cash"), ("Cash Withdrawal ATM", "ATM / Cash"),
    ("Bank ATM Cash", "ATM / Cash"),
    ("Miscellaneous Charge", "Other"), ("Unknown Vendor", "Other"),
    ("Bank Fee", "Other"), ("Service Charge", "Other"),
]


def _make_pipeline():
    feats = FeatureUnion(
        transformer_list=[
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=9000,
                    sublinear_tf=True,
                ),
            ),
            (
                "char_wb",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 4),
                    min_df=1,
                    max_features=9000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    return Pipeline([
        ("features", feats),
        (
            "clf",
            LogisticRegression(
                max_iter=1000,
                C=5.0,
                class_weight="balanced",
                solver="lbfgs",
                multi_class="multinomial",
            ),
        ),
    ])


class ExpenseCategorizer:

    MODEL_PATH = Path(__file__).parent / "categorizer_model_v2.pkl"

    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self._load_or_train()

    def _load_or_train(self):
        if self.MODEL_PATH.exists():
            try:
                with open(self.MODEL_PATH, "rb") as f:
                    self.pipeline = pickle.load(f)
            except Exception:
                self.train(_SEED_DATA)
        else:
            self.train(_SEED_DATA)

    def save(self):
        with open(self.MODEL_PATH, "wb") as f:
            pickle.dump(self.pipeline, f)

    def train(self, labeled_data: list[tuple[str, str]], verbose: bool = False):
        all_data = list(set(labeled_data + _SEED_DATA))
        texts, labels = zip(*all_data)

        self.pipeline = _make_pipeline()

        if len(set(labels)) > 1 and len(texts) > 10:
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=0.2, random_state=42,
                stratify=labels if min(pd.Series(labels).value_counts()) > 1 else None,
            )
            self.pipeline.fit(X_train, y_train)
            if verbose:
                y_pred = self.pipeline.predict(X_test)
                print(classification_report(y_test, y_pred))
        else:
            self.pipeline.fit(texts, labels)

        self.save()

    def ml_max_confidence_prediction(self, description: str) -> tuple[float, str]:
        if not self.pipeline:
            return 0.0, "Other"
        proba = self.pipeline.predict_proba([description])[0]
        best_idx = int(np.argmax(proba))
        return float(proba[best_idx]), str(self.pipeline.classes_[best_idx])

    def predict_one(self, description: str, confidence_threshold: float = 0.22) -> str:
        rule_cat = rule_based_category(description)
        if rule_cat:
            return rule_cat
        conf, lbl = self.ml_max_confidence_prediction(description)
        if conf >= confidence_threshold:
            return lbl
        return "Other"

    def predict_batch(self, descriptions: pd.Series) -> pd.Series:
        return descriptions.apply(lambda d: self.predict_one(str(d)))

    def categorize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        text_col = "description" if "description" in df.columns else "merchant"
        texts = df[text_col].astype(str)

        if "source_category" in df.columns:
            preset = df["source_category"].apply(map_source_category_label)
        else:
            preset = pd.Series(np.nan, index=df.index)

        df["category"] = preset
        need_ml = df["category"].isna()
        if need_ml.any():
            df.loc[need_ml, "category"] = texts[need_ml].map(self.predict_one)
        return df

    def uncertain_rows(self, df: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
        """Descriptions with lowest ML confidence (rules ignored for ranking)."""
        col = "description" if "description" in df.columns else "merchant"
        scored = []
        for _, row in df.iterrows():
            text = str(row[col])
            conf, ml_guess = self.ml_max_confidence_prediction(text)
            scored.append({"description": text, "ml_confidence": conf, "ml_guess": ml_guess})

        sdf = pd.DataFrame(scored)
        return sdf.sort_values("ml_confidence").head(top_k)

    def category_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        expenses = df[df["amount"] < 0].copy()
        expenses["amount"] = expenses["amount"].abs()
        summary = (
            expenses.groupby("category")["amount"]
            .agg(["sum", "count", "mean"])
            .rename(columns={"sum": "total", "count": "transactions", "mean": "avg_per_txn"})
            .sort_values("total", ascending=False)
            .reset_index()
        )
        if summary.empty:
            return summary
        summary["pct_of_total"] = (summary["total"] / summary["total"].sum() * 100).round(1)
        return summary
