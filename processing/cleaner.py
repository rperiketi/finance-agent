"""
Phase 1 – Data Pipeline: Clean and standardise transaction descriptions
and produce the canonical DataFrame used downstream.

Output schema:
    date        – Timestamp
    merchant    – cleaned merchant name
    amount      – float (negative = expense, positive = income)
    type        – 'debit' | 'credit'
    description – original raw description (kept for ML)
    year        – int
    month       – int
    week        – int
    day_of_week – int (0=Mon … 6=Sun)
    source_category – optional labels from CSV
    is_transfer – heuristic P2P / transfer flag
"""

import re
import pandas as pd


# ── Merchant cleaning rules ───────────────────────────────────────────────────
_STRIP_PATTERNS = [
    r"\bpurchase\b", r"\btransaction\b", r"\bpayment\b",
    r"\bpos\b", r"\batm\b", r"\bcard\b",
    r"\d{4,}",          # long numeric sequences (card/ref numbers)
    r"#\w+",            # reference tags
    r"\*+\w*",          # asterisk sequences
    r"\s{2,}",          # multiple spaces
]

_MERCHANT_MAP = {
    r"starbucks":       "Starbucks",
    r"uber\s*(eats)?":  "Uber",
    r"lyft":            "Lyft",
    r"amazon":          "Amazon",
    r"netflix":         "Netflix",
    r"spotify":         "Spotify",
    r"walmart":         "Walmart",
    r"target":          "Target",
    r"costco":          "Costco",
    r"whole\s*foods":   "Whole Foods",
    r"mcdonald":        "McDonald's",
    r"chipotle":        "Chipotle",
    r"domino":          "Domino's",
    r"doordash":        "DoorDash",
    r"grubhub":         "GrubHub",
    r"instacart":       "Instacart",
    r"apple\s*(store|pay|inc)": "Apple",
    r"google\s*(pay|llc|store)": "Google",
    r"microsoft":       "Microsoft",
    r"hulu":            "Hulu",
    r"disney\+?":       "Disney+",
    r"venmo":           "Venmo",
    r"paypal":          "PayPal",
    r"zelle":           "Zelle",
    r"cvs":             "CVS Pharmacy",
    r"walgreen":        "Walgreens",
    r"shell":           "Shell Gas",
    r"chevron":         "Chevron",
    r"bp\b":            "BP Gas",
    r"exxon":           "ExxonMobil",
    r"salary|payroll|direct\s*deposit": "Salary",
}

_TRANSFER_PATTERN = re.compile(
    r"zelle|venmo|paypal|cash\s*app|xfer|transfer\s*to|\btransfer\b|payment\s+to",
    re.IGNORECASE,
)


def clean_merchant(raw: str) -> str:
    """Return a normalised merchant name from a raw transaction description."""
    text = str(raw).lower().strip()

    for pattern in _STRIP_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    text = text.strip()

    for pattern, canonical in _MERCHANT_MAP.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            return canonical

    words = text.split()
    return " ".join(w.capitalize() for w in words[:4]) or "Unknown"


def enrich_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"]        = df["date"].dt.year
    df["month"]       = df["date"].dt.month
    df["week"]        = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["date"].dt.dayofweek
    return df


def _clip_amounts(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "none":
        return df

    if mode == "iqr":
        df = df.copy()
        neg = df["amount"] < 0
        pos = df["amount"] > 0
        if neg.any():
            a = df.loc[neg, "amount"].abs()
            if len(a.dropna()) >= 4:
                q1, q3 = a.quantile(0.25), a.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    cap = q3 + 1.5 * iqr
                    df.loc[neg, "amount"] = -a.clip(lower=0.01, upper=cap)
        if pos.any():
            a = df.loc[pos, "amount"]
            if len(a.dropna()) >= 4:
                q1, q3 = a.quantile(0.25), a.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    lo = max(0.01, q1 - 1.5 * iqr)
                    hi = q3 + 1.5 * iqr
                    df.loc[pos, "amount"] = a.clip(lower=lo, upper=hi)
        return df

    df = df.copy()
    q_low = df["amount"].quantile(0.01)
    q_high = df["amount"].quantile(0.99)
    df["amount"] = df["amount"].clip(lower=q_low, upper=q_high)
    return df


def clean_transactions(df: pd.DataFrame, *, amount_clip_mode: str = "quantile") -> pd.DataFrame:
    """
    Accept the raw loaded DataFrame and return the fully cleaned one.
    amount_clip_mode: 'quantile' (default), 'none', 'iqr'
    """
    df = df.copy()

    if "description" not in df.columns:
        df["description"] = "unknown"

    df["merchant"] = df["description"].apply(clean_merchant)

    text_blob = df["description"].astype(str) + " " + df["merchant"].astype(str)
    df["is_transfer"] = text_blob.apply(lambda s: bool(_TRANSFER_PATTERN.search(s)))

    df = _clip_amounts(df, amount_clip_mode.strip().lower())

    df = enrich_dates(df)

    base = ["date", "merchant", "description", "amount", "type"]
    extras = []
    if "source_category" in df.columns:
        extras.append("source_category")
    extras.extend(["year", "month", "week", "day_of_week", "is_transfer"])
    cols = base + extras
    return df[[c for c in cols if c in df.columns]]


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["year", "month"])
    income   = grp.apply(lambda x: x.loc[x["amount"] > 0, "amount"].sum())
    expenses = grp.apply(lambda x: x.loc[x["amount"] < 0, "amount"].sum().abs())
    summary = pd.DataFrame({"income": income, "expenses": expenses}).reset_index()
    summary["net_savings"] = summary["income"] - summary["expenses"]
    summary["period"] = pd.to_datetime(
        summary["year"].astype(str) + "-" + summary["month"].astype(str).str.zfill(2)
    )
    return summary.sort_values("period").reset_index(drop=True)
