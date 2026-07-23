"""
Phase 1 : Data Pipeline: Load and parse raw bank transaction CSV.
Supports multiple column naming conventions and user overrides.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from ingestion.mapping import InferredColumnMapping
from ingestion.schema import CREDIT_COLUMN_ALIASES, DEBIT_COLUMN_ALIASES, _normalise_col_name


def _find_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    nm = {_normalise_col_name(c): c for c in df.columns}
    for alias in aliases:
        if alias in nm:
            return nm[alias]
    for norm, orig in nm.items():
        for alias in aliases:
            if alias in norm or norm in alias:
                return orig
    return None


def clean_numeric_series(series: pd.Series, european_decimal: bool = False) -> pd.Series:
    s = series.astype(str)
    s = s.str.replace(r"[€£₹$]", "", regex=True).str.strip()
    if european_decimal:
        s = s.str.replace(r"\s", "", regex=True)
        s = s.str.replace(r"(?<=\d)\.(?=\d{3}(\D|$))", "", regex=True)
        s = s.str.replace(",", ".", regex=False)
    else:
        s = s.str.replace(r"[,$\s]", "", regex=True)
    s = s.str.replace(r"\((.+)\)", r"-\1", regex=True)
    return pd.to_numeric(s, errors="coerce")


def normalize_raw_dataframe(
    raw: pd.DataFrame,
    mapping: InferredColumnMapping,
    *,
    dayfirst: bool | None = None,
    european_decimal: bool = False,
) -> pd.DataFrame:
    """
    Build canonical transaction DataFrame: date, description, amount, type;
    optional source_category if mapping.category is set.
    """
    raw = raw.copy()
    raw.columns = raw.columns.str.strip()

    if not mapping.date:
        raise ValueError(
            f"Could not find a date column. Columns: {list(raw.columns)}"
        )

    df = pd.DataFrame()
    parse_kw: dict = {"errors": "coerce"}
    if dayfirst is not None:
        df["date"] = pd.to_datetime(raw[mapping.date], dayfirst=dayfirst, **parse_kw)
    else:
        # Try both orientations; pick series with higher parse rate
        d1 = pd.to_datetime(raw[mapping.date], dayfirst=False, **parse_kw)
        d2 = pd.to_datetime(raw[mapping.date], dayfirst=True, **parse_kw)
        df["date"] = d1 if d1.notna().sum() >= d2.notna().sum() else d2

    if mapping.description and mapping.description in raw.columns:
        df["description"] = raw[mapping.description].astype(str).str.strip()
    else:
        str_cols = raw.select_dtypes(include=["object", "string"]).columns.tolist()
        fallback = str_cols[0] if str_cols else None
        if fallback is not None:
            df["description"] = raw[fallback].astype(str).str.strip()
        else:
            df["description"] = "unknown"

    if mapping.debit_column and mapping.credit_column:
        debit = clean_numeric_series(raw[mapping.debit_column], european_decimal).fillna(0)
        credit = clean_numeric_series(raw[mapping.credit_column], european_decimal).fillna(0)
        df["amount"] = credit - debit
    elif mapping.amount and mapping.amount in raw.columns:
        df["amount"] = clean_numeric_series(raw[mapping.amount], european_decimal)
    else:
        debit_c = _find_col(raw, DEBIT_COLUMN_ALIASES)
        credit_c = _find_col(raw, CREDIT_COLUMN_ALIASES)
        if debit_c and credit_c:
            debit = clean_numeric_series(raw[debit_c], european_decimal).fillna(0)
            credit = clean_numeric_series(raw[credit_c], european_decimal).fillna(0)
            df["amount"] = credit - debit
        else:
            num_cols = raw.select_dtypes(include="number").columns.tolist()
            if num_cols:
                df["amount"] = pd.to_numeric(raw[num_cols[0]], errors="coerce")
            else:
                raise ValueError(
                    f"Could not find amount columns. Found: {list(raw.columns)}"
                )

    if mapping.type and mapping.type in raw.columns:
        tc = mapping.type
        df["type"] = (
            raw[tc].astype(str).str.strip().str.lower().map(
                lambda t: (
                    "credit"
                    if "credit" in t or t == "cr"
                    else ("debit" if "debit" in t or t == "dr" else t)
                )
            )
        )
    else:
        df["type"] = np.where(df["amount"] >= 0, "credit", "debit")

    mask_debit = df["type"] == "debit"
    df.loc[mask_debit & (df["amount"] > 0), "amount"] *= -1

    mask_credit = df["type"] == "credit"
    df.loc[mask_credit & (df["amount"] < 0), "amount"] *= -1

    if mapping.category and mapping.category in raw.columns:
        df["source_category"] = raw[mapping.category].astype(str).str.strip()

    df.dropna(subset=["date", "amount"], inplace=True)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_transactions(
    filepath: Union[str, Path],
    *,
    column_mapping: InferredColumnMapping | None = None,
    dayfirst: bool | None = None,
    european_decimal: bool = False,
) -> pd.DataFrame:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV not found: {filepath}")

    raw = pd.read_csv(filepath, low_memory=False)
    if column_mapping is None:
        from ingestion.mapping import infer_transaction_columns

        column_mapping = infer_transaction_columns(
            raw, dayfirst=dayfirst, european_decimal=european_decimal
        )
    return normalize_raw_dataframe(
        raw, column_mapping, dayfirst=dayfirst, european_decimal=european_decimal
    )


def get_sample_data() -> pd.DataFrame:
    np.random.seed(42)
    merchants = [
        ("Starbucks Coffee", "debit"), ("Uber Ride", "debit"),
        ("Amazon Purchase", "debit"), ("Netflix Subscription", "debit"),
        ("Salary Deposit", "credit"), ("Walmart Groceries", "debit"),
        ("Electric Bill", "debit"), ("ATM Withdrawal", "debit"),
        ("Restaurant Dining", "debit"), ("Gym Membership", "debit"),
        ("Gas Station", "debit"), ("Freelance Payment", "credit"),
        ("Pharmacy", "debit"), ("Online Shopping", "debit"),
    ]
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="2D")
    rows = []
    for d in dates:
        m, t = merchants[np.random.randint(len(merchants))]
        amt = np.random.uniform(5, 500) if t == "debit" else np.random.uniform(500, 5000)
        rows.append({"date": d, "description": m,
                     "amount": -round(amt, 2) if t == "debit" else round(amt, 2),
                     "type": t})
    return pd.DataFrame(rows)
