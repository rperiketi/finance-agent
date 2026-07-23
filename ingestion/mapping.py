"""
Column inference and optional user overrides for transaction CSVs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import pandas as pd

from ingestion.schema import (
    AMOUNT_ALIASES,
    CATEGORY_ALIASES,
    CREDIT_COLUMN_ALIASES,
    DATE_ALIASES,
    DEBIT_COLUMN_ALIASES,
    DESC_ALIASES,
    TYPE_ALIASES,
    _normalise_col_name,
)


@dataclass
class InferredColumnMapping:
    """Detected or user-selected CSV column names (original casing preserved)."""

    date: str | None = None
    description: str | None = None
    amount: str | None = None
    type: str | None = None
    category: str | None = None
    debit_column: str | None = None
    credit_column: str | None = None
    method: dict[str, str] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)


def _norm_map(df: pd.DataFrame) -> dict[str, str]:
    return {_normalise_col_name(c): c for c in df.columns}


def _find_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    """Exact match on normalised name, then fuzzy substring match."""
    nm = _norm_map(df)
    for alias in aliases:
        if alias in nm:
            return nm[alias]
    for norm, orig in nm.items():
        for alias in aliases:
            if alias in norm or norm in alias:
                return orig
    return None


def _score_date_column(series: pd.Series, dayfirst: bool | None) -> float:
    s = series.head(500)
    if s.empty:
        return 0.0
    parsed = pd.to_datetime(s, dayfirst=bool(dayfirst) if dayfirst is not None else False, errors="coerce")
    if dayfirst is None:
        parsed_alt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        ok = max(parsed.notna().mean(), parsed_alt.notna().mean())
    else:
        ok = parsed.notna().mean()
    return float(ok)


def _clean_numeric_sample(series: pd.Series, european: bool) -> pd.Series:
    s = series.astype(str)
    s = s.str.replace(r"[€£₹$]", "", regex=True).str.strip()
    if european:
        s = s.str.replace(r"\s", "", regex=True)
        s = s.str.replace(r"(?<=\d)\.(?=\d{3}(\D|$))", "", regex=True)
        s = s.str.replace(",", ".", regex=False)
    else:
        s = s.str.replace(r"[,$\s]", "", regex=True)
    s = s.str.replace(r"\((.+)\)", r"-\1", regex=True)
    return pd.to_numeric(s, errors="coerce")


def _score_amount_column(series: pd.Series, european: bool) -> float:
    num = _clean_numeric_sample(series.head(500), european)
    frac = float(num.notna().sum() / max(len(num), 1))
    if frac < 0.45:
        return frac * 0.4
    vals = num.dropna().abs()
    if vals.empty:
        return 0.0
    vmax = float(vals.max())
    if vmax > 1e10:
        return frac * 0.35
    return frac


def _score_description_column(series: pd.Series) -> float:
    if series.dtype != object and not pd.api.types.is_string_dtype(series):
        return 0.0
    s = series.astype(str).head(350)
    s = s[s.str.lower() != "nan"]
    if s.empty:
        return 0.0
    lens = s.str.len()
    uniq = s.nunique() / len(s)
    return float(min(1.0, lens.mean() / 35.0) * 0.55 + min(1.0, uniq) * 0.45)


def infer_transaction_columns(
    raw: pd.DataFrame,
    *,
    dayfirst: bool | None = None,
    european_decimal: bool = False,
) -> InferredColumnMapping:
    raw = raw.copy()
    raw.columns = raw.columns.str.strip()
    out = InferredColumnMapping()
    method: dict[str, str] = {}
    scores: dict[str, float] = {}

    date_alias = _find_col(raw, DATE_ALIASES)
    desc_alias = _find_col(raw, DESC_ALIASES)
    type_alias = _find_col(raw, TYPE_ALIASES)
    cat_alias = _find_col(raw, CATEGORY_ALIASES)
    debit_a = _find_col(raw, DEBIT_COLUMN_ALIASES)
    credit_a = _find_col(raw, CREDIT_COLUMN_ALIASES)

    single_amt_aliases = [
        a for a in AMOUNT_ALIASES
        if a not in set(DEBIT_COLUMN_ALIASES + CREDIT_COLUMN_ALIASES)
    ]
    amount_alias = _find_col(raw, single_amt_aliases)

    bd_name, bd_score = None, -1.0
    ba_name, ba_score = None, -1.0
    bdes_name, bdes_score = None, -1.0

    for col in raw.columns:
        ser = raw[col]
        ds = _score_date_column(ser, dayfirst)
        if ds > bd_score:
            bd_name, bd_score = col, ds
        ascr = _score_amount_column(ser, european_decimal)
        if ascr > ba_score:
            ba_name, ba_score = col, ascr
        dscr = _score_description_column(ser)
        if dscr > bdes_score:
            bdes_name, bdes_score = col, dscr

    if date_alias:
        out.date = date_alias
        method["date"] = "alias"
        scores["date"] = 1.0
    elif bd_name:
        out.date = bd_name
        method["date"] = "inferred"
        scores["date"] = float(bd_score)

    if debit_a and credit_a:
        out.debit_column = debit_a
        out.credit_column = credit_a
        out.amount = None
        method["amount"] = "split_debit_credit"
        scores["amount"] = 1.0
    elif amount_alias:
        out.amount = amount_alias
        method["amount"] = "alias"
        scores["amount"] = 1.0
    elif ba_name and ba_score >= 0.45 and ba_name != out.date:
        out.amount = ba_name
        method["amount"] = "inferred"
        scores["amount"] = float(ba_score)
    elif ba_name and ba_name != out.date:
        out.amount = ba_name
        method["amount"] = "inferred_low_confidence"
        scores["amount"] = float(ba_score)

    if desc_alias:
        out.description = desc_alias
        method["description"] = "alias"
        scores["description"] = 1.0
    elif bdes_name:
        if bdes_name in (out.date, out.amount):
            others = [
                c for c in raw.columns
                if c not in (out.date, out.amount, out.debit_column, out.credit_column)
            ]
            best_alt, best_alt_s = None, -1.0
            for c in others:
                sc = _score_description_column(raw[c])
                if sc > best_alt_s:
                    best_alt, best_alt_s = c, sc
            out.description = best_alt or bdes_name
        else:
            out.description = bdes_name
        method["description"] = "inferred"
        scores["description"] = float(bdes_score)

    if type_alias:
        out.type = type_alias
        method["type"] = "alias"
    if cat_alias:
        out.category = cat_alias
        method["category"] = "alias"

    out.method = method
    out.scores = scores
    return out


def merge_user_mapping(
    inferred: InferredColumnMapping,
    user: dict[str, str | None] | None,
) -> InferredColumnMapping:
    if not user:
        return inferred
    out = replace(
        inferred,
        method=dict(inferred.method),
        scores=dict(inferred.scores),
    )

    def pick(key: str) -> str | None:
        val = user.get(key)
        if val and str(val).strip() and str(val).lower() != "auto":
            return str(val).strip()
        return None

    for key in ("date", "description", "type", "category"):
        v = pick(key)
        if v:
            setattr(out, key, v)
            out.method[key] = "user"

    explicit_split = (
        pick("debit_column") is not None and pick("credit_column") is not None
    )
    want_split = user.get("amount_mode") == "split" or bool(explicit_split)

    amt = pick("amount")
    dc, cc = pick("debit_column"), pick("credit_column")

    if want_split and dc and cc:
        out.debit_column = dc
        out.credit_column = cc
        out.amount = None
        out.method["amount"] = "user_split"
    elif want_split:
        pass
    elif amt:
        out.amount = amt
        out.debit_column = out.credit_column = None
        out.method["amount"] = "user_single"

    return out


def mapping_to_report_dict(m: InferredColumnMapping) -> dict:
    return {
        "date": m.date,
        "description": m.description,
        "amount": m.amount,
        "debit_column": m.debit_column,
        "credit_column": m.credit_column,
        "type": m.type,
        "category": m.category,
        "method": dict(m.method),
        "scores": dict(m.scores),
    }
