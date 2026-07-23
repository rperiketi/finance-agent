"""Ingestion agent: reads and validates raw financial-statement data from
almost any source.

Built to take *any* transaction export, not one fixed column layout or file
format:

- **Formats**: CSV/plain text, Excel (.xlsx/.xls), JSON (a list of records,
  or a dict wrapping one under a common key like "transactions"/"data"), or
  a remote URL pointing at any of the above.
- **Columns**: matched fuzzily (exact -> known alias -> substring), so
  `txn_description`, `transaction_date`, `debit_amount`, etc. all resolve.
- **Amount sign**: a single signed column (negative = expense, positive =
  income — the standard bank-statement convention), separate debit/credit
  columns, or a single *unsigned* amount column paired with a debit/credit
  type indicator column (e.g. a `movement` column with "debit"/"credit"
  values) are all detected and normalized to one signed `amount` per
  transaction. If no sign information exists at all, the dataset is assumed
  to be an expenditures-only export and every row is treated as an expense
  — a warning is attached so this assumption is visible rather than silent.
"""

from __future__ import annotations

import io
import json
import urllib.request
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from langgraph_finance.state import FinanceState, IngestionReport, Transaction

REQUIRED_COLUMNS = ("date", "description")

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "date": ("date", "transaction_date", "txn_date", "trans_date", "posted_date", "value_date"),
    "description": (
        "description", "desc", "narrative", "memo", "details", "payee",
        "merchant", "particulars", "transaction_description", "txn_description",
    ),
    "amount": ("amount", "amt", "value", "transaction_amount", "txn_amount", "total"),
    "debit": ("debit", "debit_amount", "withdrawal", "withdrawal_amt", "debit_amt", "dr_amount"),
    "credit": ("credit", "credit_amount", "deposit", "deposit_amt", "credit_amt", "cr_amount"),
    "type": ("type", "transaction_type", "txn_type", "movement", "dr_cr", "debit_credit", "direction"),
}

DEBIT_VALUES = {"debit", "dr", "withdrawal", "expense", "out", "d", "-"}
CREDIT_VALUES = {"credit", "cr", "deposit", "income", "in", "c", "+"}

JSON_LIST_WRAPPER_KEYS = ("transactions", "data", "results", "records", "items")
MAX_URL_BYTES = 25 * 1024 * 1024  # 25MB
URL_FETCH_TIMEOUT_SECONDS = 15

_CONTENT_TYPE_TO_EXT = {
    "text/csv": "csv",
    "application/csv": "csv",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/json": "json",
    "text/plain": "csv",
}


def _find_column(columns: list[str], target: str, taken: set[str]) -> str | None:
    lowered = {col.strip().lower(): col for col in columns if col not in taken}
    aliases = FIELD_ALIASES.get(target, (target,))

    # 1. exact match against target or a known alias
    for alias in (target, *aliases):
        if alias in lowered:
            return lowered[alias]

    # 2. substring containment (e.g. "txn_description" contains "description")
    for col_lower, original in lowered.items():
        if any(alias in col_lower for alias in aliases):
            return original

    return None


def _resolve_amount(
    raw_df: pd.DataFrame, columns: list[str], taken: set[str]
) -> tuple[pd.Series | None, dict[str, str], list[str]]:
    """Returns (signed amount series, columns used, warnings)."""
    warnings: list[str] = []
    mapping: dict[str, str] = {}

    amount_col = _find_column(columns, "amount", taken)
    if amount_col is not None:
        taken.add(amount_col)
        mapping["amount"] = amount_col
        amount = pd.to_numeric(raw_df[amount_col], errors="coerce")

        type_col = _find_column(columns, "type", taken)
        non_missing = amount.dropna()
        looks_unsigned = not non_missing.empty and (non_missing >= 0).all()

        if type_col is not None:
            taken.add(type_col)
            mapping["type"] = type_col
            if looks_unsigned:
                type_vals = raw_df[type_col].astype(str).str.strip().str.lower()
                is_debit = type_vals.isin(DEBIT_VALUES)
                is_credit = type_vals.isin(CREDIT_VALUES)
                signed = amount.abs()
                signed = signed.where(~is_debit, -signed)
                # anything not recognized as credit defaults to expense (debit)
                signed = signed.where(is_credit | is_debit, -signed.abs())
                amount = signed
            # if amounts already carry a sign, the type column is redundant — trust the sign.
        elif looks_unsigned:
            warnings.append(
                "No sign or debit/credit indicator found — assuming this is an "
                "expenditures-only export and treating every amount as an expense."
            )
            amount = -amount.abs()

        return amount, mapping, warnings

    debit_col = _find_column(columns, "debit", taken)
    credit_col = _find_column(columns, "credit", taken)
    if debit_col is not None or credit_col is not None:
        if debit_col is not None:
            taken.add(debit_col)
            mapping["debit"] = debit_col
        if credit_col is not None:
            taken.add(credit_col)
            mapping["credit"] = credit_col
        debit_vals = pd.to_numeric(raw_df[debit_col], errors="coerce") if debit_col else pd.Series(
            np.nan, index=raw_df.index
        )
        credit_vals = pd.to_numeric(raw_df[credit_col], errors="coerce") if credit_col else pd.Series(
            np.nan, index=raw_df.index
        )
        both_missing = debit_vals.isna() & credit_vals.isna()
        amount = credit_vals.fillna(0).abs() - debit_vals.fillna(0).abs()
        amount = amount.where(~both_missing, np.nan)
        return amount, mapping, warnings

    return None, mapping, warnings


def _ingest_dataframe(raw_df: pd.DataFrame) -> tuple[list[Transaction], IngestionReport]:
    """Validates and cleans an already-parsed DataFrame into transactions,
    regardless of what format it originally came from."""
    report = IngestionReport(
        success=False, rows_received=len(raw_df), rows_valid=0, rows_dropped=0,
        warnings=[], errors=[],
    )

    if raw_df.empty:
        report["errors"].append("No rows found in the provided data.")
        return [], report

    columns = [str(c) for c in raw_df.columns]
    taken: set[str] = set()

    date_col = _find_column(columns, "date", taken)
    if date_col:
        taken.add(date_col)
    desc_col = _find_column(columns, "description", taken)
    if desc_col:
        taken.add(desc_col)

    amount, amount_mapping, amount_warnings = _resolve_amount(raw_df, columns, taken)

    missing = []
    if date_col is None:
        missing.append("date")
    if desc_col is None:
        missing.append("description")
    if amount is None:
        missing.append("amount (or debit/credit columns)")
    if missing:
        report["errors"].append(
            f"Missing required column(s): {', '.join(missing)}. Found columns: {columns}"
        )
        return [], report

    report["column_mapping_used"] = {"date": date_col, "description": desc_col, **amount_mapping}
    report["warnings"].extend(amount_warnings)

    working = pd.DataFrame({
        "date": raw_df[date_col],
        "description": raw_df[desc_col],
        "amount_parsed": amount,
    })
    working["date_parsed"] = pd.to_datetime(working["date"], errors="coerce")
    description_missing = working["description"].isna()
    working["description"] = working["description"].astype(str).str.strip()

    valid_mask = (
        working["date_parsed"].notna()
        & working["amount_parsed"].notna()
        & ~description_missing
        & (working["description"] != "")
    )
    dropped = int((~valid_mask).sum())
    clean = working[valid_mask].copy()

    transactions: list[Transaction] = [
        Transaction(
            date=row.date_parsed.strftime("%Y-%m-%d"),
            description=row.description,
            amount=round(float(row.amount_parsed), 2),
        )
        for row in clean.itertuples(index=False)
    ]

    report["rows_valid"] = len(transactions)
    report["rows_dropped"] = dropped
    if dropped:
        report["warnings"].append(
            f"Dropped {dropped} of {report['rows_received']} row(s) with invalid/missing date, "
            "description, or amount."
        )
    if len(transactions) < 5:
        report["warnings"].append("Fewer than 5 valid transactions — results may be unreliable.")

    report["success"] = len(transactions) > 0
    if not report["success"]:
        report["errors"].append("No valid transactions remained after validation.")

    return transactions, report


def _dataframe_from_json_records(records) -> pd.DataFrame:
    if isinstance(records, list):
        return pd.DataFrame(records)
    if isinstance(records, dict):
        for key in JSON_LIST_WRAPPER_KEYS:
            if isinstance(records.get(key), list):
                return pd.DataFrame(records[key])
        return pd.DataFrame([records])
    raise ValueError("Expected a JSON list of records, or an object containing one.")


def _dataframe_from_bytes(data: bytes, filename: str | None) -> pd.DataFrame:
    ext = ""
    if filename and "." in filename:
        ext = filename.rsplit(".", 1)[-1].lower()

    if ext in ("xlsx", "xls"):
        return pd.read_excel(io.BytesIO(data))
    if ext == "json":
        return _dataframe_from_json_records(json.loads(data.decode("utf-8")))
    if ext in ("csv", "txt"):
        return pd.read_csv(io.StringIO(data.decode("utf-8", errors="replace")))

    # No (or unrecognized) extension — sniff the content itself.
    if data[:2] == b"PK":  # xlsx/xls are zip archives
        return pd.read_excel(io.BytesIO(data))
    text = data.decode("utf-8", errors="replace").strip()
    if text[:1] in ("[", "{"):
        return _dataframe_from_json_records(json.loads(text))
    return pd.read_csv(io.StringIO(text))


def _fetch_url(url: str) -> tuple[bytes | None, str | None, str | None]:
    """Returns (content_bytes, filename_hint, error_message)."""
    if not url.lower().startswith(("http://", "https://")):
        return None, None, "URL must start with http:// or https://"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "finance-agent/1.0"})
        with urllib.request.urlopen(req, timeout=URL_FETCH_TIMEOUT_SECONDS) as resp:
            content_type = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
            data = resp.read(MAX_URL_BYTES + 1)
    except Exception as exc:
        return None, None, f"Failed to fetch URL: {exc}"

    if len(data) > MAX_URL_BYTES:
        return None, None, f"Remote file exceeds the {MAX_URL_BYTES // (1024 * 1024)}MB limit."

    path = urlparse(url).path
    ext_from_url = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    ext_from_type = _CONTENT_TYPE_TO_EXT.get(content_type, "")
    filename_hint = f"remote.{ext_from_url or ext_from_type or 'csv'}"
    return data, filename_hint, None


def _empty_report(error: str) -> IngestionReport:
    return IngestionReport(
        success=False, rows_received=0, rows_valid=0, rows_dropped=0,
        warnings=[], errors=[error],
    )


def ingest(csv_text: str) -> tuple[list[Transaction], IngestionReport]:
    """Ingests plain CSV text. Kept as a direct entry point (in addition to
    `ingest_from_source`) since it's the simplest, most common case."""
    if not csv_text or not csv_text.strip():
        return [], _empty_report("No CSV content provided.")

    try:
        raw_df = pd.read_csv(io.StringIO(csv_text))
    except Exception as exc:
        return [], _empty_report(f"Failed to parse CSV: {exc}")

    return _ingest_dataframe(raw_df)


def ingest_from_source(
    csv_text: str | None = None,
    file_bytes: bytes | None = None,
    filename: str | None = None,
    json_records: list | dict | None = None,
    source_url: str | None = None,
) -> tuple[list[Transaction], IngestionReport]:
    """General entry point: accepts CSV text, raw file bytes (CSV/Excel/
    JSON — sniffed from `filename` or content), inline JSON records, or a
    URL to fetch any of the above from."""
    if source_url:
        fetched_bytes, url_filename, err = _fetch_url(source_url)
        if err:
            return [], _empty_report(err)
        file_bytes = fetched_bytes
        filename = filename or url_filename

    if json_records is not None:
        try:
            raw_df = _dataframe_from_json_records(json_records)
        except Exception as exc:
            return [], _empty_report(f"Failed to parse JSON data: {exc}")
        return _ingest_dataframe(raw_df)

    if file_bytes is not None:
        try:
            raw_df = _dataframe_from_bytes(file_bytes, filename)
        except Exception as exc:
            return [], _empty_report(f"Failed to parse file: {exc}")
        return _ingest_dataframe(raw_df)

    if csv_text and csv_text.strip():
        return ingest(csv_text)

    return [], _empty_report(
        "No data provided. Supply a file, csv_text, json data, or a url."
    )


def ingestion_node(state: FinanceState) -> FinanceState:
    """LangGraph node wrapper around `ingest_from_source`."""
    transactions, report = ingest_from_source(
        csv_text=state.get("csv_text"),
        file_bytes=state.get("file_bytes"),
        filename=state.get("filename"),
        json_records=state.get("json_records"),
        source_url=state.get("source_url"),
    )
    errors = list(state.get("errors", []))
    errors.extend(report["errors"])
    return {
        **state,
        "transactions": transactions,
        "ingestion_report": report,
        "errors": errors,
    }
