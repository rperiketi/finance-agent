"""
Phase 5 – Ingestion Agent

Responsibility: Accept a file path or uploaded bytes, validate the CSV,
run the loader + cleaner pipeline, and return a clean DataFrame plus
a validation report.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import pandas as pd

from ingestion.loader import normalize_raw_dataframe, get_sample_data
from ingestion.mapping import (
    infer_transaction_columns,
    merge_user_mapping,
    mapping_to_report_dict,
)
from processing.cleaner import clean_transactions, monthly_summary


@dataclass
class IngestionReport:
    success: bool
    rows_raw: int = 0
    rows_clean: int = 0
    date_range: str = ""
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    column_mapping_used: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "OK" if self.success else "FAILED"
        lines = [
            f"[IngestionAgent] Status: {status}",
            f"  Rows loaded  : {self.rows_raw}",
            f"  Rows cleaned : {self.rows_clean}",
            f"  Date range   : {self.date_range}",
        ]
        if self.column_mapping_used:
            cm = self.column_mapping_used
            lines.append("  Column mapping:")
            for k in ("date", "description", "amount", "debit_column", "credit_column", "type", "category"):
                if cm.get(k):
                    lines.append(f"    {k}: {cm[k]}")
        for w in self.warnings:
            lines.append(f"  [W] {w}")
        for e in self.errors:
            lines.append(f"  [E] {e}")
        return "\n".join(lines)


class IngestionAgent:
    """
    Stateless agent that loads, validates, and cleans transaction data.

    Usage
    -----
        agent = IngestionAgent()
        df, report = agent.run("data/transactions.csv")
        df, report = agent.run(uploaded_bytes)   # from Streamlit uploader
        df, report = agent.run(None)             # use built-in sample data
    """

    def run(
        self,
        source: Union[str, Path, bytes, io.BytesIO, None],
        *,
        upload_name: str | None = None,
        column_mapping: dict[str, str | None] | None = None,
        dayfirst: bool | None = None,
        european_decimal: bool = False,
        amount_clip_mode: str = "quantile",
    ) -> tuple[pd.DataFrame, IngestionReport]:
        report = IngestionReport(success=False)

        if source is None:
            try:
                raw_df = get_sample_data()
                report.warnings.append("No file provided – using built-in sample data.")
                report.column_mapping_used = {"note": "sample data (canonical columns)"}
            except Exception as exc:
                report.errors.append(f"Sample load failed: {exc}")
                return pd.DataFrame(), report
            report.rows_raw = len(raw_df)
        elif isinstance(source, (bytes, io.BytesIO)):
            buf = io.BytesIO(source) if isinstance(source, bytes) else source
            buf.seek(0)
            name = (upload_name or "").lower()
            try:
                if name.endswith(".xlsx"):
                    raw_df = pd.read_excel(buf, engine="openpyxl")
                else:
                    raw_df = pd.read_csv(buf, low_memory=False)
                raw_df.columns = raw_df.columns.str.strip()
                inferred = infer_transaction_columns(
                    raw_df, dayfirst=dayfirst, european_decimal=european_decimal
                )
                merged = merge_user_mapping(inferred, column_mapping)
                report.column_mapping_used = mapping_to_report_dict(merged)
                raw_df = normalize_raw_dataframe(
                    raw_df,
                    merged,
                    dayfirst=dayfirst,
                    european_decimal=european_decimal,
                )
            except Exception as exc:
                report.errors.append(f"Load failed: {exc}")
                return pd.DataFrame(), report
        else:
            try:
                path = Path(source)
                if path.suffix.lower() == ".xlsx":
                    raw_file = pd.read_excel(path, engine="openpyxl")
                else:
                    raw_file = pd.read_csv(source, low_memory=False)
                raw_file.columns = raw_file.columns.str.strip()
                inferred = infer_transaction_columns(
                    raw_file, dayfirst=dayfirst, european_decimal=european_decimal
                )
                merged = merge_user_mapping(inferred, column_mapping)
                report.column_mapping_used = mapping_to_report_dict(merged)
                raw_df = normalize_raw_dataframe(
                    raw_file,
                    merged,
                    dayfirst=dayfirst,
                    european_decimal=european_decimal,
                )
            except Exception as exc:
                report.errors.append(f"Load failed: {exc}")
                return pd.DataFrame(), report

        report.rows_raw = len(raw_df)

        if raw_df.empty:
            report.errors.append("CSV is empty.")
            return pd.DataFrame(), report

        if len(raw_df) < 5:
            report.warnings.append("Very few transactions – results may be unreliable.")

        try:
            clean_df = clean_transactions(raw_df, amount_clip_mode=amount_clip_mode)
        except Exception as exc:
            report.errors.append(f"Cleaning failed: {exc}")
            return raw_df, report

        report.rows_clean = len(clean_df)
        if not clean_df.empty:
            dmin = clean_df["date"].min().strftime("%Y-%m-%d")
            dmax = clean_df["date"].max().strftime("%Y-%m-%d")
            report.date_range = f"{dmin} → {dmax}"

        if report.rows_clean < report.rows_raw * 0.5:
            report.warnings.append(
                f"High drop rate: {report.rows_raw - report.rows_clean} rows removed during cleaning."
            )

        meth = report.column_mapping_used.get("method") or {}
        for field_name, how in meth.items():
            if "low_confidence" in str(how):
                report.warnings.append(
                    f"Column '{field_name}' was auto-detected with low confidence ({how}). "
                    "Consider setting column mapping manually."
                )

        report.success = True
        return clean_df, report

    def get_monthly_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        return monthly_summary(df)
