"""Shared state passed between agents on the LangGraph graph.

Kept as plain JSON-serializable structures (dicts/lists of primitives) rather
than DataFrames so the same state can flow straight in/out of the Flask API.
"""

from __future__ import annotations

from typing import Any, TypedDict


class Transaction(TypedDict, total=False):
    date: str  # ISO format YYYY-MM-DD
    description: str
    amount: float  # signed: negative = expense/debit, positive = income/credit
    category: str
    category_confidence: float


class IngestionReport(TypedDict, total=False):
    success: bool
    rows_received: int
    rows_valid: int
    rows_dropped: int
    warnings: list[str]
    errors: list[str]
    column_mapping_used: dict[str, str]


class AnalysisResult(TypedDict, total=False):
    total_income: float
    total_expenses: float
    net_savings: float
    savings_rate_pct: float | None
    category_totals: dict[str, float]
    top_categories: list[dict[str, Any]]
    monthly_expenses: dict[str, float]
    monthly_income: dict[str, float]
    category_monthly_totals: dict[str, dict[str, float]]
    month_over_month_pct: float | None
    recurring_merchants: list[dict[str, Any]]
    insights: list[dict[str, str]]  # {"text": ..., "kind": "good"|"warning"|"neutral"}
    summary_text: str
    retrieved_context: list[str]


class PredictionResult(TypedDict, total=False):
    method: str
    next_month_total: float
    next_month_label: str
    trend: str
    confidence: float
    category_forecasts: dict[str, float]


class FinanceState(TypedDict, total=False):
    # Input — exactly one of these is expected to carry data; the ingestion
    # agent checks them in this order: source_url, file_bytes, json_records,
    # csv_text.
    csv_text: str
    file_bytes: bytes | None
    filename: str | None
    json_records: list | None
    source_url: str | None

    # Ingestion agent output
    transactions: list[Transaction]
    ingestion_report: IngestionReport

    # Categorization agent output
    categorized_transactions: list[Transaction]
    categorization_notes: list[str]

    # Analysis agent output
    analysis: AnalysisResult

    # Prediction agent output
    prediction: PredictionResult

    # Cross-cutting
    errors: list[str]


def new_state(
    csv_text: str | None = None,
    *,
    file_bytes: bytes | None = None,
    filename: str | None = None,
    json_records: list | None = None,
    source_url: str | None = None,
) -> FinanceState:
    return FinanceState(
        csv_text=csv_text or "",
        file_bytes=file_bytes,
        filename=filename,
        json_records=json_records,
        source_url=source_url,
        errors=[],
    )
