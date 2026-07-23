"""Prediction agent: forecasts next month's total and per-category spend.

Uses deterministic linear-trend extrapolation over monthly totals rather than
an LLM call, so forecasts are reproducible and cheap to unit test. Confidence
is derived from trend fit quality (R^2) and the number of months observed.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from langgraph_finance.state import FinanceState, PredictionResult, Transaction


def _month_key(date_str: str) -> str:
    return date_str[:7]  # YYYY-MM


def _next_month_label(months: list[str]) -> str:
    last = months[-1]
    year, month = int(last[:4]), int(last[5:7])
    if month == 12:
        year, month = year + 1, 1
    else:
        month += 1
    return f"{year:04d}-{month:02d}"


def _forecast_series(monthly_values: list[float]) -> tuple[float, str, float]:
    """Returns (forecast, trend_label, confidence) for one monthly series."""
    n = len(monthly_values)
    if n == 0:
        return 0.0, "unknown", 0.0
    if n == 1:
        return round(monthly_values[0], 2), "insufficient_data", 0.3

    x = np.arange(n)
    y = np.array(monthly_values, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    forecast = max(slope * n + intercept, 0.0)

    mean_y = y.mean()
    ss_tot = ((y - mean_y) ** 2).sum()
    if ss_tot == 0:
        r2 = 1.0
    else:
        y_pred = slope * x + intercept
        ss_res = ((y - y_pred) ** 2).sum()
        r2 = max(0.0, 1 - ss_res / ss_tot)
    confidence = round(min(0.95, 0.4 + 0.5 * r2 + 0.05 * min(n, 6)), 2)

    if abs(slope) < 0.01 * (mean_y or 1):
        trend = "stable"
    elif slope > 0:
        trend = "increasing"
    else:
        trend = "decreasing"

    return round(float(forecast), 2), trend, confidence


def predict(transactions: list[Transaction]) -> PredictionResult:
    result = PredictionResult(
        method="linear_trend",
        next_month_total=0.0,
        next_month_label="",
        trend="unknown",
        confidence=0.0,
        category_forecasts={},
    )
    expenses = [t for t in transactions if float(t["amount"]) < 0]
    if not expenses:
        return result

    monthly_totals: dict[str, float] = defaultdict(float)
    monthly_by_category: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for t in expenses:
        month = _month_key(t["date"])
        amt = abs(float(t["amount"]))
        monthly_totals[month] += amt
        monthly_by_category[t.get("category") or "Other"][month] += amt

    months_sorted = sorted(monthly_totals.keys())
    total_series = [monthly_totals[m] for m in months_sorted]
    forecast_total, trend, confidence = _forecast_series(total_series)

    result["next_month_total"] = forecast_total
    result["next_month_label"] = _next_month_label(months_sorted)
    result["trend"] = trend
    result["confidence"] = confidence

    category_forecasts: dict[str, float] = {}
    for category, by_month in monthly_by_category.items():
        series = [by_month.get(m, 0.0) for m in months_sorted]
        cat_forecast, _, _ = _forecast_series(series)
        category_forecasts[category] = cat_forecast
    result["category_forecasts"] = category_forecasts

    return result


def prediction_node(state: FinanceState) -> FinanceState:
    """LangGraph node wrapper around `predict`."""
    transactions = state.get("categorized_transactions") or state.get("transactions", [])
    prediction = predict(transactions)
    return {**state, "prediction": prediction}
