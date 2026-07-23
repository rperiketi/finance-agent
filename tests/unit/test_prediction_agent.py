import pytest

from langgraph_finance.agents.prediction_agent import predict


def _txn(date, amount, category="Other", description="TEST MERCHANT"):
    """`amount` is the expense magnitude — stored signed negative internally,
    matching the (negative = expense) convention `predict` expects."""
    return {"date": date, "description": description, "amount": -abs(amount), "category": category}


def test_increasing_trend_forecast():
    transactions = [
        _txn("2026-01-05", 100.0, "Groceries"),
        _txn("2026-02-05", 110.0, "Groceries"),
        _txn("2026-03-05", 120.0, "Groceries"),
    ]

    result = predict(transactions)

    assert result["trend"] == "increasing"
    assert result["next_month_label"] == "2026-04"
    assert result["next_month_total"] > 120.0
    assert 0.0 < result["confidence"] <= 0.95
    assert result["category_forecasts"]["Groceries"] == pytest.approx(result["next_month_total"], abs=0.01)


def test_stable_trend_forecast():
    transactions = [
        _txn("2026-01-05", 50.0),
        _txn("2026-02-05", 50.0),
        _txn("2026-03-05", 50.0),
    ]

    result = predict(transactions)

    assert result["trend"] == "stable"
    assert result["next_month_total"] == pytest.approx(50.0, abs=0.5)


def test_decreasing_trend_forecast():
    transactions = [
        _txn("2026-01-05", 150.0),
        _txn("2026-02-05", 120.0),
        _txn("2026-03-05", 90.0),
    ]

    result = predict(transactions)

    assert result["trend"] == "decreasing"
    assert result["next_month_total"] < 90.0
    assert result["next_month_total"] >= 0.0  # forecast is clamped at zero


def test_single_month_is_insufficient_data():
    transactions = [_txn("2026-01-05", 75.0)]

    result = predict(transactions)

    assert result["trend"] == "insufficient_data"
    assert result["next_month_total"] == 75.0
    assert result["next_month_label"] == "2026-02"
    assert result["confidence"] == 0.3


def test_empty_transactions_returns_zeroed_result():
    result = predict([])

    assert result["next_month_total"] == 0.0
    assert result["trend"] == "unknown"
    assert result["category_forecasts"] == {}


def test_december_rollover_to_next_year():
    transactions = [
        _txn("2025-11-05", 100.0),
        _txn("2025-12-05", 110.0),
    ]

    result = predict(transactions)

    assert result["next_month_label"] == "2026-01"


def test_income_transactions_are_excluded_from_forecast():
    transactions = [
        _txn("2026-01-05", 50.0),
        _txn("2026-02-05", 50.0),
        _txn("2026-03-05", 50.0),
        {"date": "2026-01-10", "description": "SALARY", "amount": 3000.0, "category": "Income"},
    ]

    result = predict(transactions)

    assert result["trend"] == "stable"
    assert "Income" not in result["category_forecasts"]


def test_multiple_categories_forecast_independently():
    transactions = [
        _txn("2026-01-05", 100.0, "Groceries"),
        _txn("2026-02-05", 110.0, "Groceries"),
        _txn("2026-03-05", 120.0, "Groceries"),
        _txn("2026-01-06", 15.0, "Subscriptions"),
        _txn("2026-02-06", 15.0, "Subscriptions"),
        _txn("2026-03-06", 15.0, "Subscriptions"),
    ]

    result = predict(transactions)

    assert set(result["category_forecasts"].keys()) == {"Groceries", "Subscriptions"}
    assert result["category_forecasts"]["Groceries"] > result["category_forecasts"]["Subscriptions"]
