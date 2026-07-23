import pytest

from langgraph_finance.agents.analysis_agent import analyze, compute_patterns, generate_summary
from tests.conftest import make_summary_fake_llm

CATEGORIZED_TRANSACTIONS = [
    {"date": "2026-01-05", "description": "ACME CORP PAYROLL", "amount": 3000.0, "category": "Income"},
    {"date": "2026-01-06", "description": "WHOLE FOODS MARKET #1234", "amount": -100.0, "category": "Groceries"},
    {"date": "2026-01-10", "description": "STARBUCKS STORE 4021", "amount": -50.0, "category": "Dining"},
    {"date": "2026-01-15", "description": "SHELL OIL 57443210", "amount": -40.0, "category": "Transportation"},
    {"date": "2026-01-20", "description": "PG&E ELECTRIC BILL", "amount": -60.0, "category": "Utilities"},
    {"date": "2026-01-25", "description": "NETFLIX.COM", "amount": -15.0, "category": "Subscriptions"},
    {"date": "2026-02-05", "description": "ACME CORP PAYROLL", "amount": 3000.0, "category": "Income"},
    {"date": "2026-02-06", "description": "WHOLE FOODS MARKET #1234", "amount": -110.0, "category": "Groceries"},
    {"date": "2026-02-10", "description": "STARBUCKS STORE 4021", "amount": -55.0, "category": "Dining"},
    {"date": "2026-02-15", "description": "SHELL OIL 57443210", "amount": -42.0, "category": "Transportation"},
    {"date": "2026-02-20", "description": "PG&E ELECTRIC BILL", "amount": -62.0, "category": "Utilities"},
    {"date": "2026-02-25", "description": "NETFLIX.COM", "amount": -15.0, "category": "Subscriptions"},
    {"date": "2026-03-05", "description": "ACME CORP PAYROLL", "amount": 3000.0, "category": "Income"},
    {"date": "2026-03-06", "description": "WHOLE FOODS MARKET #1234", "amount": -120.0, "category": "Groceries"},
    {"date": "2026-03-10", "description": "STARBUCKS STORE 4021", "amount": -60.0, "category": "Dining"},
    {"date": "2026-03-15", "description": "SHELL OIL 57443210", "amount": -44.0, "category": "Transportation"},
    {"date": "2026-03-20", "description": "PG&E ELECTRIC BILL", "amount": -64.0, "category": "Utilities"},
    {"date": "2026-03-25", "description": "NETFLIX.COM", "amount": -15.0, "category": "Subscriptions"},
]


def test_compute_patterns_income_and_expenses():
    result = compute_patterns(CATEGORIZED_TRANSACTIONS)

    assert result["total_income"] == 9000.0
    assert result["total_expenses"] == 852.0
    assert result["net_savings"] == 8148.0
    assert result["savings_rate_pct"] == pytest.approx(90.5, abs=0.1)


def test_compute_patterns_category_totals_exclude_income():
    result = compute_patterns(CATEGORIZED_TRANSACTIONS)

    assert "Income" not in result["category_totals"]
    assert result["category_totals"]["Groceries"] == 330.0
    assert result["category_totals"]["Subscriptions"] == 45.0
    assert result["top_categories"][0]["category"] == "Groceries"
    assert result["top_categories"][0]["pct_of_total"] == pytest.approx(38.7, abs=0.1)


def test_month_over_month_pct_uses_expenses_only():
    result = compute_patterns(CATEGORIZED_TRANSACTIONS)
    assert result["month_over_month_pct"] == pytest.approx(6.7, abs=0.1)


def test_monthly_income_and_expenses_tracked_separately():
    result = compute_patterns(CATEGORIZED_TRANSACTIONS)

    assert result["monthly_income"] == {"2026-01": 3000.0, "2026-02": 3000.0, "2026-03": 3000.0}
    assert result["monthly_expenses"]["2026-01"] == 265.0
    assert result["monthly_expenses"]["2026-02"] == 284.0
    assert result["monthly_expenses"]["2026-03"] == 303.0


def test_category_monthly_totals_cover_every_month_per_category():
    result = compute_patterns(CATEGORIZED_TRANSACTIONS)

    groceries = result["category_monthly_totals"]["Groceries"]
    assert groceries == {"2026-01": 100.0, "2026-02": 110.0, "2026-03": 120.0}
    assert "Income" not in result["category_monthly_totals"]


def test_recurring_merchants_detected():
    result = compute_patterns(CATEGORIZED_TRANSACTIONS)
    descriptions = {r["description"] for r in result["recurring_merchants"]}

    assert "NETFLIX.COM" in descriptions
    assert "WHOLE FOODS MARKET #1234" in descriptions
    assert "ACME CORP PAYROLL" not in descriptions  # income, not an expense merchant

    netflix = next(r for r in result["recurring_merchants"] if r["description"] == "NETFLIX.COM")
    assert netflix["months_seen"] == 3
    assert netflix["avg_amount"] == 15.0


def test_insights_flag_strong_savings_rate():
    result = compute_patterns(CATEGORIZED_TRANSACTIONS)
    kinds = {i["kind"] for i in result["insights"]}
    texts = " ".join(i["text"] for i in result["insights"])

    assert "good" in kinds
    assert "90.5%" in texts or "90.5" in texts


def test_insights_flag_deficit_spending():
    deficit_txns = [
        {"date": "2026-01-05", "description": "SALARY", "amount": 500.0, "category": "Income"},
        {"date": "2026-01-06", "description": "RENT", "amount": -1200.0, "category": "Housing"},
    ]
    result = compute_patterns(deficit_txns)
    warning_texts = [i["text"] for i in result["insights"] if i["kind"] == "warning"]
    assert any("deficit" in t.lower() for t in warning_texts)


def test_insights_note_missing_income_for_expense_only_dataset():
    expense_only = [
        {"date": "2026-01-05", "description": "GROCERIES", "amount": -50.0, "category": "Groceries"},
    ]
    result = compute_patterns(expense_only)
    assert result["insights"][0]["kind"] == "neutral"
    assert "no income" in result["insights"][0]["text"].lower()


def test_empty_transactions_returns_placeholder_summary():
    result = compute_patterns([])

    assert result["total_expenses"] == 0.0
    assert result["total_income"] == 0.0
    assert result["summary_text"] == "No transactions available to analyze."


def test_generate_summary_uses_llm_when_provided(in_memory_store):
    base = compute_patterns(CATEGORIZED_TRANSACTIONS)
    fake_llm = make_summary_fake_llm("Grounded LLM summary text.")

    result = generate_summary(base, llm=fake_llm, store=in_memory_store)

    assert result["summary_text"] == "Grounded LLM summary text."
    assert len(result["retrieved_context"]) > 0
    assert len(fake_llm.calls) == 1


def test_generate_summary_falls_back_to_rule_based_without_llm(in_memory_store, monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)

    base = compute_patterns(CATEGORIZED_TRANSACTIONS)
    result = generate_summary(base, llm=None, store=in_memory_store)

    assert "Total expenses analyzed" in result["summary_text"]


def test_analyze_end_to_end_without_llm(in_memory_store, monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    result = analyze(CATEGORIZED_TRANSACTIONS, llm=None, store=in_memory_store)

    assert result["total_expenses"] == 852.0
    assert "Total expenses analyzed" in result["summary_text"]
