"""Analysis agent: finds spending patterns and generates a grounded summary.

Splits transactions by sign (negative = expense, positive = income) so it
can report income, expenses, and net savings the way a real bank statement
would, not just raw expenditure totals. Pattern-finding (totals, category
breakdown, category-over-time, month-over-month change, recurring merchants,
rule-based insights) is pure deterministic computation so it's cheap to
test. The narrative summary is rule-based by default and optionally
enhanced by Azure OpenAI, grounded with budgeting-benchmark snippets
retrieved from ChromaDB.
"""

from __future__ import annotations

import json
from collections import defaultdict

from langgraph_finance.retrieval.vector_store import FinanceVectorStore, get_default_store
from langgraph_finance.state import AnalysisResult, FinanceState, Transaction

TOP_CATEGORY_LIMIT = 5


def _month_key(date_str: str) -> str:
    return date_str[:7]  # YYYY-MM


def _build_insights(result: AnalysisResult) -> list[dict[str, str]]:
    insights: list[dict[str, str]] = []
    savings_rate = result.get("savings_rate_pct")

    if not result.get("total_income"):
        insights.append({
            "text": "No income transactions were detected — figures reflect expenditures only.",
            "kind": "neutral",
        })
    elif savings_rate is not None:
        if savings_rate < 0:
            insights.append({
                "text": f"Spending exceeded income by ${abs(result['net_savings']):,.0f} this period — a deficit.",
                "kind": "warning",
            })
        elif savings_rate < 10:
            insights.append({
                "text": f"Savings rate is {savings_rate:.1f}% — critically low. Aim for at least 20%.",
                "kind": "warning",
            })
        elif savings_rate < 20:
            insights.append({
                "text": f"Savings rate is {savings_rate:.1f}% — below the 20% target, but positive.",
                "kind": "neutral",
            })
        else:
            insights.append({
                "text": f"Strong savings rate of {savings_rate:.1f}% — at or above the recommended 20% benchmark.",
                "kind": "good",
            })

    if result.get("top_categories"):
        top = result["top_categories"][0]
        if top["pct_of_total"] > 40:
            insights.append({
                "text": (
                    f"{top['category']} makes up {top['pct_of_total']}% of spending "
                    f"(${top['total']:,.0f}) — a concentrated category worth reviewing."
                ),
                "kind": "warning",
            })
        else:
            insights.append({
                "text": (
                    f"Largest spending category is {top['category']} at ${top['total']:,.0f} "
                    f"({top['pct_of_total']}% of total)."
                ),
                "kind": "neutral",
            })

    mom = result.get("month_over_month_pct")
    if mom is not None:
        if mom > 15:
            insights.append({
                "text": f"Spending rose {mom}% versus the prior month — worth a closer look.",
                "kind": "warning",
            })
        elif mom < -15:
            insights.append({
                "text": f"Spending fell {abs(mom)}% versus the prior month — nice trend.",
                "kind": "good",
            })
        else:
            insights.append({
                "text": f"Spending is roughly stable month over month ({mom:+.1f}%).",
                "kind": "neutral",
            })

    recurring = result.get("recurring_merchants") or []
    if recurring:
        monthly_recurring_cost = sum(r["avg_amount"] for r in recurring)
        insights.append({
            "text": (
                f"{len(recurring)} recurring charge(s) detected, totaling roughly "
                f"${monthly_recurring_cost:,.0f}/month — check for unused subscriptions."
            ),
            "kind": "neutral",
        })

    return insights[:5]


def compute_patterns(transactions: list[Transaction]) -> AnalysisResult:
    result = AnalysisResult(
        total_income=0.0,
        total_expenses=0.0,
        net_savings=0.0,
        savings_rate_pct=None,
        category_totals={},
        top_categories=[],
        monthly_expenses={},
        monthly_income={},
        category_monthly_totals={},
        month_over_month_pct=None,
        recurring_merchants=[],
        insights=[],
        summary_text="",
        retrieved_context=[],
    )
    if not transactions:
        result["summary_text"] = "No transactions available to analyze."
        return result

    total_income = 0.0
    total_expenses = 0.0
    category_totals: dict[str, float] = defaultdict(float)
    monthly_income: dict[str, float] = defaultdict(float)
    monthly_expenses: dict[str, float] = defaultdict(float)
    merchant_months: dict[str, set[str]] = defaultdict(set)
    merchant_totals: dict[str, float] = defaultdict(float)
    merchant_counts: dict[str, int] = defaultdict(int)
    category_monthly: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for t in transactions:
        amt = float(t["amount"])
        month = _month_key(t["date"])
        if amt >= 0:
            total_income += amt
            monthly_income[month] += amt
            continue

        spend = abs(amt)
        total_expenses += spend
        monthly_expenses[month] += spend
        category = t.get("category") or "Other"
        category_totals[category] += spend
        category_monthly[category][month] += spend
        merchant_months[t["description"]].add(month)
        merchant_totals[t["description"]] += spend
        merchant_counts[t["description"]] += 1

    result["total_income"] = round(total_income, 2)
    result["total_expenses"] = round(total_expenses, 2)
    result["net_savings"] = round(total_income - total_expenses, 2)
    result["savings_rate_pct"] = (
        round((total_income - total_expenses) / total_income * 100, 1) if total_income > 0 else None
    )

    result["category_totals"] = {k: round(v, 2) for k, v in category_totals.items()}
    top = sorted(category_totals.items(), key=lambda kv: kv[1], reverse=True)
    result["top_categories"] = [
        {
            "category": cat,
            "total": round(amt, 2),
            "pct_of_total": round(amt / total_expenses * 100, 1) if total_expenses else 0.0,
        }
        for cat, amt in top[:TOP_CATEGORY_LIMIT]
    ]

    result["monthly_expenses"] = {k: round(v, 2) for k, v in sorted(monthly_expenses.items())}
    result["monthly_income"] = {k: round(v, 2) for k, v in sorted(monthly_income.items())}

    months_sorted = sorted(monthly_expenses.keys())
    if len(months_sorted) >= 2:
        prev_total = monthly_expenses[months_sorted[-2]]
        last_total = monthly_expenses[months_sorted[-1]]
        result["month_over_month_pct"] = (
            round((last_total - prev_total) / prev_total * 100, 1) if prev_total else None
        )

    recurring = [
        {
            "description": desc,
            "months_seen": len(months),
            "avg_amount": round(merchant_totals[desc] / merchant_counts[desc], 2),
            "occurrences": merchant_counts[desc],
        }
        for desc, months in merchant_months.items()
        if len(months) >= 2 and merchant_counts[desc] >= 2
    ]
    recurring.sort(key=lambda r: r["months_seen"], reverse=True)
    result["recurring_merchants"] = recurring[:10]

    # Category-over-time, capped to the top categories + a folded "Other" bucket
    # so the stacked chart never blows past the series-count ceiling.
    all_months = months_sorted
    top_cat_names = {c["category"] for c in result["top_categories"]}
    cat_over_time: dict[str, dict[str, float]] = {
        cat: {m: round(category_monthly[cat].get(m, 0.0), 2) for m in all_months}
        for cat in top_cat_names
    }
    fold_totals: dict[str, float] = defaultdict(float)
    for cat, by_month in category_monthly.items():
        if cat not in top_cat_names:
            for m, v in by_month.items():
                fold_totals[m] += v
    if fold_totals:
        existing_other = cat_over_time.get("Other", {m: 0.0 for m in all_months})
        cat_over_time["Other"] = {
            m: round(existing_other.get(m, 0.0) + fold_totals.get(m, 0.0), 2) for m in all_months
        }
    result["category_monthly_totals"] = cat_over_time

    result["insights"] = _build_insights(result)

    return result


def _rule_based_summary(result: AnalysisResult) -> str:
    n_months = len(result["monthly_expenses"])
    lines = [
        f"Total expenses analyzed: ${result['total_expenses']:,.2f} across {n_months} month(s)."
    ]
    if result["total_income"]:
        rate_note = (
            f" ({result['savings_rate_pct']}% savings rate)."
            if result["savings_rate_pct"] is not None
            else "."
        )
        lines.append(
            f"Total income: ${result['total_income']:,.2f}; "
            f"net savings: ${result['net_savings']:,.2f}{rate_note}"
        )
    if result["top_categories"]:
        top = result["top_categories"][0]
        lines.append(
            f"Largest category: {top['category']} at ${top['total']:,.2f} "
            f"({top['pct_of_total']}% of total)."
        )
    if result["month_over_month_pct"] is not None:
        pct = result["month_over_month_pct"]
        direction = "up" if pct >= 0 else "down"
        lines.append(f"Spending is {direction} {abs(pct)}% versus the prior month.")
    if result["recurring_merchants"]:
        names = ", ".join(r["description"] for r in result["recurring_merchants"][:3])
        lines.append(f"Recurring charges detected from: {names}.")
    return " ".join(lines)


def generate_summary(
    result: AnalysisResult,
    llm=None,
    store: FinanceVectorStore | None = None,
) -> AnalysisResult:
    result["summary_text"] = _rule_based_summary(result)

    store = store or get_default_store()
    top_cats = ", ".join(c["category"] for c in result["top_categories"])
    context = store.query_knowledge(top_cats or "spending", k=3) if top_cats else []
    result["retrieved_context"] = context

    if llm is None:
        try:
            from langgraph_finance.agents.categorization_agent import build_azure_llm

            llm = build_azure_llm()
        except Exception:
            llm = None

    if llm is not None:
        try:
            data_json = json.dumps({
                "total_income": result["total_income"],
                "total_expenses": result["total_expenses"],
                "net_savings": result["net_savings"],
                "savings_rate_pct": result["savings_rate_pct"],
                "top_categories": result["top_categories"],
                "month_over_month_pct": result["month_over_month_pct"],
            })
            benchmarks = "\n".join(f"- {c}" for c in context)
            prompt = (
                "Write a concise (3-4 sentence) financial summary for a household based on "
                f"this data:\n{data_json}\n\n"
                f"Ground your observations in these budgeting benchmarks where relevant:\n{benchmarks}"
            )
            response = llm.invoke([
                {
                    "role": "system",
                    "content": (
                        "You are a helpful, concise personal finance analyst. "
                        "Do not invent numbers that are not present in the data."
                    ),
                },
                {"role": "user", "content": prompt},
            ])
            content = response.content if hasattr(response, "content") else str(response)
            if content and content.strip():
                result["summary_text"] = content.strip()
        except Exception:
            pass  # keep the rule-based summary on any LLM failure

    return result


def analyze(
    transactions: list[Transaction],
    llm=None,
    store: FinanceVectorStore | None = None,
) -> AnalysisResult:
    result = compute_patterns(transactions)
    if not transactions:
        return result
    return generate_summary(result, llm=llm, store=store)


def analysis_node(state: FinanceState) -> FinanceState:
    """LangGraph node wrapper around `analyze`."""
    transactions = state.get("categorized_transactions") or state.get("transactions", [])
    analysis = analyze(transactions)
    return {**state, "analysis": analysis}
