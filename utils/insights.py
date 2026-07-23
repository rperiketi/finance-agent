"""
Phase 4 – Recommendation Engine (statistical insights layer)

Generates measurable, rule-based financial insights from categorized data.
The results are fed to the LLM prompt in the recommendation agent.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional


def _discretionary_slice(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude rows flagged as P2P/transfers when analysing discretionary patterns."""
    if df is None or df.empty or "is_transfer" not in df.columns:
        return df
    return df[~df["is_transfer"]]


def _last_two_months(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the DataFrame into the last complete month and the one before."""
    df = df.copy()
    df["period"] = df["date"].dt.to_period("M")
    periods = sorted(df["period"].unique())
    if len(periods) < 2:
        return df, df
    curr = df[df["period"] == periods[-1]]
    prev = df[df["period"] == periods[-2]]
    return curr, prev


def _expense_by_category(df: pd.DataFrame) -> pd.Series:
    expenses = df[df["amount"] < 0].copy()
    expenses["amount"] = expenses["amount"].abs()
    return expenses.groupby("category")["amount"].sum()


def spending_change_insights(df: pd.DataFrame) -> list[str]:
    """Detect significant month-over-month changes per category."""
    df = _discretionary_slice(df)
    curr, prev = _last_two_months(df)
    curr_cat = _expense_by_category(curr)
    prev_cat = _expense_by_category(prev)

    insights = []
    all_cats = set(curr_cat.index) | set(prev_cat.index)
    for cat in all_cats:
        c = curr_cat.get(cat, 0)
        p = prev_cat.get(cat, 0)
        if p == 0 and c > 50:
            insights.append(f"New spending in {cat}: ${c:.0f} this month (none last month).")
        elif p > 0:
            pct = (c - p) / p * 100
            if pct > 30:
                insights.append(
                    f"{cat} spending increased by {pct:.0f}% "
                    f"(${p:.0f} → ${c:.0f})."
                )
            elif pct < -30:
                insights.append(
                    f"{cat} spending decreased by {abs(pct):.0f}% "
                    f"(${p:.0f} → ${c:.0f}). Well done!"
                )
    return insights


def subscription_detector(df: pd.DataFrame) -> list[str]:
    """Find recurring charges that look like subscriptions."""
    total_months = df["date"].dt.to_period("M").nunique()
    df = _discretionary_slice(df)
    expenses = df[df["amount"] < 0].copy()
    expenses["amount"] = expenses["amount"].abs()
    expenses["month"] = expenses["date"].dt.to_period("M")

    insights = []
    for merchant, grp in expenses.groupby("merchant"):
        months_seen = grp["month"].nunique()
        if months_seen >= 2:
            avg_amt = grp["amount"].mean()
            std_amt = grp["amount"].std()
            # Low variation + multiple months = likely subscription
            if std_amt / avg_amt < 0.15 if avg_amt > 0 else False:
                insights.append(
                    f"Recurring charge detected: {merchant} ~${avg_amt:.0f}/month "
                    f"(seen in {months_seen}/{total_months} months, "
                    f"annual cost ~${avg_amt * 12:.0f})."
                )
    return insights


def savings_opportunity(df: pd.DataFrame, forecast_total: float) -> list[str]:
    """Suggest categories where spending could be reduced."""
    df = _discretionary_slice(df)
    curr, prev = _last_two_months(df)
    curr_cat = _expense_by_category(curr)
    total_curr = curr_cat.sum()

    insights = []
    for cat, amt in curr_cat.items():
        pct = amt / total_curr * 100 if total_curr > 0 else 0
        if pct > 25 and cat not in ("Income", "Utilities"):
            save_10 = amt * 0.10
            insights.append(
                f"{cat} is {pct:.0f}% of this month's spend (${amt:.0f}). "
                f"A 10% reduction would save ~${save_10:.0f}/month."
            )

    if forecast_total > 0 and total_curr > 0:
        diff = forecast_total - total_curr
        if diff > 0:
            insights.append(
                f"Forecast suggests spending ${diff:.0f} MORE next month "
                f"(${total_curr:.0f} → ${forecast_total:.0f}). "
                f"Consider reviewing discretionary categories."
            )
        else:
            insights.append(
                f"Great trend! Forecast suggests ${abs(diff):.0f} LESS next month."
            )
    return insights


def income_vs_expense_ratio(df: pd.DataFrame) -> list[str]:
    """Compute savings rate and flag if dangerously low."""
    income   = df[df["amount"] > 0]["amount"].sum()
    expenses = df[df["amount"] < 0]["amount"].abs().sum()
    if income == 0:
        return ["No income detected – upload a statement with salary/credit entries."]
    ratio = (income - expenses) / income * 100
    if ratio < 0:
        return [f"You are spending MORE than you earn: deficit of ${expenses - income:.0f}."]
    elif ratio < 10:
        return [f"Savings rate is {ratio:.1f}% — critically low. Target at least 20%."]
    elif ratio < 20:
        return [f"Savings rate is {ratio:.1f}%. Room for improvement – aim for 20%+."]
    else:
        return [f"Strong savings rate of {ratio:.1f}%. Keep it up!"]


def build_summary_text(df: pd.DataFrame, forecast_total: float = 0) -> str:
    """Return a plain-English summary block ready to feed to an LLM."""
    lines = []

    # Overall
    income   = df[df["amount"] > 0]["amount"].sum()
    expenses = df[df["amount"] < 0]["amount"].abs().sum()
    lines.append(f"Total income: ${income:,.0f}")
    lines.append(f"Total expenses: ${expenses:,.0f}")
    lines.append(f"Net savings: ${income - expenses:,.0f}")

    # Category breakdown
    cat_totals = _expense_by_category(df).sort_values(ascending=False)
    lines.append("\nTop spending categories:")
    for cat, amt in cat_totals.head(5).items():
        lines.append(f"  - {cat}: ${amt:,.0f}")

    # Insights
    lines.append("\nInsights:")
    for ins in spending_change_insights(df):
        lines.append(f"  • {ins}")
    for ins in subscription_detector(df):
        lines.append(f"  • {ins}")
    for ins in income_vs_expense_ratio(df):
        lines.append(f"  • {ins}")
    if forecast_total:
        for ins in savings_opportunity(df, forecast_total):
            lines.append(f"  • {ins}")

    return "\n".join(lines)


def generate_rule_based_recommendations(df: pd.DataFrame, forecast_total: float = 0) -> list[str]:
    """Return up to 6 actionable recommendations based on statistical rules."""
    recs = []
    recs += spending_change_insights(df)
    recs += subscription_detector(df)
    recs += income_vs_expense_ratio(df)
    recs += savings_opportunity(df, forecast_total)
    # Deduplicate and limit
    seen, out = set(), []
    for r in recs:
        key = r[:60]
        if key not in seen:
            seen.add(key)
            out.append(r)
        if len(out) >= 6:
            break
    return out
