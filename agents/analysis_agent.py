"""
Phase 5 – Analysis Agent

Responsibility: Receives cleaned DataFrame from IngestionAgent,
runs categorization and forecasting, and returns structured results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from models.categorizer import ExpenseCategorizer
from models.forecaster import SpendingForecaster


@dataclass
class AnalysisResult:
    categorized_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    category_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    uncertain_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    forecast_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    category_forecast: pd.DataFrame = field(default_factory=pd.DataFrame)
    next_month_estimate: dict = field(default_factory=dict)
    model_used: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return not self.categorized_df.empty

    def __str__(self) -> str:
        lines = ["[AnalysisAgent] Results:"]
        if self.next_month_estimate:
            est = self.next_month_estimate
            lines.append(f"  Forecast model  : {est.get('model', 'N/A')}")
            lines.append(f"  Next 30d spend  : ${est.get('forecast_total_30d', 0):,.0f}")
            lines.append(f"  Last 30d actual : ${est.get('last_30d_actual', 0):,.0f}")
            lines.append(f"  Trend           : {est.get('change_pct', 0):+.1f}%")
        if not self.category_summary.empty:
            lines.append("\n  Top categories:")
            for _, row in self.category_summary.head(5).iterrows():
                lines.append(f"    {row['category']:<22} ${row['total']:>8,.0f}  ({row['pct_of_total']:.1f}%)")
        for e in self.errors:
            lines.append(f"  [E] {e}")
        return "\n".join(lines)


class AnalysisAgent:
    """
    Runs full ML analysis pipeline:
      1. Expense categorization (TF-IDF + Logistic Regression)
      2. Spending forecast (Prophet / Linear Regression)
      3. Per-category forecast

    Usage
    -----
        agent = AnalysisAgent()
        result = agent.run(clean_df, forecast_days=30)
    """

    def __init__(self):
        self.categorizer = ExpenseCategorizer()
        self._forecaster: Optional[SpendingForecaster] = None

    def run(self, df: pd.DataFrame, forecast_days: int = 30) -> AnalysisResult:
        result = AnalysisResult()

        if df is None or df.empty:
            result.errors.append("Empty DataFrame – nothing to analyse.")
            return result

        # ── Step 1: Categorize ────────────────────────────────────────────────
        try:
            result.categorized_df = self.categorizer.categorize_df(df)
            result.category_summary = self.categorizer.category_summary(result.categorized_df)
            result.uncertain_df = self.categorizer.uncertain_rows(result.categorized_df)
        except Exception as exc:
            result.errors.append(f"Categorization failed: {exc}")
            result.categorized_df = df.copy()
            result.categorized_df["category"] = "Other"

        # ── Step 2: Forecast ──────────────────────────────────────────────────
        self._forecaster = None
        try:
            self._forecaster = SpendingForecaster(result.categorized_df)
            result.forecast_df = self._forecaster.forecast(days=forecast_days)
            result.next_month_estimate = self._forecaster.next_month_estimate()
            result.model_used = self._forecaster.model_used
        except Exception as exc:
            result.errors.append(f"Forecasting failed: {exc}")

        # ── Step 3: Per-category forecast ──────────────────────────────────────
        try:
            if not result.categorized_df.empty and self._forecaster is not None:
                result.category_forecast = self._forecaster.category_forecast(
                    result.categorized_df, days=forecast_days
                )
        except Exception as exc:
            result.errors.append(f"Category forecast failed: {exc}")

        return result

    def retrain_categorizer(self, labeled_data: list[tuple[str, str]]):
        """Retrain the ML model with user-verified labels."""
        self.categorizer.train(labeled_data, verbose=True)
