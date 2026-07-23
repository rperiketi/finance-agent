"""
Phase 5 – Recommendation Agent

Combines statistical insights with optional LLM reasoning to generate
actionable financial recommendations.

LLM is optional: if OPENAI_API_KEY is not set, falls back to pure
rule-based insights (still very useful).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd

from utils.insights import (
    generate_rule_based_recommendations,
    build_summary_text,
)

# LLM is optional
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


@dataclass
class RecommendationResult:
    rule_based: list[str] = field(default_factory=list)
    llm_insights: list[str] = field(default_factory=list)
    summary_text: str = ""
    used_llm: bool = False

    @property
    def all_insights(self) -> list[str]:
        """Merged, deduplicated list prioritising LLM output."""
        combined = self.llm_insights + self.rule_based
        seen, out = set(), []
        for item in combined:
            key = item[:60]
            if key not in seen:
                seen.add(key)
                out.append(item)
        return out[:8]

    def __str__(self) -> str:
        lines = ["[RecommendationAgent] Insights:"]
        for i, rec in enumerate(self.all_insights, 1):
            lines.append(f"  {i}. {rec}")
        if self.used_llm:
            lines.append("  (Enhanced by LLM reasoning)")
        return "\n".join(lines)


class RecommendationAgent:
    """
    Generates financial insights and recommendations.

    Parameters
    ----------
    use_llm : bool
        If True and OPENAI_API_KEY is set, enhances insights with GPT.
    llm_model : str
        OpenAI model to use (default: gpt-3.5-turbo for cost efficiency).
    """

    SYSTEM_PROMPT = (
        "You are a concise, honest personal finance advisor. "
        "Given a spending summary, generate exactly 3 actionable insights. "
        "Each insight must be:\n"
        "- One sentence\n"
        "- Specific (include dollar amounts or percentages)\n"
        "- Actionable (tell the user what to DO)\n"
        "Return a numbered list only. No preamble, no headers."
    )

    def __init__(self, use_llm: bool = False, llm_model: str = "gpt-3.5-turbo"):
        self.use_llm = use_llm and _OPENAI_AVAILABLE
        self.llm_model = llm_model
        self._client = None
        if self.use_llm:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self._client = OpenAI(api_key=api_key)
            else:
                self.use_llm = False

    def run(
        self,
        df: pd.DataFrame,
        forecast_total: float = 0,
        category_summary: pd.DataFrame | None = None,
    ) -> RecommendationResult:
        result = RecommendationResult()

        if df is None or df.empty:
            result.rule_based = ["No data available for analysis."]
            return result

        # ── Rule-based insights (always) ──────────────────────────────────────
        result.rule_based = generate_rule_based_recommendations(df, forecast_total)
        result.summary_text = build_summary_text(df, forecast_total)

        # ── LLM enhancement (optional) ────────────────────────────────────────
        if self.use_llm and self._client:
            try:
                response = self._client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user",   "content": result.summary_text},
                    ],
                    temperature=0.4,
                    max_tokens=300,
                )
                raw = response.choices[0].message.content.strip()
                result.llm_insights = self._parse_numbered_list(raw)
                result.used_llm = True
            except Exception:
                pass  # silently fall back to rule-based

        return result

    @staticmethod
    def _parse_numbered_list(text: str) -> list[str]:
        """Extract items from a numbered list returned by LLM."""
        import re
        lines = text.strip().split("\n")
        items = []
        for line in lines:
            line = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
            if line:
                items.append(line)
        return items
