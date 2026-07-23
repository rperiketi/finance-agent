"""Shared fixtures for unit and integration tests.

No test in this suite calls Azure OpenAI or downloads embedding models —
the categorization LLM is stubbed with `FakeLLM`, and ChromaDB uses the
offline hashing embedding function, so the whole suite runs fast and
network-free (safe for GitHub Actions).
"""

from __future__ import annotations

import json
import re
import threading
import time
from types import SimpleNamespace

import pytest

from langgraph_finance.retrieval.vector_store import FinanceVectorStore

SAMPLE_CSV = """date,description,amount
2026-01-05,WHOLE FOODS MARKET #1234,100.00
2026-01-10,STARBUCKS STORE 4021,50.00
2026-01-15,SHELL OIL 57443210,40.00
2026-01-20,PG&E ELECTRIC BILL,60.00
2026-01-25,NETFLIX.COM,15.00
2026-02-05,WHOLE FOODS MARKET #1234,110.00
2026-02-10,STARBUCKS STORE 4021,55.00
2026-02-15,SHELL OIL 57443210,42.00
2026-02-20,PG&E ELECTRIC BILL,62.00
2026-02-25,NETFLIX.COM,15.00
2026-03-05,WHOLE FOODS MARKET #1234,120.00
2026-03-10,STARBUCKS STORE 4021,60.00
2026-03-15,SHELL OIL 57443210,44.00
2026-03-20,PG&E ELECTRIC BILL,64.00
2026-03-25,NETFLIX.COM,15.00
"""

CATEGORY_RULES = {
    "WHOLE FOODS": "Groceries",
    "STARBUCKS": "Dining",
    "SHELL OIL": "Transportation",
    "PG&E": "Utilities",
    "NETFLIX": "Subscriptions",
}


class FakeLLM:
    """Stand-in for AzureChatOpenAI's `.invoke()` interface.

    `delay_seconds` (optional) holds each call open briefly so concurrency
    tests can observe real overlap; `max_in_flight` tracks the highest
    number of simultaneous `invoke()` calls observed, for asserting a
    concurrency cap is actually honored.
    """

    def __init__(self, response_fn, delay_seconds: float = 0.0):
        self._response_fn = response_fn
        self._delay_seconds = delay_seconds
        self.calls: list[list[dict]] = []
        self._lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0

    def invoke(self, messages):
        with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            self.calls.append(messages)
            if self._delay_seconds:
                time.sleep(self._delay_seconds)
            content = self._response_fn(messages)
            return SimpleNamespace(content=content)
        finally:
            with self._lock:
                self.in_flight -= 1


def make_categorization_fake_llm(
    category_rules: dict[str, str] = CATEGORY_RULES, delay_seconds: float = 0.0
) -> FakeLLM:
    def response_fn(messages) -> str:
        user_content = messages[-1]["content"]
        items = re.findall(r'^(\d+)\.\s+"([^"]+)"', user_content, re.MULTILINE)
        assignments = []
        for idx_str, desc in items:
            category = "Other"
            for keyword, cat in category_rules.items():
                if keyword.lower() in desc.lower():
                    category = cat
                    break
            assignments.append({"index": int(idx_str), "category": category})
        return json.dumps(assignments)

    return FakeLLM(response_fn, delay_seconds=delay_seconds)


def make_summary_fake_llm(summary_text: str = "LLM-generated summary.") -> FakeLLM:
    return FakeLLM(lambda messages: summary_text)


def make_dual_purpose_fake_llm(
    category_rules: dict[str, str] = CATEGORY_RULES,
    summary_text: str = "Integration test summary.",
) -> FakeLLM:
    """Serves both the categorization agent (JSON array) and the analysis
    agent (plain-text summary) from one stub, routed by system prompt content
    — used when exercising the full LangGraph pipeline end to end."""

    def response_fn(messages) -> str:
        system_content = messages[0]["content"]
        if "categorizer" in system_content.lower():
            user_content = messages[-1]["content"]
            items = re.findall(r'^(\d+)\.\s+"([^"]+)"', user_content, re.MULTILINE)
            assignments = []
            for idx_str, desc in items:
                category = "Other"
                for keyword, cat in category_rules.items():
                    if keyword.lower() in desc.lower():
                        category = cat
                        break
                assignments.append({"index": int(idx_str), "category": category})
            return json.dumps(assignments)
        return summary_text

    return FakeLLM(response_fn)


@pytest.fixture
def sample_csv_text() -> str:
    return SAMPLE_CSV


@pytest.fixture
def in_memory_store() -> FinanceVectorStore:
    return FinanceVectorStore.in_memory()


@pytest.fixture
def fake_categorization_llm() -> FakeLLM:
    return make_categorization_fake_llm()


@pytest.fixture
def patched_llm_for_pipeline(monkeypatch) -> FakeLLM:
    """Patches the one place both agents obtain an Azure client, so the full
    LangGraph pipeline runs end to end without real Azure OpenAI credentials."""
    fake_llm = make_dual_purpose_fake_llm()
    monkeypatch.setattr(
        "langgraph_finance.agents.categorization_agent.build_azure_llm",
        lambda: fake_llm,
    )
    return fake_llm


@pytest.fixture
def patched_default_store(monkeypatch) -> FinanceVectorStore:
    """Patches the default (persistent) vector store lookup with an ephemeral,
    in-memory one so tests never write to disk or share state across runs."""
    store = FinanceVectorStore.in_memory()
    monkeypatch.setattr(
        "langgraph_finance.agents.categorization_agent.get_default_store", lambda: store
    )
    monkeypatch.setattr(
        "langgraph_finance.agents.analysis_agent.get_default_store", lambda: store
    )
    return store
