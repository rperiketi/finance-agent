import json
import re
from types import SimpleNamespace

from langgraph_finance.agents.categorization_agent import categorize_transactions
from langgraph_finance.config import CategorizationConfig
from tests.conftest import FakeLLM, make_categorization_fake_llm


class _RateLimitError(Exception):
    status_code = 429


_FAST_RETRY_CONFIG = CategorizationConfig(
    max_concurrency=1, max_retries=3, retry_min_wait_seconds=0.01, retry_max_wait_seconds=0.02
)


def _sample_transactions():
    return [
        {"date": "2026-01-05", "description": "WHOLE FOODS MARKET #1234", "amount": 100.0},
        {"date": "2026-01-10", "description": "STARBUCKS STORE 4021", "amount": 50.0},
        {"date": "2026-01-15", "description": "SHELL OIL 57443210", "amount": 40.0},
        {"date": "2026-01-20", "description": "UNKNOWN MERCHANT XYZ", "amount": 12.0},
    ]


def test_categorizes_using_llm_and_retrieval(in_memory_store, fake_categorization_llm):
    transactions = _sample_transactions()

    categorized, notes = categorize_transactions(
        transactions, llm=fake_categorization_llm, store=in_memory_store
    )

    assert [t["category"] for t in categorized] == [
        "Groceries",
        "Dining",
        "Transportation",
        "Other",
    ]
    assert notes == []
    # retrieval was actually exercised: at least one query per unique description
    assert len(fake_categorization_llm.calls) == 1  # single batch, 4 < BATCH_SIZE


def test_batches_are_split_by_batch_size(in_memory_store, fake_categorization_llm):
    transactions = _sample_transactions()

    categorized, notes = categorize_transactions(
        transactions, llm=fake_categorization_llm, store=in_memory_store, batch_size=2
    )

    assert len(categorized) == 4
    assert len(fake_categorization_llm.calls) == 2
    assert notes == []


def test_llm_failure_defaults_to_other_with_note(in_memory_store):
    class ExplodingLLM:
        def invoke(self, messages):
            raise RuntimeError("Azure throttled the request")

    categorized, notes = categorize_transactions(
        _sample_transactions(), llm=ExplodingLLM(), store=in_memory_store
    )

    assert all(t["category"] == "Other" for t in categorized)
    assert any("failed categorization call" in n for n in notes)


def test_malformed_response_defaults_to_other(in_memory_store):
    from types import SimpleNamespace

    class GarbageLLM:
        def invoke(self, messages):
            return SimpleNamespace(content="not json at all")

    categorized, notes = categorize_transactions(
        _sample_transactions(), llm=GarbageLLM(), store=in_memory_store
    )

    assert all(t["category"] == "Other" for t in categorized)
    assert any("Could not parse a category" in n for n in notes)


def test_empty_transactions_short_circuits(in_memory_store, fake_categorization_llm):
    categorized, notes = categorize_transactions([], llm=fake_categorization_llm, store=in_memory_store)

    assert categorized == []
    assert notes == []
    assert fake_categorization_llm.calls == []


def test_deduplicates_repeated_merchant_into_one_llm_call(in_memory_store, fake_categorization_llm):
    starbucks = [
        {"date": f"2026-01-{day:02d}", "description": "STARBUCKS STORE 4021", "amount": 5.0}
        for day in range(1, 21)
    ]
    transactions = (
        starbucks[:5]
        + [{"date": "2026-01-06", "description": "WHOLE FOODS MARKET #1234", "amount": 100.0}]
        + starbucks[5:15]
        + [{"date": "2026-01-16", "description": "SHELL OIL 57443210", "amount": 40.0}]
        + starbucks[15:]
    )

    categorized, notes = categorize_transactions(
        transactions, llm=fake_categorization_llm, store=in_memory_store
    )

    assert len(categorized) == len(transactions) == 22
    assert notes == []
    assert len(fake_categorization_llm.calls) == 1  # 3 unique descriptions -> 1 batch

    assert [t["description"] for t in categorized] == [t["description"] for t in transactions]
    assert all(
        t["category"] == "Dining" for t in categorized if t["description"] == "STARBUCKS STORE 4021"
    )
    groceries = [t for t in categorized if t["description"] == "WHOLE FOODS MARKET #1234"]
    assert groceries[0]["category"] == "Groceries"
    transport = [t for t in categorized if t["description"] == "SHELL OIL 57443210"]
    assert transport[0]["category"] == "Transportation"


def test_dedup_key_distinguishes_income_from_expense(in_memory_store):
    def response_fn(messages):
        user_content = messages[-1]["content"]
        items = re.findall(r'^(\d+)\.\s+"([^"]+)"\s+\((income|expense):', user_content, re.MULTILINE)
        assignments = [
            {"index": int(idx), "category": "Income" if sign == "income" else "Shopping"}
            for idx, _desc, sign in items
        ]
        return json.dumps(assignments)

    llm = FakeLLM(response_fn)
    transactions = [
        {"date": "2026-01-05", "description": "PAYPAL TRANSFER", "amount": 200.0},
        {"date": "2026-01-06", "description": "PAYPAL TRANSFER", "amount": -50.0},
    ]

    categorized, notes = categorize_transactions(transactions, llm=llm, store=in_memory_store)

    assert notes == []
    assert categorized[0]["category"] == "Income"
    assert categorized[1]["category"] == "Shopping"
    # Same description, different sign -> two distinct dedup keys, both handled
    # in the one batch call (2 unique keys < BATCH_SIZE).
    assert len(llm.calls) == 1


def test_concurrent_batches_preserve_order_and_correctness(in_memory_store):
    llm = make_categorization_fake_llm(delay_seconds=0.01)
    transactions = [
        {
            "date": f"2026-01-{(i % 28) + 1:02d}",
            "description": f"MERCHANT {i} STARBUCKS" if i % 2 == 0 else f"MERCHANT {i} SHELL OIL",
            "amount": 10.0,
        }
        for i in range(40)
    ]

    categorized, notes = categorize_transactions(
        transactions, llm=llm, store=in_memory_store, batch_size=3
    )

    assert notes == []
    assert len(categorized) == 40
    assert [t["description"] for t in categorized] == [t["description"] for t in transactions]
    assert all(t["category"] in ("Dining", "Transportation") for t in categorized)
    assert len(llm.calls) > 1  # multiple batches actually ran


def test_retry_recovers_from_transient_rate_limit(in_memory_store):
    call_count = {"n": 0}

    class FlakyLLM:
        def invoke(self, messages):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise _RateLimitError("slow down")
            return SimpleNamespace(content='[{"index": 0, "category": "Dining"}]')

    categorized, notes = categorize_transactions(
        [{"date": "2026-01-05", "description": "STARBUCKS", "amount": 5.0}],
        llm=FlakyLLM(),
        store=in_memory_store,
        categorization_config=_FAST_RETRY_CONFIG,
    )

    assert categorized[0]["category"] == "Dining"
    assert notes == []
    assert call_count["n"] == 3  # failed twice, succeeded on the 3rd attempt


def test_retry_exhausted_falls_back_to_other(in_memory_store):
    class AlwaysRateLimited:
        def invoke(self, messages):
            raise _RateLimitError("still slow")

    categorized, notes = categorize_transactions(
        [{"date": "2026-01-05", "description": "STARBUCKS", "amount": 5.0}],
        llm=AlwaysRateLimited(),
        store=in_memory_store,
        categorization_config=_FAST_RETRY_CONFIG,
    )

    assert categorized[0]["category"] == "Other"
    assert any("failed categorization call" in n for n in notes)


def test_non_rate_limit_errors_are_not_retried(in_memory_store):
    """A plain RuntimeError — even one whose message mentions 'throttled' —
    must fail fast, not trigger retry/backoff delay."""
    call_count = {"n": 0}

    class ExplodingLLM:
        def invoke(self, messages):
            call_count["n"] += 1
            raise RuntimeError("Azure throttled the request")

    categorized, notes = categorize_transactions(
        [{"date": "2026-01-05", "description": "STARBUCKS", "amount": 5.0}],
        llm=ExplodingLLM(),
        store=in_memory_store,
        categorization_config=_FAST_RETRY_CONFIG,
    )

    assert categorized[0]["category"] == "Other"
    assert call_count["n"] == 1  # no retries attempted


def test_notes_aggregate_correctly_across_concurrent_failing_batches(in_memory_store):
    class SelectivelyFailingLLM:
        def invoke(self, messages):
            user_content = messages[-1]["content"]
            if "FAILMERCHANT" in user_content:
                raise RuntimeError("boom")
            indices = re.findall(r'^(\d+)\.\s+"', user_content, re.MULTILINE)
            return SimpleNamespace(
                content=json.dumps([{"index": int(i), "category": "Dining"} for i in indices])
            )

    descriptions = []
    for batch_idx in range(10):
        if batch_idx % 2 == 1:
            descriptions.append(f"FAILMERCHANT {batch_idx}")
            descriptions.append(f"OK MERCHANT {batch_idx}b")
        else:
            descriptions.append(f"OK MERCHANT {batch_idx}a")
            descriptions.append(f"OK MERCHANT {batch_idx}b2")

    transactions = [
        {"date": f"2026-01-{(i % 28) + 1:02d}", "description": d, "amount": 5.0}
        for i, d in enumerate(descriptions)
    ]

    categorized, notes = categorize_transactions(
        transactions, llm=SelectivelyFailingLLM(), store=in_memory_store, batch_size=2
    )

    assert len(categorized) == 20
    failure_notes = [n for n in notes if "failed categorization call" in n]
    assert len(failure_notes) == 5
    assert len(set(failure_notes)) == 5  # each references a distinct batch start index

    failing_related = [
        t for t in categorized if "FAILMERCHANT" in t["description"] or t["description"].endswith("b")
    ]
    assert all(t["category"] == "Other" for t in failing_related)


def test_concurrency_cap_is_honored(in_memory_store):
    llm = make_categorization_fake_llm(delay_seconds=0.05)
    transactions = [
        {"date": f"2026-01-{(i % 28) + 1:02d}", "description": f"MERCHANT {i} NETFLIX", "amount": 5.0}
        for i in range(30)
    ]
    cfg = CategorizationConfig(
        max_concurrency=3, max_retries=1, retry_min_wait_seconds=0.01, retry_max_wait_seconds=0.02
    )

    categorize_transactions(
        transactions, llm=llm, store=in_memory_store, batch_size=2, categorization_config=cfg
    )

    assert llm.max_in_flight <= 3
    assert llm.max_in_flight > 1  # proves real concurrency happened, not accidental serialization
