"""Categorization agent: labels each line item using Azure OpenAI, grounded
with few-shot examples retrieved from ChromaDB rather than relying purely
on the model's own memory of merchant names.

Two performance levers keep this workable on large statements:
  - Deduplication: each unique (description, income/expense) pair is
    categorized once and the result is broadcast to every transaction
    sharing that pair, instead of re-asking the LLM for every repeat
    occurrence of the same merchant.
  - Bounded concurrency + retry: batches are sent to Azure OpenAI in
    parallel (default up to 8 at a time, configurable), with retry/backoff
    on rate-limit-shaped errors, instead of one blocking call at a time.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from langgraph_finance.config import (
    CategorizationConfig,
    DEFAULT_CATEGORIES,
    get_azure_config,
    get_categorization_config,
)
from langgraph_finance.retrieval.vector_store import FinanceVectorStore, get_default_store
from langgraph_finance.state import FinanceState, Transaction

BATCH_SIZE = 15

TxnKey = tuple[str, str]  # (description, "income" | "expense")

SYSTEM_PROMPT = (
    "You are a precise financial transaction categorizer. Assign each transaction "
    f"description to exactly one category from this list:\n{', '.join(DEFAULT_CATEGORIES)}\n\n"
    "Use the labeled examples as guidance for how similar merchants have been "
    "categorized before. Respond with ONLY a JSON array, one object per transaction, "
    "in the same order given, like:\n"
    '[{"index": 0, "category": "Groceries"}, {"index": 1, "category": "Dining"}]\n'
    "No prose, no markdown fences."
)


def build_azure_llm():
    """Constructs the Azure OpenAI chat client. Raises if not configured."""
    from langchain_openai import AzureChatOpenAI

    cfg = get_azure_config()
    if not cfg.is_configured:
        raise RuntimeError(
            "Azure OpenAI is not configured. Set AZURE_OPENAI_API_KEY, "
            "AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT in the environment."
        )
    return AzureChatOpenAI(
        azure_deployment=cfg.deployment,
        azure_endpoint=cfg.endpoint,
        api_key=cfg.api_key,
        api_version=cfg.api_version,
        # Reasoning-tier deployments (e.g. gpt-5-mini) reject any temperature
        # override other than the default (1), so we don't set one here.
    )


def _txn_key(t: Transaction) -> TxnKey:
    return (t["description"], "income" if t["amount"] > 0 else "expense")


def _dedupe_transactions(
    transactions: list[Transaction],
) -> tuple[list[Transaction], dict[TxnKey, list[int]]]:
    """Groups transactions by (description, income/expense sign).

    Returns representative transactions (first occurrence of each key, in
    first-seen order) plus a map of key -> original indices, so the LLM only
    ever sees one instance of a repeated merchant.
    """
    key_to_indices: dict[TxnKey, list[int]] = {}
    representatives: list[Transaction] = []
    for i, t in enumerate(transactions):
        key = _txn_key(t)
        if key not in key_to_indices:
            key_to_indices[key] = []
            representatives.append(t)
        key_to_indices[key].append(i)
    return representatives, key_to_indices


def _is_retryable(exc: BaseException) -> bool:
    """Matches concrete rate-limit/transient signals only — never message
    text — so an unrelated error whose text happens to mention "throttled"
    or "rate limit" fails fast instead of retrying needlessly."""
    if getattr(exc, "status_code", None) == 429:
        return True
    return type(exc).__name__ in ("RateLimitError", "APITimeoutError", "APIConnectionError")


def _invoke_with_retry(llm, messages: list[dict], cfg: CategorizationConfig):
    @retry(
        stop=stop_after_attempt(cfg.max_retries),
        wait=wait_exponential_jitter(initial=cfg.retry_min_wait_seconds, max=cfg.retry_max_wait_seconds),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    )
    def _call():
        return llm.invoke(messages)

    return _call()


def _build_batch_prompt(batch: list[Transaction], store: FinanceVectorStore) -> str:
    seen_examples: dict[str, str] = {}
    for txn in batch:
        for ex in store.query_similar_examples(txn["description"], k=2):
            seen_examples[ex["description"]] = ex["category"]

    example_lines = "\n".join(f'- "{d}" -> {c}' for d, c in list(seen_examples.items())[:12])
    item_lines = "\n".join(
        f'{i}. "{t["description"]}" ({"income" if t["amount"] > 0 else "expense"}: '
        f'${abs(t["amount"]):.2f})'
        for i, t in enumerate(batch)
    )
    return (
        f"Labeled examples (for reference only):\n{example_lines}\n\n"
        f"Transactions to categorize:\n{item_lines}"
    )


def _parse_response(content: str, batch_len: int) -> dict[int, str]:
    text = re.sub(r"^```(json)?|```$", "", content.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

    result: dict[int, str] = {}
    for item in data:
        try:
            idx = int(item["index"])
            category = str(item["category"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if category not in DEFAULT_CATEGORIES:
            category = "Other"
        if 0 <= idx < batch_len:
            result[idx] = category
    return result


def _categorize_batch(
    batch: list[Transaction],
    batch_start: int,
    llm,
    store: FinanceVectorStore,
    cfg: CategorizationConfig,
) -> tuple[int, dict[int, str], list[str], list[TxnKey]]:
    """Categorizes one batch of representative (deduplicated) transactions.

    Runs on a worker thread and returns its results rather than mutating any
    shared state, so the caller can safely aggregate on the main thread with
    no locks.
    """
    notes: list[str] = []
    assignments: dict[int, str] = {}
    try:
        prompt = _build_batch_prompt(batch, store)
        response = _invoke_with_retry(
            llm,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            cfg,
        )
        content = response.content if hasattr(response, "content") else str(response)
        assignments = _parse_response(content, len(batch))
    except Exception as exc:
        notes.append(f"Batch starting at index {batch_start} failed categorization call: {exc}")

    for i, txn in enumerate(batch):
        if i not in assignments:
            assignments[i] = "Other"
            notes.append(
                f"Could not parse a category for '{txn['description']}' — defaulted to Other."
            )

    keys = [_txn_key(t) for t in batch]
    return batch_start, assignments, notes, keys


def categorize_transactions(
    transactions: list[Transaction],
    llm=None,
    store: FinanceVectorStore | None = None,
    batch_size: int = BATCH_SIZE,
    categorization_config: CategorizationConfig | None = None,
) -> tuple[list[Transaction], list[str]]:
    if not transactions:
        return [], []

    store = store or get_default_store()
    llm = llm or build_azure_llm()
    cfg = categorization_config or get_categorization_config()

    representatives, _ = _dedupe_transactions(transactions)
    batches = [
        (start, representatives[start:start + batch_size])
        for start in range(0, len(representatives), batch_size)
    ]

    notes: list[str] = []
    key_to_category: dict[TxnKey, str] = {}

    max_workers = min(cfg.max_concurrency, len(batches)) or 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_categorize_batch, batch, start, llm, store, cfg)
            for start, batch in batches
        ]
        # Only the main thread writes to `notes`/`key_to_category`, so no
        # lock is needed even though batches complete out of order.
        for future in as_completed(futures):
            _, assignments, batch_notes, keys = future.result()
            notes.extend(batch_notes)
            for i, key in enumerate(keys):
                key_to_category[key] = assignments.get(i, "Other")

    # Final assembly walks the *original* transaction order in one linear
    # pass, so output order/length always matches input regardless of the
    # order batches completed in.
    categorized: list[Transaction] = [
        {**t, "category": key_to_category.get(_txn_key(t), "Other")} for t in transactions
    ]

    return categorized, notes


def categorization_node(state: FinanceState) -> FinanceState:
    """LangGraph node wrapper around `categorize_transactions`."""
    transactions = state.get("transactions", [])
    categorized, notes = categorize_transactions(transactions)
    return {**state, "categorized_transactions": categorized, "categorization_notes": notes}
