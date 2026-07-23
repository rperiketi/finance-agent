"""ChromaDB-backed retrieval so agents pull grounded context instead of
relying purely on LLM memory.

Two collections:
  - `categorization_examples`: labeled (description -> category) pairs used
    as few-shot context for the categorization agent.
  - `financial_knowledge`: short budgeting-benchmark snippets used as
    grounding context for the analysis agent's narrative summary.
"""

from __future__ import annotations

import chromadb

from langgraph_finance.config import ChromaConfig, get_chroma_config
from langgraph_finance.retrieval.embeddings import get_embedding_function
from langgraph_finance.retrieval.seed_data import CATEGORIZATION_EXAMPLES, KNOWLEDGE_SNIPPETS


class FinanceVectorStore:
    def __init__(
        self,
        client: "chromadb.ClientAPI | None" = None,
        embedding_function=None,
        config: ChromaConfig | None = None,
    ):
        self.embedding_function = embedding_function or get_embedding_function()
        self.config = config or get_chroma_config()
        self.client = client or chromadb.PersistentClient(path=self.config.persist_directory)
        self._categorization_collection = None
        self._knowledge_collection = None

    @classmethod
    def in_memory(cls, embedding_function=None) -> "FinanceVectorStore":
        """Ephemeral, non-persisted store — used by tests."""
        return cls(client=chromadb.EphemeralClient(), embedding_function=embedding_function)

    @property
    def categorization_collection(self):
        if self._categorization_collection is None:
            coll = self.client.get_or_create_collection(
                name=self.config.categorization_collection,
                embedding_function=self.embedding_function,
            )
            if coll.count() == 0:
                coll.add(
                    documents=[ex["description"] for ex in CATEGORIZATION_EXAMPLES],
                    metadatas=[{"category": ex["category"]} for ex in CATEGORIZATION_EXAMPLES],
                    ids=[f"cat-ex-{i}" for i in range(len(CATEGORIZATION_EXAMPLES))],
                )
            self._categorization_collection = coll
        return self._categorization_collection

    @property
    def knowledge_collection(self):
        if self._knowledge_collection is None:
            coll = self.client.get_or_create_collection(
                name=self.config.knowledge_collection,
                embedding_function=self.embedding_function,
            )
            if coll.count() == 0:
                coll.add(
                    documents=[s["text"] for s in KNOWLEDGE_SNIPPETS],
                    metadatas=[{"topic": s["topic"]} for s in KNOWLEDGE_SNIPPETS],
                    ids=[f"knowledge-{i}" for i in range(len(KNOWLEDGE_SNIPPETS))],
                )
            self._knowledge_collection = coll
        return self._knowledge_collection

    def query_similar_examples(self, description: str, k: int = 3) -> list[dict]:
        coll = self.categorization_collection
        n = min(k, coll.count()) or 1
        res = coll.query(query_texts=[description], n_results=n)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        return [
            {"description": doc, "category": meta.get("category", "Other")}
            for doc, meta in zip(docs, metas)
        ]

    def query_knowledge(self, query_text: str, k: int = 3) -> list[str]:
        coll = self.knowledge_collection
        n = min(k, coll.count()) or 1
        res = coll.query(query_texts=[query_text], n_results=n)
        return res.get("documents", [[]])[0]


_default_store: FinanceVectorStore | None = None


def get_default_store() -> FinanceVectorStore:
    global _default_store
    if _default_store is None:
        _default_store = FinanceVectorStore()
    return _default_store
