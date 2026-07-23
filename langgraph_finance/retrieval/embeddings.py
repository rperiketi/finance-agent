"""Deterministic, offline embedding function for ChromaDB.

We deliberately avoid downloading a sentence-transformer model (chromadb's
default) so unit/integration tests and CI runs stay fast and network-free.
A hashing-trick bag-of-words vectorizer is enough to give ChromaDB
meaningful nearest-neighbor behavior for short transaction descriptions and
short knowledge snippets; swap this out for Azure OpenAI embeddings in
`AzureEmbeddingFunction` when running against production traffic.
"""

from __future__ import annotations

import os

from chromadb.api.types import Documents, Embeddings
from chromadb.api.types import EmbeddingFunction as ChromaEmbeddingFunction
from sklearn.feature_extraction.text import HashingVectorizer

_VECTORIZER = HashingVectorizer(
    n_features=256,
    alternate_sign=False,
    norm="l2",
    lowercase=True,
    ngram_range=(1, 2),
)


class HashingEmbeddingFunction(ChromaEmbeddingFunction):
    """Offline embedding function used for local dev, tests, and CI.

    Subclasses chromadb's `EmbeddingFunction` protocol (rather than just
    duck-typing `__call__`) so the default `embed_query` implementation
    (which delegates to `__call__`) is inherited correctly.
    """

    def __init__(self) -> None:
        pass  # no external resources to initialize

    def __call__(self, input: Documents) -> Embeddings:
        matrix = _VECTORIZER.transform(list(input))
        return matrix.toarray().tolist()

    @staticmethod
    def name() -> str:
        return "hashing-bow-256"

    def get_config(self) -> dict:
        return {"n_features": 256}

    @staticmethod
    def build_from_config(config: dict) -> "HashingEmbeddingFunction":
        return HashingEmbeddingFunction()


class AzureEmbeddingFunction(ChromaEmbeddingFunction):
    """Wraps Azure OpenAI embeddings for production use.

    Only imports `langchain_openai` lazily so environments without Azure
    credentials configured can still run the offline hashing embedder.
    """

    def __init__(self, deployment: str | None = None):
        from langchain_openai import AzureOpenAIEmbeddings

        self._deployment = deployment or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        self._embedder = AzureOpenAIEmbeddings(
            azure_deployment=self._deployment,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        )

    def __call__(self, input: Documents) -> Embeddings:
        return self._embedder.embed_documents(list(input))

    @staticmethod
    def name() -> str:
        return "azure-openai-embeddings"

    def get_config(self) -> dict:
        return {"deployment": self._deployment}

    @staticmethod
    def build_from_config(config: dict) -> "AzureEmbeddingFunction":
        return AzureEmbeddingFunction(deployment=config.get("deployment"))


def get_embedding_function(use_azure: bool = False):
    if use_azure:
        return AzureEmbeddingFunction()
    return HashingEmbeddingFunction()
