"""Environment-driven configuration for the LangGraph finance system."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_CATEGORIES = [
    "Income",
    "Groceries",
    "Dining",
    "Transportation",
    "Housing",
    "Utilities",
    "Entertainment",
    "Healthcare",
    "Shopping",
    "Travel",
    "Subscriptions",
    "Insurance",
    "Education",
    "Personal Care",
    "Fees & Charges",
    "Other",
]


@dataclass(frozen=True)
class AzureOpenAIConfig:
    api_key: str | None
    endpoint: str | None
    deployment: str | None
    api_version: str

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.endpoint and self.deployment)

    @classmethod
    def from_env(cls) -> "AzureOpenAIConfig":
        return cls(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        )


@dataclass(frozen=True)
class ChromaConfig:
    persist_directory: str
    categorization_collection: str = "categorization_examples"
    knowledge_collection: str = "financial_knowledge"

    @classmethod
    def from_env(cls) -> "ChromaConfig":
        return cls(
            persist_directory=os.getenv(
                "CHROMA_PERSIST_DIR",
                os.path.join(os.path.dirname(__file__), "chroma_db"),
            )
        )


@dataclass(frozen=True)
class CategorizationConfig:
    max_concurrency: int = 8
    max_retries: int = 5
    retry_min_wait_seconds: float = 1.0
    retry_max_wait_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> "CategorizationConfig":
        return cls(
            max_concurrency=int(os.getenv("CATEGORIZATION_MAX_CONCURRENCY", "8")),
            max_retries=int(os.getenv("CATEGORIZATION_MAX_RETRIES", "5")),
            retry_min_wait_seconds=float(os.getenv("CATEGORIZATION_RETRY_MIN_WAIT", "1.0")),
            retry_max_wait_seconds=float(os.getenv("CATEGORIZATION_RETRY_MAX_WAIT", "30.0")),
        )


def get_azure_config() -> AzureOpenAIConfig:
    return AzureOpenAIConfig.from_env()


def get_chroma_config() -> ChromaConfig:
    return ChromaConfig.from_env()


def get_categorization_config() -> CategorizationConfig:
    return CategorizationConfig.from_env()
