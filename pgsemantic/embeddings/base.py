"""Abstract embedding provider interface.

Uses typing.Protocol — not ABC. All providers must implement this interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class ProviderConfig:
    """Configuration for an embedding provider."""

    model_name: str
    dimensions: int
    batch_size: int = 32


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol that all embedding providers must implement."""

    config: ProviderConfig

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per text."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text. Used at search time."""
        ...
