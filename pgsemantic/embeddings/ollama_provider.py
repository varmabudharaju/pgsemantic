"""Ollama embedding provider.

Model: nomic-embed-text, 768 dimensions, runs locally via Ollama.
No API key required. Outperforms OpenAI text-embedding-ada-002 on MTEB benchmarks.

Requires Ollama running locally: https://ollama.com
Pull the model first: ollama pull nomic-embed-text
"""
from __future__ import annotations

import logging
from urllib.parse import urljoin

import httpx

from pgsemantic.config import OLLAMA_BASE_URL, OLLAMA_DIMENSIONS, OLLAMA_MODEL
from pgsemantic.embeddings.base import ProviderConfig
from pgsemantic.exceptions import EmbeddingProviderError

logger = logging.getLogger(__name__)


class OllamaProvider:
    """Embedding provider using Ollama (nomic-embed-text)."""

    config = ProviderConfig(
        model_name=OLLAMA_MODEL,
        dimensions=OLLAMA_DIMENSIONS,
        batch_size=32,
    )

    def __init__(self, base_url: str = OLLAMA_BASE_URL) -> None:
        self._base_url = base_url
        self._embed_url = urljoin(base_url.rstrip("/") + "/", "api/embed")
        # Verify Ollama is reachable
        try:
            resp = httpx.get(base_url, timeout=5.0)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise EmbeddingProviderError(
                f"Cannot connect to Ollama at {base_url}. "
                "Is Ollama running? Install from https://ollama.com "
                "and run: ollama pull nomic-embed-text"
            ) from e

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using Ollama API."""
        if not texts:
            raise ValueError("texts must be non-empty")
        try:
            resp = httpx.post(
                self._embed_url,
                json={"model": OLLAMA_MODEL, "input": texts},
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings: list[list[float]] = data["embeddings"]
            return embeddings
        except httpx.HTTPError as e:
            raise EmbeddingProviderError(
                f"Ollama embedding failed: {e}. "
                "Make sure the model is pulled: ollama pull nomic-embed-text"
            ) from e
        except (KeyError, TypeError) as e:
            raise EmbeddingProviderError(
                f"Unexpected response from Ollama: {e}"
            ) from e

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        return self.embed([text])[0]
