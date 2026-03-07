"""OpenAI embedding provider.

Model: text-embedding-3-small, 1536 dimensions.
Requires OPENAI_API_KEY environment variable.
"""
from __future__ import annotations

import logging

from openai import OpenAI

from pgsemantic.config import OPENAI_DIMENSIONS, OPENAI_MODEL
from pgsemantic.embeddings.base import ProviderConfig
from pgsemantic.exceptions import EmbeddingProviderError

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """Embedding provider using OpenAI embeddings API."""

    def __init__(self, api_key: str, model_name: str = OPENAI_MODEL, dimensions: int = OPENAI_DIMENSIONS) -> None:
        self._model_name = model_name
        self.config = ProviderConfig(
            model_name=model_name,
            dimensions=dimensions,
            batch_size=100,
        )
        try:
            self._client = OpenAI(api_key=api_key)
        except Exception as e:
            raise EmbeddingProviderError(
                f"Failed to initialize OpenAI client: {e}"
            ) from e

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using OpenAI API."""
        if not texts:
            raise ValueError("texts must be non-empty")
        try:
            response = self._client.embeddings.create(
                model=self._model_name,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise EmbeddingProviderError(f"OpenAI embedding failed: {e}") from e

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        return self.embed([text])[0]
