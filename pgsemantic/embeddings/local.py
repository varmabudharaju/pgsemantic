"""Local embedding provider using sentence-transformers.

Model: all-MiniLM-L6-v2, 384 dimensions, 22 MB download, zero cost.
"""
from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

from pgsemantic.config import DEFAULT_LOCAL_DIMENSIONS, DEFAULT_LOCAL_MODEL
from pgsemantic.embeddings.base import ProviderConfig
from pgsemantic.exceptions import EmbeddingProviderError

logger = logging.getLogger(__name__)


class LocalProvider:
    """Embedding provider using sentence-transformers (all-MiniLM-L6-v2)."""

    config = ProviderConfig(
        model_name=DEFAULT_LOCAL_MODEL,
        dimensions=DEFAULT_LOCAL_DIMENSIONS,
        batch_size=32,
    )

    def __init__(self) -> None:
        try:
            self._model = SentenceTransformer(DEFAULT_LOCAL_MODEL)
        except Exception as e:
            raise EmbeddingProviderError(
                f"Failed to load model '{DEFAULT_LOCAL_MODEL}': {e}"
            ) from e

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using sentence-transformers."""
        if not texts:
            raise ValueError("texts must be non-empty")
        try:
            embeddings = self._model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()  # type: ignore[no-any-return]
        except Exception as e:
            raise EmbeddingProviderError(f"Embedding failed: {e}") from e

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        return self.embed([text])[0]
