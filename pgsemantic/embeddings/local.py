"""Local embedding provider using sentence-transformers.

Model: all-MiniLM-L6-v2, 384 dimensions, 22 MB download, zero cost.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
        import io
        import sys

        # Redirect stderr for the entire init to suppress HuggingFace warnings
        # that print directly to stderr (e.g. "unauthenticated requests" warning).
        original_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            self._suppress_noise()

            from sentence_transformers import SentenceTransformer

            self._model: SentenceTransformer = SentenceTransformer(
                DEFAULT_LOCAL_MODEL, trust_remote_code=False
            )
        except Exception as e:
            sys.stderr = original_stderr
            raise EmbeddingProviderError(
                f"Failed to load model '{DEFAULT_LOCAL_MODEL}': {e}"
            ) from e
        finally:
            sys.stderr = original_stderr

    @staticmethod
    def _suppress_noise() -> None:
        """Suppress verbose logging from HuggingFace/transformers/httpx."""
        import os
        import warnings

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
        os.environ.setdefault("HF_HUB_VERBOSITY", "error")

        # Suppress Python warnings from transformers and huggingface_hub
        warnings.filterwarnings("ignore", module="transformers")
        warnings.filterwarnings("ignore", module="huggingface_hub")
        warnings.filterwarnings("ignore", message=".*unauthenticated.*")
        warnings.filterwarnings("ignore", message=".*Hugging Face.*")

        import transformers

        transformers.logging.set_verbosity_error()

        # Suppress httpx "HTTP Request: HEAD/GET" log spam and HF warnings
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

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
