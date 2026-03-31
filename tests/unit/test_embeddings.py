"""Tests for embedding providers — interface compliance and batching."""
from unittest.mock import MagicMock, patch

import pytest

from pgsemantic.config import (
    DEFAULT_LOCAL_DIMENSIONS,
    OLLAMA_DIMENSIONS,
    OPENAI_DIMENSIONS,
)
from pgsemantic.embeddings import get_provider
from pgsemantic.embeddings.base import ProviderConfig


class TestProviderConfig:
    def test_defaults(self) -> None:
        config = ProviderConfig(model_name="test", dimensions=384)
        assert config.batch_size == 32

    def test_custom_batch_size(self) -> None:
        config = ProviderConfig(model_name="test", dimensions=384, batch_size=64)
        assert config.batch_size == 64


class TestLocalProvider:
    def _make_provider(self) -> "LocalProvider":
        """Create a LocalProvider with mocked SentenceTransformer."""
        import numpy as np

        from pgsemantic.embeddings.local import LocalProvider

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([
            [0.1] * DEFAULT_LOCAL_DIMENSIONS,
            [0.2] * DEFAULT_LOCAL_DIMENSIONS,
        ])

        # Build provider without calling __init__ (which imports sentence_transformers)
        provider = object.__new__(LocalProvider)
        provider.config = ProviderConfig(
            model_name="all-MiniLM-L6-v2",
            dimensions=DEFAULT_LOCAL_DIMENSIONS,
            batch_size=32,
        )
        provider._model = mock_model
        return provider

    def test_embed_returns_correct_dimensions(self) -> None:
        provider = self._make_provider()
        result = provider.embed(["hello", "world"])
        assert len(result) == 2
        assert len(result[0]) == DEFAULT_LOCAL_DIMENSIONS

    def test_embed_query_returns_single_vector(self) -> None:
        import numpy as np

        provider = self._make_provider()
        provider._model.encode.return_value = np.array([[0.1] * DEFAULT_LOCAL_DIMENSIONS])
        result = provider.embed_query("test")
        assert len(result) == DEFAULT_LOCAL_DIMENSIONS

    def test_embed_empty_raises(self) -> None:
        provider = self._make_provider()
        with pytest.raises(ValueError, match="non-empty"):
            provider.embed([])

    def test_config_values(self) -> None:
        provider = self._make_provider()
        assert provider.config.model_name == "all-MiniLM-L6-v2"
        assert provider.config.dimensions == DEFAULT_LOCAL_DIMENSIONS


class TestOpenAIProvider:
    @patch("pgsemantic.embeddings.openai_provider.OpenAI")
    def test_embed_returns_correct_dimensions(self, mock_openai_class: MagicMock) -> None:
        mock_client = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * OPENAI_DIMENSIONS
        mock_response = MagicMock()
        mock_response.data = [mock_embedding, mock_embedding]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        from pgsemantic.embeddings.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="sk-test")
        result = provider.embed(["hello", "world"])
        assert len(result) == 2
        assert len(result[0]) == OPENAI_DIMENSIONS

    @patch("pgsemantic.embeddings.openai_provider.OpenAI")
    def test_embed_empty_raises(self, mock_openai_class: MagicMock) -> None:
        from pgsemantic.embeddings.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="sk-test")
        with pytest.raises(ValueError, match="non-empty"):
            provider.embed([])

    @patch("pgsemantic.embeddings.openai_provider.OpenAI")
    def test_config_values(self, mock_openai_class: MagicMock) -> None:
        from pgsemantic.embeddings.openai_provider import OpenAIProvider

        provider = OpenAIProvider(api_key="sk-test")
        assert provider.config.model_name == "text-embedding-3-small"
        assert provider.config.dimensions == OPENAI_DIMENSIONS


class TestOllamaProvider:
    @patch("pgsemantic.embeddings.ollama_provider.httpx.get")
    def test_config_values(self, mock_get: MagicMock) -> None:
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.raise_for_status = MagicMock()

        from pgsemantic.embeddings.ollama_provider import OllamaProvider

        provider = OllamaProvider()
        assert provider.config.model_name == "nomic-embed-text"
        assert provider.config.dimensions == OLLAMA_DIMENSIONS

    @patch("pgsemantic.embeddings.ollama_provider.httpx.post")
    @patch("pgsemantic.embeddings.ollama_provider.httpx.get")
    def test_embed_returns_correct_shape(
        self, mock_get: MagicMock, mock_post: MagicMock
    ) -> None:
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.raise_for_status = MagicMock()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "embeddings": [
                [0.1] * OLLAMA_DIMENSIONS,
                [0.2] * OLLAMA_DIMENSIONS,
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        from pgsemantic.embeddings.ollama_provider import OllamaProvider

        provider = OllamaProvider()
        result = provider.embed(["hello", "world"])
        assert len(result) == 2
        assert len(result[0]) == OLLAMA_DIMENSIONS

    @patch("pgsemantic.embeddings.ollama_provider.httpx.get")
    def test_embed_empty_raises(self, mock_get: MagicMock) -> None:
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.raise_for_status = MagicMock()

        from pgsemantic.embeddings.ollama_provider import OllamaProvider

        provider = OllamaProvider()
        with pytest.raises(ValueError, match="non-empty"):
            provider.embed([])

    @patch("pgsemantic.embeddings.ollama_provider.httpx.get")
    def test_connection_failure_raises(self, mock_get: MagicMock) -> None:
        import httpx

        from pgsemantic.exceptions import EmbeddingProviderError

        mock_get.side_effect = httpx.ConnectError("Connection refused")

        from pgsemantic.embeddings.ollama_provider import OllamaProvider

        with pytest.raises(EmbeddingProviderError, match="Cannot connect to Ollama"):
            OllamaProvider()

    @patch("pgsemantic.embeddings.ollama_provider.httpx.post")
    @patch("pgsemantic.embeddings.ollama_provider.httpx.get")
    def test_embed_query_returns_single_vector(
        self, mock_get: MagicMock, mock_post: MagicMock
    ) -> None:
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.raise_for_status = MagicMock()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "embeddings": [[0.1] * OLLAMA_DIMENSIONS]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        from pgsemantic.embeddings.ollama_provider import OllamaProvider

        provider = OllamaProvider()
        result = provider.embed_query("test")
        assert len(result) == OLLAMA_DIMENSIONS


class TestGetProvider:
    @patch("pgsemantic.embeddings.get_provider")
    def test_get_local(self, mock_get: MagicMock) -> None:
        mock_provider = MagicMock()
        mock_provider.config = ProviderConfig(
            model_name="all-MiniLM-L6-v2", dimensions=DEFAULT_LOCAL_DIMENSIONS,
        )
        mock_get.return_value = mock_provider
        provider = mock_get("local")
        assert provider.config.model_name == "all-MiniLM-L6-v2"

    @patch("pgsemantic.embeddings.openai_provider.OpenAI")
    def test_get_openai(self, mock_openai: MagicMock) -> None:
        provider = get_provider("openai", api_key="sk-test")
        assert provider.config.model_name == "text-embedding-3-small"

    @patch("pgsemantic.embeddings.ollama_provider.httpx.get")
    def test_get_ollama(self, mock_get: MagicMock) -> None:
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.raise_for_status = MagicMock()
        provider = get_provider("ollama")
        assert provider.config.model_name == "nomic-embed-text"

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            get_provider("unknown")

    def test_openai_without_key_raises(self) -> None:
        with pytest.raises(ValueError, match="API key"):
            get_provider("openai")
