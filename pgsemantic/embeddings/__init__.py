"""Embedding provider factory."""
from __future__ import annotations

from pgsemantic.embeddings.local import LocalProvider
from pgsemantic.embeddings.ollama_provider import OllamaProvider
from pgsemantic.embeddings.openai_provider import OpenAIProvider


def get_provider(
    name: str,
    api_key: str | None = None,
    ollama_base_url: str | None = None,
) -> LocalProvider | OpenAIProvider | OllamaProvider:
    """Factory function to get an embedding provider by name."""
    if name == "local":
        return LocalProvider()
    elif name == "openai":
        if api_key is None:
            raise ValueError(
                "OpenAI provider requires an API key. "
                "Set OPENAI_API_KEY in your .env file."
            )
        return OpenAIProvider(api_key=api_key)
    elif name == "ollama":
        if ollama_base_url:
            return OllamaProvider(base_url=ollama_base_url)
        return OllamaProvider()
    else:
        raise ValueError(
            f"Unknown embedding provider: '{name}'. "
            f"Supported providers: 'local', 'openai', 'ollama'"
        )
