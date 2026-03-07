"""Embedding provider factory."""
from __future__ import annotations

from typing import TYPE_CHECKING

from pgsemantic.config import LOCAL_MODEL_CONFIGS, OPENAI_MODEL_CONFIGS

if TYPE_CHECKING:
    from pgsemantic.embeddings.local import LocalProvider
    from pgsemantic.embeddings.ollama_provider import OllamaProvider
    from pgsemantic.embeddings.openai_provider import OpenAIProvider


def get_provider(
    name: str,
    api_key: str | None = None,
    ollama_base_url: str | None = None,
) -> LocalProvider | OpenAIProvider | OllamaProvider:
    """Factory function to get an embedding provider by name."""
    if name in LOCAL_MODEL_CONFIGS:
        from pgsemantic.embeddings.local import LocalProvider
        model_name, dimensions = LOCAL_MODEL_CONFIGS[name]
        return LocalProvider(model_name=model_name, dimensions=dimensions)
    elif name in OPENAI_MODEL_CONFIGS:
        if api_key is None:
            raise ValueError(
                "OpenAI provider requires an API key. "
                "Set OPENAI_API_KEY in your .env file."
            )
        from pgsemantic.embeddings.openai_provider import OpenAIProvider
        model_name, dimensions = OPENAI_MODEL_CONFIGS[name]
        return OpenAIProvider(api_key=api_key, model_name=model_name, dimensions=dimensions)
    elif name == "ollama":
        from pgsemantic.embeddings.ollama_provider import OllamaProvider
        if ollama_base_url:
            return OllamaProvider(base_url=ollama_base_url)
        return OllamaProvider()
    else:
        raise ValueError(
            f"Unknown embedding provider: '{name}'. "
            f"Supported: {list(LOCAL_MODEL_CONFIGS) + list(OPENAI_MODEL_CONFIGS) + ['ollama']}"
        )
