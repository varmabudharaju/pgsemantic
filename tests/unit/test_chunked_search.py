"""Tests for chunked search config and storage."""
from __future__ import annotations

from pgsemantic.config import (
    DEFAULT_LOCAL_DIMENSIONS,
    DEFAULT_LOCAL_MODEL,
    HNSW_EF_CONSTRUCTION,
    HNSW_M,
    ProjectConfig,
    TableConfig,
)


def _make_chunked_config(
    table: str = "articles",
    chunked: bool = True,
    chunk_max_tokens: int = 256,
    chunk_overlap: int = 64,
) -> TableConfig:
    return TableConfig(
        table=table,
        schema="public",
        column="body",
        embedding_column="embedding",
        model="local",
        model_name=DEFAULT_LOCAL_MODEL,
        dimensions=DEFAULT_LOCAL_DIMENSIONS,
        hnsw_m=HNSW_M,
        hnsw_ef_construction=HNSW_EF_CONSTRUCTION,
        applied_at="2026-03-30T00:00:00Z",
        storage_mode="external",
        shadow_table=f"pgsemantic_embeddings_{table}",
        chunked=chunked,
        chunk_max_tokens=chunk_max_tokens,
        chunk_overlap=chunk_overlap,
    )


class TestChunkedConfig:
    def test_chunked_defaults_to_false(self) -> None:
        tc = TableConfig(
            table="t", schema="public", column="c",
            embedding_column="embedding", model="local",
            model_name="m", dimensions=384,
            hnsw_m=16, hnsw_ef_construction=64,
            applied_at="2026-01-01",
        )
        assert tc.chunked is False
        assert tc.chunk_max_tokens == 256
        assert tc.chunk_overlap == 64

    def test_chunked_config_round_trip(self) -> None:
        from dataclasses import asdict
        tc = _make_chunked_config()
        data = asdict(tc)
        restored = TableConfig(**data)
        assert restored.chunked is True
        assert restored.chunk_max_tokens == 256
        assert restored.chunk_overlap == 64

    def test_project_config_with_chunked_table(self) -> None:
        config = ProjectConfig(
            version="0.3.0",
            tables=[_make_chunked_config()],
        )
        tc = config.get_table_config("articles")
        assert tc is not None
        assert tc.chunked is True
