"""Tests for chunked search config and storage."""
from __future__ import annotations

from unittest.mock import MagicMock

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


class TestChunkedShadowTable:
    def test_create_chunked_shadow_table_sql(self) -> None:
        from pgsemantic.db.vectors import create_chunked_shadow_table
        mock_conn = MagicMock()
        create_chunked_shadow_table(mock_conn, "pgsemantic_embeddings_articles", 384, "public")
        call_args = mock_conn.execute.call_args_list
        sql = str(call_args[0][0][0])
        assert "chunk_index" in sql
        assert "chunk_text" in sql
        assert "row_id" in sql
        assert "PRIMARY KEY" in sql


class TestChunkedSearch:
    def test_search_chunked_deduplicates_by_row_id(self) -> None:
        from pgsemantic.db.vectors import search_chunked
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            {"row_id": "1", "chunk_text": "chunk 1a", "chunk_index": 0, "similarity": 0.9},
            {"row_id": "1", "chunk_text": "chunk 1b", "chunk_index": 1, "similarity": 0.7},
            {"row_id": "2", "chunk_text": "chunk 2a", "chunk_index": 0, "similarity": 0.85},
        ]
        # First call is SET hnsw.ef_search, second is the actual query
        mock_conn.execute.side_effect = [MagicMock(), mock_result]

        results = search_chunked(
            conn=mock_conn, table="articles", column="body",
            query_vector=[0.1] * 384,
            shadow_table="pgsemantic_embeddings_articles",
            schema="public", pk_columns=["id"], limit=10,
        )
        assert len(results) == 2
        assert results[0]["matched_chunk"] == "chunk 1a"
        assert results[0]["chunk_similarity"] == 0.9


class TestBulkInsertChunks:
    def test_insert_chunks_for_row(self) -> None:
        from pgsemantic.db.vectors import bulk_insert_chunks
        mock_conn = MagicMock()
        chunks = ["chunk 0", "chunk 1", "chunk 2"]
        embeddings = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
        bulk_insert_chunks(
            conn=mock_conn, shadow_table="pgsemantic_embeddings_articles",
            row_id="42", chunks=chunks, embeddings=embeddings,
            source_column="body", model_name="all-MiniLM-L6-v2", schema="public",
        )
        calls = mock_conn.execute.call_args_list
        assert len(calls) >= 4  # 1 delete + 3 inserts
        mock_conn.commit.assert_called_once()
