"""Tests for cross-table unified search."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from pgsemantic.config import (
    DEFAULT_LOCAL_DIMENSIONS,
    DEFAULT_LOCAL_MODEL,
    HNSW_EF_CONSTRUCTION,
    HNSW_M,
    OPENAI_DIMENSIONS,
    OPENAI_MODEL,
    ProjectConfig,
    TableConfig,
)


def _make_table_config(
    table: str = "products",
    model: str = "local",
    model_name: str = DEFAULT_LOCAL_MODEL,
    dimensions: int = DEFAULT_LOCAL_DIMENSIONS,
    column: str = "description",
    schema: str = "public",
    storage_mode: str = "inline",
    shadow_table: str | None = None,
) -> TableConfig:
    return TableConfig(
        table=table,
        schema=schema,
        column=column,
        embedding_column="embedding",
        model=model,
        model_name=model_name,
        dimensions=dimensions,
        hnsw_m=HNSW_M,
        hnsw_ef_construction=HNSW_EF_CONSTRUCTION,
        applied_at="2026-03-30T00:00:00Z",
        storage_mode=storage_mode,
        shadow_table=shadow_table,
    )


class TestSearchAll:
    def test_merges_results_from_two_tables(self) -> None:
        """search_all should merge results from multiple tables sorted by similarity."""
        from pgsemantic.db.vectors import search_all

        config = ProjectConfig(
            version="0.3.0",
            tables=[
                _make_table_config(table="products"),
                _make_table_config(table="articles"),
            ],
        )

        mock_conn = MagicMock()
        mock_provider = MagicMock()
        mock_provider.embed_query.return_value = [0.1] * 384

        with patch("pgsemantic.db.vectors.search_similar") as mock_search:
            mock_search.side_effect = [
                [{"id": 1, "content": "Widget", "similarity": 0.8}],
                [{"id": 2, "content": "News", "similarity": 0.9}],
            ]

            results = search_all(
                conn=mock_conn,
                query="test",
                providers={"local": mock_provider},
                project_config=config,
                limit=10,
            )

        assert len(results) == 2
        assert results[0]["similarity"] == 0.9
        assert results[0]["_source_table"] == "articles"
        assert results[1]["similarity"] == 0.8
        assert results[1]["_source_table"] == "products"

    def test_respects_global_limit(self) -> None:
        """search_all should return at most `limit` results total."""
        from pgsemantic.db.vectors import search_all

        config = ProjectConfig(
            version="0.3.0",
            tables=[
                _make_table_config(table="products"),
                _make_table_config(table="articles"),
            ],
        )

        mock_conn = MagicMock()
        mock_provider = MagicMock()
        mock_provider.embed_query.return_value = [0.1] * 384

        with patch("pgsemantic.db.vectors.search_similar") as mock_search:
            mock_search.side_effect = [
                [{"id": i, "content": f"P{i}", "similarity": 0.9 - i * 0.01} for i in range(5)],
                [{"id": i, "content": f"A{i}", "similarity": 0.85 - i * 0.01} for i in range(5)],
            ]

            results = search_all(
                conn=mock_conn,
                query="test",
                providers={"local": mock_provider},
                project_config=config,
                limit=3,
            )

        assert len(results) == 3

    def test_different_models_embed_once_each(self) -> None:
        """search_all should call embed_query once per unique model, not per table."""
        from pgsemantic.db.vectors import search_all

        config = ProjectConfig(
            version="0.3.0",
            tables=[
                _make_table_config(table="products", model="local"),
                _make_table_config(table="articles", model="local"),
                _make_table_config(
                    table="docs", model="openai",
                    model_name=OPENAI_MODEL, dimensions=OPENAI_DIMENSIONS,
                ),
            ],
        )

        mock_conn = MagicMock()
        mock_local = MagicMock()
        mock_local.embed_query.return_value = [0.1] * 384
        mock_openai = MagicMock()
        mock_openai.embed_query.return_value = [0.2] * 1536

        with patch("pgsemantic.db.vectors.search_similar", return_value=[]):
            search_all(
                conn=mock_conn,
                query="test",
                providers={"local": mock_local, "openai": mock_openai},
                project_config=config,
                limit=10,
            )

        mock_local.embed_query.assert_called_once_with("test")
        mock_openai.embed_query.assert_called_once_with("test")

    def test_empty_config_returns_empty(self) -> None:
        """search_all with no configured tables should return empty list."""
        from pgsemantic.db.vectors import search_all

        config = ProjectConfig(version="0.3.0", tables=[])
        mock_conn = MagicMock()

        results = search_all(
            conn=mock_conn,
            query="test",
            providers={},
            project_config=config,
            limit=10,
        )

        assert results == []

    def test_tags_results_with_source_table_and_schema(self) -> None:
        """Each result should include _source_table and _source_schema."""
        from pgsemantic.db.vectors import search_all

        config = ProjectConfig(
            version="0.3.0",
            tables=[
                _make_table_config(table="products", schema="inventory"),
            ],
        )

        mock_conn = MagicMock()
        mock_provider = MagicMock()
        mock_provider.embed_query.return_value = [0.1] * 384

        with patch("pgsemantic.db.vectors.search_similar") as mock_search:
            mock_search.return_value = [
                {"id": 1, "content": "Widget", "similarity": 0.8},
            ]

            results = search_all(
                conn=mock_conn,
                query="test",
                providers={"local": mock_provider},
                project_config=config,
                limit=10,
            )

        assert results[0]["_source_table"] == "products"
        assert results[0]["_source_schema"] == "inventory"

    def test_skips_tables_with_missing_provider(self) -> None:
        """Tables whose model has no provider should be skipped gracefully."""
        from pgsemantic.db.vectors import search_all

        config = ProjectConfig(
            version="0.3.0",
            tables=[
                _make_table_config(table="products", model="local"),
                _make_table_config(
                    table="docs", model="openai",
                    model_name=OPENAI_MODEL, dimensions=OPENAI_DIMENSIONS,
                ),
            ],
        )

        mock_conn = MagicMock()
        mock_local = MagicMock()
        mock_local.embed_query.return_value = [0.1] * 384

        # Only provide "local", not "openai"
        with patch("pgsemantic.db.vectors.search_similar") as mock_search:
            mock_search.return_value = [
                {"id": 1, "content": "Widget", "similarity": 0.8},
            ]

            results = search_all(
                conn=mock_conn,
                query="test",
                providers={"local": mock_local},
                project_config=config,
                limit=10,
            )

        # Only the "local" table should have been searched
        assert len(results) == 1
        assert results[0]["_source_table"] == "products"
        mock_search.assert_called_once()  # only one call, not two
