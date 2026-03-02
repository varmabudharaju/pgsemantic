"""Tests for MCP server — provider caching, tool error handling, config lookup."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pgsemantic.config import (
    DEFAULT_LOCAL_DIMENSIONS,
    DEFAULT_LOCAL_MODEL,
    HNSW_EF_CONSTRUCTION,
    HNSW_M,
    ProjectConfig,
    TableConfig,
)


class TestProviderCache:
    def test_cache_returns_same_instance(self) -> None:
        """Provider cache should return the same provider for the same key."""
        from pgsemantic.mcp_server.server import _provider_cache

        _provider_cache.clear()

        # Mock the LocalProvider class directly
        mock_provider = MagicMock()
        mock_provider.config = MagicMock(
            model_name="all-MiniLM-L6-v2", dimensions=384
        )
        with patch("pgsemantic.embeddings.get_provider", return_value=mock_provider):
            from pgsemantic.mcp_server.server import _get_or_create_provider

            p1 = _get_or_create_provider("local")
            p2 = _get_or_create_provider("local")
            assert p1 is p2

        _provider_cache.clear()

    def test_cache_different_keys(self) -> None:
        """Different model names should produce different cache entries."""
        from pgsemantic.mcp_server.server import _provider_cache

        _provider_cache.clear()

        mock_local = MagicMock()
        mock_openai = MagicMock()

        def mock_get_provider(name: str, **kwargs: object) -> MagicMock:
            return mock_local if name == "local" else mock_openai

        with patch("pgsemantic.embeddings.get_provider", side_effect=mock_get_provider):
            from pgsemantic.mcp_server.server import _get_or_create_provider

            p_local = _get_or_create_provider("local")
            p_openai = _get_or_create_provider("openai", api_key="sk-test")

        assert p_local is not p_openai

        _provider_cache.clear()


class TestGetTableConfig:
    def _make_project_config(self) -> ProjectConfig:
        return ProjectConfig(
            version="0.1.0",
            tables=[
                TableConfig(
                    table="products",
                    schema="public",
                    column="description",
                    embedding_column="embedding",
                    model="local",
                    model_name=DEFAULT_LOCAL_MODEL,
                    dimensions=DEFAULT_LOCAL_DIMENSIONS,
                    hnsw_m=HNSW_M,
                    hnsw_ef_construction=HNSW_EF_CONSTRUCTION,
                    applied_at="2026-03-01T00:00:00Z",
                ),
            ],
        )

    @patch("pgsemantic.mcp_server.server.load_project_config")
    @patch("pgsemantic.mcp_server.server.load_settings")
    def test_returns_config_for_valid_table(
        self,
        mock_settings: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Should return config tuple for a configured table."""
        mock_settings.return_value = MagicMock(
            database_url="postgresql://test:test@localhost/testdb",
            openai_api_key=None,
            ollama_base_url="http://localhost:11434",
        )
        mock_config.return_value = self._make_project_config()

        from pgsemantic.mcp_server.server import _get_table_config

        result = _get_table_config("products")
        db_url, tc, api_key, ollama_url = result
        assert db_url == "postgresql://test:test@localhost/testdb"
        assert tc.column == "description"
        assert tc.model == "local"
        assert tc.model_name == DEFAULT_LOCAL_MODEL
        assert tc.storage_mode == "inline"
        assert tc.source_columns == ["description"]

    @patch("pgsemantic.mcp_server.server.load_project_config")
    @patch("pgsemantic.mcp_server.server.load_settings")
    def test_raises_for_missing_table(
        self,
        mock_settings: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Should raise ToolError for a table not in config."""
        from mcp.server.fastmcp.exceptions import ToolError

        mock_settings.return_value = MagicMock(
            database_url="postgresql://test:test@localhost/testdb",
        )
        mock_config.return_value = self._make_project_config()

        from pgsemantic.mcp_server.server import _get_table_config

        with pytest.raises(ToolError, match="not found"):
            _get_table_config("nonexistent_table")

    @patch("pgsemantic.mcp_server.server.load_project_config")
    @patch("pgsemantic.mcp_server.server.load_settings")
    def test_raises_for_missing_database_url(
        self,
        mock_settings: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Should raise ToolError when DATABASE_URL is not set."""
        from mcp.server.fastmcp.exceptions import ToolError

        mock_settings.return_value = MagicMock(database_url=None)
        mock_config.return_value = self._make_project_config()

        from pgsemantic.mcp_server.server import _get_table_config

        with pytest.raises(ToolError, match="DATABASE_URL"):
            _get_table_config("products")

    @patch("pgsemantic.mcp_server.server.load_project_config")
    @patch("pgsemantic.mcp_server.server.load_settings")
    def test_raises_for_no_config(
        self,
        mock_settings: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        """Should raise ToolError when .pgsemantic.json doesn't exist."""
        from mcp.server.fastmcp.exceptions import ToolError

        mock_settings.return_value = MagicMock(
            database_url="postgresql://test:test@localhost/testdb",
        )
        mock_config.return_value = None

        from pgsemantic.mcp_server.server import _get_table_config

        with pytest.raises(ToolError, match="No pgsemantic config"):
            _get_table_config("products")


class TestMcpToolErrorHandling:
    """Test that MCP tools wrap errors correctly."""

    @patch("pgsemantic.mcp_server.server._get_table_config")
    def test_semantic_search_wraps_exception(
        self, mock_config: MagicMock
    ) -> None:
        """semantic_search should convert PgvectorSetupError to ToolError."""
        from mcp.server.fastmcp.exceptions import ToolError

        from pgsemantic.exceptions import TableNotWatchedError

        mock_config.side_effect = ToolError("Table not found")

        from pgsemantic.mcp_server.server import semantic_search

        with pytest.raises(ToolError):
            semantic_search(query="test", table="missing")

    @patch("pgsemantic.mcp_server.server._get_table_config")
    def test_get_embedding_status_wraps_exception(
        self, mock_config: MagicMock
    ) -> None:
        """get_embedding_status should convert errors to ToolError."""
        from mcp.server.fastmcp.exceptions import ToolError

        mock_config.side_effect = ToolError("Not configured")

        from pgsemantic.mcp_server.server import get_embedding_status

        with pytest.raises(ToolError):
            get_embedding_status(table="missing")
