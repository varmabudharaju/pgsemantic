"""Tests for natural language database Q&A MCP tools."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestGetSchemaContext:
    @patch("pgsemantic.mcp_server.server.get_connection")
    @patch("pgsemantic.mcp_server.server.load_settings")
    def test_returns_table_schema(self, mock_settings, mock_conn_ctx):
        mock_settings.return_value = MagicMock(database_url="postgresql://test@localhost/db")

        mock_conn = MagicMock()
        mock_conn_ctx.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn_ctx.return_value.__exit__ = MagicMock(return_value=False)

        # Mock get_user_tables
        with patch("pgsemantic.mcp_server.server.get_user_tables") as mock_tables:
            mock_tables.return_value = [
                {"table_name": "articles", "table_schema": "public"},
            ]

            # Mock columns query
            mock_cols = MagicMock()
            mock_cols.fetchall.return_value = [
                {"column_name": "id", "data_type": "integer", "is_nullable": "NO"},
                {"column_name": "title", "data_type": "text", "is_nullable": "YES"},
            ]

            # Mock sample rows query
            mock_samples = MagicMock()
            mock_samples.fetchall.return_value = [
                {"id": 1, "title": "Test Article"},
            ]

            mock_conn.execute.side_effect = [mock_cols, mock_samples]

            with patch("pgsemantic.mcp_server.server.get_row_count_estimate", return_value=100):
                from pgsemantic.mcp_server.server import get_schema_context
                result = get_schema_context()

        assert len(result["tables"]) == 1
        assert result["tables"][0]["name"] == "articles"
        assert len(result["tables"][0]["columns"]) == 2

    @patch("pgsemantic.mcp_server.server.get_connection")
    @patch("pgsemantic.mcp_server.server.load_settings")
    def test_filters_internal_tables(self, mock_settings, mock_conn_ctx):
        mock_settings.return_value = MagicMock(database_url="postgresql://test@localhost/db")

        mock_conn = MagicMock()
        mock_conn_ctx.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn_ctx.return_value.__exit__ = MagicMock(return_value=False)

        with patch("pgsemantic.mcp_server.server.get_user_tables") as mock_tables:
            mock_tables.return_value = [
                {"table_name": "articles", "table_schema": "public"},
                {"table_name": "pgvector_setup_queue", "table_schema": "public"},
                {"table_name": "pgsemantic_embeddings_articles", "table_schema": "public"},
            ]

            mock_cols = MagicMock()
            mock_cols.fetchall.return_value = [
                {"column_name": "id", "data_type": "integer", "is_nullable": "NO"},
            ]
            mock_samples = MagicMock()
            mock_samples.fetchall.return_value = []
            mock_conn.execute.side_effect = [mock_cols, mock_samples]

            with patch("pgsemantic.mcp_server.server.get_row_count_estimate", return_value=10):
                from pgsemantic.mcp_server.server import get_schema_context
                result = get_schema_context()

        assert len(result["tables"]) == 1
        assert result["tables"][0]["name"] == "articles"


class TestExecuteSafeSql:
    def test_rejects_insert(self):
        from mcp.server.fastmcp.exceptions import ToolError

        from pgsemantic.mcp_server.server import execute_safe_sql

        with (
            patch("pgsemantic.mcp_server.server._get_db_url", return_value="postgresql://test"),
            pytest.raises(ToolError, match="Only SELECT"),
        ):
            execute_safe_sql("INSERT INTO articles VALUES (1, 'test')")

    def test_rejects_drop(self):
        from mcp.server.fastmcp.exceptions import ToolError

        from pgsemantic.mcp_server.server import execute_safe_sql

        with (
            patch("pgsemantic.mcp_server.server._get_db_url", return_value="postgresql://test"),
            pytest.raises(ToolError, match="forbidden keywords"),
        ):
            execute_safe_sql("SELECT 1; DROP TABLE articles")

    def test_rejects_delete(self):
        from mcp.server.fastmcp.exceptions import ToolError

        from pgsemantic.mcp_server.server import execute_safe_sql

        with (
            patch("pgsemantic.mcp_server.server._get_db_url", return_value="postgresql://test"),
            pytest.raises(ToolError, match="Only SELECT"),
        ):
            execute_safe_sql("DELETE FROM articles WHERE id = 1")

    @patch("pgsemantic.mcp_server.server.get_connection")
    @patch("pgsemantic.mcp_server.server._get_db_url")
    def test_executes_valid_select(self, mock_url, mock_conn_ctx):
        mock_url.return_value = "postgresql://test"

        mock_conn = MagicMock()
        mock_conn_ctx.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn_ctx.return_value.__exit__ = MagicMock(return_value=False)

        mock_result = MagicMock()
        mock_result.fetchmany.return_value = [{"count": "42"}]
        mock_result.description = [("count",)]

        # Three execute calls: SET TRANSACTION, SET timeout, the query
        mock_conn.execute.side_effect = [MagicMock(), MagicMock(), mock_result]

        from pgsemantic.mcp_server.server import execute_safe_sql
        result = execute_safe_sql("SELECT COUNT(*) as count FROM articles")

        assert result["row_count"] == 1
        assert result["rows"][0]["count"] == "42"
        assert result["truncated"] is False

    def test_appends_limit_if_missing(self):
        from pgsemantic.mcp_server.server import execute_safe_sql

        with (
            patch("pgsemantic.mcp_server.server._get_db_url", return_value="postgresql://test"),
            patch("pgsemantic.mcp_server.server.get_connection") as mock_conn_ctx,
        ):
            mock_conn = MagicMock()
            mock_conn_ctx.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_conn_ctx.return_value.__exit__ = MagicMock(return_value=False)

            mock_result = MagicMock()
            mock_result.fetchmany.return_value = []
            mock_result.description = []
            mock_conn.execute.side_effect = [MagicMock(), MagicMock(), mock_result]

            execute_safe_sql("SELECT * FROM articles")

            # Check the actual SQL executed (3rd call)
            actual_sql = mock_conn.execute.call_args_list[2][0][0]
            assert "LIMIT 100" in actual_sql
