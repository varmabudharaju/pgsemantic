"""Tests for the migrate command."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from pgsemantic.config import (
    DEFAULT_LOCAL_DIMENSIONS,
    DEFAULT_LOCAL_MODEL,
    HNSW_EF_CONSTRUCTION,
    HNSW_M,
    ProjectConfig,
    TableConfig,
)


def _make_table_config(model: str = "local") -> TableConfig:
    return TableConfig(
        table="articles",
        schema="public",
        column="body",
        embedding_column="embedding",
        model=model,
        model_name=DEFAULT_LOCAL_MODEL if model == "local" else "text-embedding-3-small",
        dimensions=DEFAULT_LOCAL_DIMENSIONS if model == "local" else 1536,
        hnsw_m=HNSW_M,
        hnsw_ef_construction=HNSW_EF_CONSTRUCTION,
        applied_at="2026-03-30T00:00:00Z",
        storage_mode="external",
        shadow_table="pgsemantic_embeddings_articles",
    )


class TestMigrateCommand:
    @patch("pgsemantic.commands.migrate.save_project_config")
    @patch("pgsemantic.commands.migrate.get_connection")
    @patch("pgsemantic.commands.migrate.get_provider")
    @patch("pgsemantic.commands.migrate.load_project_config")
    @patch("pgsemantic.commands.migrate.load_settings")
    def test_same_model_exits_early(
        self, mock_settings, mock_config, mock_provider, mock_conn, mock_save
    ):
        from typer.testing import CliRunner

        from pgsemantic.cli import app

        mock_settings.return_value = MagicMock(
            database_url="postgresql://test@localhost/db",
            openai_api_key=None,
            ollama_base_url="http://localhost:11434",
        )
        mock_config.return_value = ProjectConfig(
            version="0.3.0",
            tables=[_make_table_config("local")],
        )

        runner = CliRunner()
        result = runner.invoke(app, ["migrate", "--table", "articles", "--model", "local"])
        assert result.exit_code == 0
        assert "already uses" in result.output

    @patch("pgsemantic.commands.migrate.save_project_config")
    @patch("pgsemantic.commands.migrate.get_connection")
    @patch("pgsemantic.commands.migrate.get_provider")
    @patch("pgsemantic.commands.migrate.load_project_config")
    @patch("pgsemantic.commands.migrate.load_settings")
    def test_missing_table_exits_with_error(
        self, mock_settings, mock_config, mock_provider, mock_conn, mock_save
    ):
        from typer.testing import CliRunner

        from pgsemantic.cli import app

        mock_settings.return_value = MagicMock(
            database_url="postgresql://test@localhost/db",
        )
        mock_config.return_value = ProjectConfig(version="0.3.0", tables=[])

        runner = CliRunner()
        result = runner.invoke(app, ["migrate", "--table", "nonexistent", "--model", "openai"])
        assert result.exit_code == 1

    @patch("pgsemantic.commands.migrate.save_project_config")
    @patch("pgsemantic.commands.migrate.get_connection")
    @patch("pgsemantic.commands.migrate.get_provider")
    @patch("pgsemantic.commands.migrate.load_project_config")
    @patch("pgsemantic.commands.migrate.load_settings")
    def test_no_config_exits_with_error(
        self, mock_settings, mock_config, mock_provider, mock_conn, mock_save
    ):
        from typer.testing import CliRunner

        from pgsemantic.cli import app

        mock_settings.return_value = MagicMock(
            database_url="postgresql://test@localhost/db",
        )
        mock_config.return_value = None

        runner = CliRunner()
        result = runner.invoke(app, ["migrate", "--table", "articles", "--model", "openai"])
        assert result.exit_code == 1
        assert "No .pgsemantic.json" in result.output

    @patch("pgsemantic.commands.migrate.save_project_config")
    @patch("pgsemantic.commands.migrate.get_connection")
    @patch("pgsemantic.commands.migrate.get_provider")
    @patch("pgsemantic.commands.migrate.load_project_config")
    @patch("pgsemantic.commands.migrate.load_settings")
    def test_no_database_url_exits_with_error(
        self, mock_settings, mock_config, mock_provider, mock_conn, mock_save
    ):
        from typer.testing import CliRunner

        from pgsemantic.cli import app

        mock_settings.return_value = MagicMock(database_url=None)

        runner = CliRunner()
        result = runner.invoke(app, ["migrate", "--table", "articles", "--model", "openai"])
        assert result.exit_code == 1
        assert "No database URL" in result.output

    def test_migration_suffix_constant(self):
        from pgsemantic.commands.migrate import MIGRATION_SUFFIX

        assert MIGRATION_SUFFIX == "_migration"

    @patch("pgsemantic.commands.migrate.save_project_config")
    @patch("pgsemantic.commands.migrate.get_connection")
    @patch("pgsemantic.commands.migrate.get_provider")
    @patch("pgsemantic.commands.migrate.load_project_config")
    @patch("pgsemantic.commands.migrate.load_settings")
    def test_model_validation_failure_exits(
        self, mock_settings, mock_config, mock_provider, mock_conn, mock_save
    ):
        from typer.testing import CliRunner

        from pgsemantic.cli import app

        mock_settings.return_value = MagicMock(
            database_url="postgresql://test@localhost/db",
            openai_api_key="fake-key",
            ollama_base_url="http://localhost:11434",
        )
        mock_config.return_value = ProjectConfig(
            version="0.3.0",
            tables=[_make_table_config("local")],
        )
        mock_provider.side_effect = ValueError("Unknown provider")

        runner = CliRunner()
        result = runner.invoke(app, ["migrate", "--table", "articles", "--model", "bad-model"])
        assert result.exit_code == 1
        assert "Model validation failed" in result.output
